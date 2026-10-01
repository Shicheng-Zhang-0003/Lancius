#include "lancius/lancius_threadpool.h"
#include "lancius/lancius_error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>

/* v12R1 fix: named constants (were magic numbers). */
#define LANCIUS_POOL_DEFAULT_THREADS 4
#define LANCIUS_POOL_MAX_THREADS 256
#define LANCIUS_POOL_QUEUE_INIT_CAP 1024
#define LANCIUS_POOL_QUEUE_MAX_CAP 1000000

struct lancius_pool {
    pthread_t* threads;
    int num_threads;
    lancius_task* queue;
    int queue_cap;
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_wait;
    bool shutdown;
    int active_tasks;
};

static void* worker_loop(void* arg) {
    lancius_pool* pool = (lancius_pool*)arg;
    if (!pool) return NULL;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_work, &pool->mutex);
        }
        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        lancius_task task = pool->queue[pool->head];
        pool->head = (pool->head + 1) % pool->queue_cap;
        pool->count--;
        pthread_mutex_unlock(&pool->mutex);

        if (task.fn) task.fn(task.arg);

        pthread_mutex_lock(&pool->mutex);
        pool->active_tasks--;
        if (pool->active_tasks == 0 && pool->count == 0) {
            pthread_cond_signal(&pool->cond_wait);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

lancius_pool* lancius_pool_create(int num_threads) {
    if (num_threads <= 0) num_threads = LANCIUS_POOL_DEFAULT_THREADS;
    if (num_threads > LANCIUS_POOL_MAX_THREADS) num_threads = LANCIUS_POOL_MAX_THREADS;
    lancius_pool* pool = (lancius_pool*)calloc(1, sizeof(lancius_pool));
    /* Despot truth: creation failure sets OOM (was silent NULL). */
    if (!pool) { lancius_set_error(LANCIUS_ERROR_OOM); return NULL; }
    pool->num_threads = num_threads;
    pool->queue_cap = LANCIUS_POOL_QUEUE_INIT_CAP;
    pool->queue = (lancius_task*)malloc(sizeof(lancius_task) * (size_t)pool->queue_cap);
    pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)num_threads);
    if (!pool->queue || !pool->threads) { lancius_set_error(LANCIUS_ERROR_OOM); free(pool->queue); free(pool->threads); free(pool); return NULL; }
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) { lancius_set_error(LANCIUS_ERROR_INTERNAL); free(pool->queue); free(pool->threads); free(pool); return NULL; }
    if (pthread_cond_init(&pool->cond_work, NULL) != 0) { lancius_set_error(LANCIUS_ERROR_INTERNAL); pthread_mutex_destroy(&pool->mutex); free(pool->queue); free(pool->threads); free(pool); return NULL; }
    if (pthread_cond_init(&pool->cond_wait, NULL) != 0) { lancius_set_error(LANCIUS_ERROR_INTERNAL); pthread_cond_destroy(&pool->cond_work); pthread_mutex_destroy(&pool->mutex); free(pool->queue); free(pool->threads); free(pool); return NULL; }
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_loop, pool) != 0) {
            /* Tear down already-started threads. */
            lancius_set_error(LANCIUS_ERROR_INTERNAL);
            pthread_mutex_lock(&pool->mutex);
            pool->shutdown = true;
            pthread_cond_broadcast(&pool->cond_work);
            pthread_mutex_unlock(&pool->mutex);
            for (int j = 0; j < i; j++) pthread_join(pool->threads[j], NULL);
            pthread_cond_destroy(&pool->cond_wait);
            pthread_cond_destroy(&pool->cond_work);
            pthread_mutex_destroy(&pool->mutex);
            free(pool->queue); free(pool->threads); free(pool);
            return NULL;
        }
    }
    return pool;
}

void lancius_pool_submit(lancius_pool* pool, lancius_task_fn fn, void* arg) {
    if (!pool || !fn) return;
    pthread_mutex_lock(&pool->mutex);
    /* Despot truth: queue-full inline fallback ran on the submitter thread
     * concurrently with workers (scratch arenas are not thread-safe) and was
     * invisible to pool_wait. Grow the queue instead; reject after shutdown. */
    if (pool->shutdown) { lancius_set_error(LANCIUS_ERROR_GRAPH_INVALID); pthread_mutex_unlock(&pool->mutex); return; }
    if (pool->count >= pool->queue_cap) {
        int new_cap = pool->queue_cap * 2;
        lancius_task *nq;
        if (new_cap <= 0 || new_cap > LANCIUS_POOL_QUEUE_MAX_CAP) { lancius_set_error(LANCIUS_ERROR_LIMIT); pthread_mutex_unlock(&pool->mutex); return; }
        /* Despot V6 truth: malloc+linearize+free (was realloc then UAF read
         * of freed pool->queue, plus leak of nq on tmp-OOM). */
        nq = (lancius_task*)malloc(sizeof(lancius_task) * (size_t)new_cap);
        if (!nq) { lancius_set_error(LANCIUS_ERROR_OOM); pthread_mutex_unlock(&pool->mutex); return; }
        /* Re-linearize ring into the grown buffer. */
        {
            int i;
            for (i = 0; i < pool->count; i++)
                nq[i] = pool->queue[(pool->head + i) % pool->queue_cap];
        }
        free(pool->queue);
        pool->queue = nq;
        pool->head = 0;
        pool->tail = pool->count;
        pool->queue_cap = new_cap;
    }
    pool->queue[pool->tail].fn = fn;
    pool->queue[pool->tail].arg = arg;
    pool->tail = (pool->tail + 1) % pool->queue_cap;
    pool->count++;
    pool->active_tasks++;
    pthread_cond_signal(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);
}

int lancius_pool_wait(lancius_pool* pool, uint32_t timeout_ms) {
    if (!pool) return -1;
    pthread_mutex_lock(&pool->mutex);
    if (timeout_ms == 0) {
        /* Infinite wait (legacy behavior). */
        while (pool->active_tasks > 0 || pool->count > 0) {
            pthread_cond_wait(&pool->cond_wait, &pool->mutex);
        }
        pthread_mutex_unlock(&pool->mutex);
        return 0;
    }
    /* v12R1 fix: bounded wait — an infinite cond_wait could hang the caller
     * forever if a task deadlocks or a worker dies. */
    struct timespec deadline;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }
    deadline.tv_sec += (time_t)(timeout_ms / 1000u);
    deadline.tv_nsec += (long)(timeout_ms % 1000u) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    int rc = 0;
    while ((pool->active_tasks > 0 || pool->count > 0) && rc == 0) {
        rc = pthread_cond_timedwait(&pool->cond_wait, &pool->mutex, &deadline);
    }
    pthread_mutex_unlock(&pool->mutex);
    if (rc == 0) return 0;
    /* ETIMEDOUT (or any other wait error) — caller decides how to proceed. */
    return -1;
}

void lancius_pool_destroy(lancius_pool* pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_work);
    pthread_cond_destroy(&pool->cond_wait);
    free(pool->queue);
    free(pool->threads);
    free(pool);
}
