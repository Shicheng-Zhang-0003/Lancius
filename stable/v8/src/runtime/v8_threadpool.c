#include "basis/v8_threadpool.h"
#include <stdlib.h>
#include <stdio.h>

struct v8_pool {
    pthread_t* threads;
    int num_threads;
    v8_task* queue;
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
    v8_pool* pool = (v8_pool*)arg;
    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_work, &pool->mutex);
        }
        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        v8_task task = pool->queue[pool->head];
        pool->head = (pool->head + 1) % pool->queue_cap;
        pool->count--;
        pthread_mutex_unlock(&pool->mutex);

        task.fn(task.arg);

        pthread_mutex_lock(&pool->mutex);
        pool->active_tasks--;
        if (pool->active_tasks == 0 && pool->count == 0) {
            pthread_cond_signal(&pool->cond_wait);
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    return NULL;
}

v8_pool* v8_pool_create(int num_threads) {
    if (num_threads <= 0) num_threads = 4;
    v8_pool* pool = (v8_pool*)calloc(1, sizeof(v8_pool));
    pool->num_threads = num_threads;
    pool->queue_cap = 1024;
    pool->queue = (v8_task*)malloc(sizeof(v8_task) * pool->queue_cap);
    pool->threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)num_threads);
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_work, NULL);
    pthread_cond_init(&pool->cond_wait, NULL);
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&pool->threads[i], NULL, worker_loop, pool);
    }
    return pool;
}

void v8_pool_submit(v8_pool* pool, v8_task_fn fn, void* arg) {
    pthread_mutex_lock(&pool->mutex);
    if (pool->count >= pool->queue_cap) {
        pthread_mutex_unlock(&pool->mutex);
        fn(arg); // Fallback synchronous execution to prevent deadlock
        return;
    }
    pool->queue[pool->tail].fn = fn;
    pool->queue[pool->tail].arg = arg;
    pool->tail = (pool->tail + 1) % pool->queue_cap;
    pool->count++;
    pool->active_tasks++;
    pthread_cond_signal(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);
}

void v8_pool_wait(v8_pool* pool) {
    pthread_mutex_lock(&pool->mutex);
    while (pool->active_tasks > 0 || pool->count > 0) {
        pthread_cond_wait(&pool->cond_wait, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

void v8_pool_destroy(v8_pool* pool) {
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
