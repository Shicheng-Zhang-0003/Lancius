/**
 * @file lancius_threadpool.h
 * @brief Threadpool: POSIX pthreads worker queue for parallel wave execution.
 */
#ifndef LANCIUS_THREADPOOL_H
#define LANCIUS_THREADPOOL_H
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

typedef void (*lancius_task_fn)(void*);

typedef struct {
    lancius_task_fn fn;
    void* arg;
} lancius_task;

typedef struct lancius_pool lancius_pool;

lancius_pool* lancius_pool_create(int num_threads);
void lancius_pool_submit(lancius_pool* pool, lancius_task_fn fn, void* arg);

/*
 * Block until all submitted tasks have drained.
 * timeout_ms == 0 waits indefinitely; otherwise the wait aborts after
 * timeout_ms milliseconds. Returns 0 on success, -1 on timeout/error.
 */
int lancius_pool_wait(lancius_pool* pool, uint32_t timeout_ms);

void lancius_pool_destroy(lancius_pool* pool);
#endif
