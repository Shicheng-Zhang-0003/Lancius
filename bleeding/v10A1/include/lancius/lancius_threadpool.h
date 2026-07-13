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
void lancius_pool_wait(lancius_pool* pool);
void lancius_pool_destroy(lancius_pool* pool);
#endif
