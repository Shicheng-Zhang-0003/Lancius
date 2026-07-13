#ifndef V8_THREADPOOL_H
#define V8_THREADPOOL_H
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

typedef void (*v8_task_fn)(void*);

typedef struct {
    v8_task_fn fn;
    void* arg;
} v8_task;

typedef struct v8_pool v8_pool;

v8_pool* v8_pool_create(int num_threads);
void v8_pool_submit(v8_pool* pool, v8_task_fn fn, void* arg);
void v8_pool_wait(v8_pool* pool);
void v8_pool_destroy(v8_pool* pool);
#endif
