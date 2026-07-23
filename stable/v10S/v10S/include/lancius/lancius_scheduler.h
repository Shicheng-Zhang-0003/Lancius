/**
 * @file lancius_scheduler.h
 * @brief Wave Scheduler: Topological dependency grouping and parallel execution dispatch.
 */
#ifndef Lancius_LANCIUS_SCHEDULER_H
#define Lancius_LANCIUS_SCHEDULER_H
#include "lancius/lancius_ir.h"
#include "lancius/lancius_threadpool.h"

typedef struct { lancius_node** nodes; uint32_t node_count; } lancius_wave;
struct lancius_memory_plan;
typedef struct { lancius_wave* waves; uint32_t wave_count; struct lancius_memory_plan* plan; void* static_pool; } lancius_schedule;

lancius_schedule* lancius_ir_schedule(lancius_graph* g);
void lancius_schedule_execute(lancius_schedule* schedule, lancius_arena* scratch);
void lancius_schedule_execute_parallel(lancius_schedule* schedule, lancius_arena* scratch, lancius_pool* pool);
void lancius_schedule_destroy(lancius_schedule* schedule);
size_t lancius_schedule_peak_memory(lancius_schedule* schedule);

typedef struct {
    size_t peak_memory_bytes;
    size_t total_allocs;
    size_t tensor_count;
} lancius_liveness_profile;

lancius_liveness_profile lancius_analyze_liveness(lancius_graph* g);

void lancius_schedule_execute_static(lancius_schedule* schedule, void* flat_buffer);
void lancius_schedule_attach_pool(lancius_schedule* schedule, void* flat_buffer, struct lancius_memory_plan* plan);
#endif
