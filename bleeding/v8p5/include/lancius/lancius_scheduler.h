#ifndef Lancius_LANCIUS_SCHEDULER_H
#define Lancius_LANCIUS_SCHEDULER_H
#include "lancius/lancius_ir.h"
#include "lancius/lancius_threadpool.h"

typedef struct { lancius_node** nodes; uint32_t node_count; } lancius_wave;
typedef struct { lancius_wave* waves; uint32_t wave_count; } lancius_schedule;

lancius_schedule* lancius_ir_schedule(lancius_graph* g);
void lancius_schedule_execute(lancius_schedule* schedule, lancius_arena* scratch);
void lancius_schedule_execute_parallel(lancius_schedule* schedule, lancius_arena* scratch, lancius_pool* pool);
void lancius_schedule_destroy(lancius_schedule* schedule);
size_t lancius_schedule_peak_memory(lancius_schedule* schedule);
#endif
