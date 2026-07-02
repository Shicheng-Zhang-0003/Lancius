#ifndef BASIS_V8_SCHEDULER_H
#define BASIS_V8_SCHEDULER_H
#include "basis/v8_ir.h"
#include "basis/v8_threadpool.h"

typedef struct { v8_node** nodes; uint32_t node_count; } v8_wave;
typedef struct { v8_wave* waves; uint32_t wave_count; } v8_schedule;

v8_schedule* v8_ir_schedule(v8_graph* g);
void v8_schedule_execute(v8_schedule* schedule, v8_arena* scratch);
void v8_schedule_execute_parallel(v8_schedule* schedule, v8_arena* scratch, v8_pool* pool);
void v8_schedule_destroy(v8_schedule* schedule);
#endif
