/**
 * @file lancius_memory_planner.h
 * @brief Linear Scan Memory Planner: Liveness analysis and graph coloring for static flat-buffer pooling.
 */
#ifndef LANCIUS_MEMORY_PLANNER_H
#define LANCIUS_MEMORY_PLANNER_H

#include "lancius_ir.h"
#include "lancius_scheduler.h"
#include <stdint.h>
#include <stddef.h>

typedef struct lancius_memory_plan {
    size_t peak_memory;     // Total bytes required for the flat buffer
    size_t* offsets;        // Array of size g->next_id: offsets[node_id] = byte offset
    uint8_t* is_pooled;     // Array of size g->next_id: 1 if this node uses the pool
} lancius_memory_plan;

// Analyzes the scheduled waves and computes the linear scan memory plan
lancius_memory_plan* lancius_build_memory_plan(lancius_schedule* sched, lancius_graph* g);

// Frees the memory plan
void lancius_memory_plan_destroy(lancius_memory_plan* plan);

#endif
