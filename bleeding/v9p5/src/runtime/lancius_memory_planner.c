#include "lancius/lancius_memory_planner.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

static inline size_t align32(size_t sz) {
    return (sz + 31) & ~(size_t)31;
}

// Helper to get byte size.
static size_t get_node_bytes(const lancius_node* n) {
    if (!n) return 0;
    size_t elems = 1;
    for (uint8_t i = 0; i < n->ndim; i++) {
        elems *= n->shape[i];
    }
    return align32(elems * sizeof(double)); // Assuming FP64
}

typedef struct {
    uint32_t node_id;
    uint32_t birth_wave;
    uint32_t death_wave;
    size_t size_bytes;
    size_t offset; // Tracks where this tensor lives in the flat buffer
} lancius_interval;

typedef struct free_block {
    size_t offset;
    size_t size;
    struct free_block* next;
} free_block;

lancius_memory_plan* lancius_build_memory_plan(lancius_schedule* sched, lancius_graph* g) {
    if (!sched || !g || sched->wave_count == 0) return NULL;

    uint32_t max_id = g->next_id;
    uint32_t* birth = (uint32_t*)malloc(max_id * sizeof(uint32_t));
    uint32_t* death = (uint32_t*)malloc(max_id * sizeof(uint32_t));

    for (uint32_t i = 0; i < max_id; i++) {
        birth[i] = UINT32_MAX;
        death[i] = 0;
    }

    // P0 RED TEAM FIX: Pin ALL terminal sink nodes to prevent backward pass memory overwrite
    // Any node that is not an input to another node is an output and must survive.
    uint32_t* is_input_to_others = (uint32_t*)calloc(max_id, sizeof(uint32_t));
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        for (uint32_t j = 0; j < n->input_count; j++) {
            if (n->inputs[j]) is_input_to_others[n->inputs[j]->id] = 1;
        }
    }


    // 1. Liveness Analysis
    for (uint32_t w = 0; w < sched->wave_count; w++) {
        for (uint32_t i = 0; i < sched->waves[w].node_count; i++) {
            lancius_node* n = sched->waves[w].nodes[i];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) {
                birth[n->id] = w;
            }
            for (uint32_t j = 0; j < n->input_count; j++) {
                uint32_t in_id = n->inputs[j]->id;
                if (birth[in_id] != UINT32_MAX) {
                    if (w > death[in_id]) death[in_id] = w;
                }
            }
        }
    }

    // 2. Build Intervals
    lancius_interval* intervals = (lancius_interval*)calloc(max_id, sizeof(lancius_interval));
    uint32_t num_intervals = 0;

    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        if (birth[n->id] != UINT32_MAX) {
            size_t sz = get_node_bytes(n);
            intervals[num_intervals].node_id = n->id;
            intervals[num_intervals].birth_wave = birth[n->id];
            intervals[num_intervals].death_wave = (death[n->id] > birth[n->id]) ? death[n->id] : birth[n->id];
            // P0 RED TEAM FIX: If this is ANY terminal sink node, pin it to the last wave!
            if (!is_input_to_others[n->id]) {
                intervals[num_intervals].death_wave = sched->wave_count;
            }
            intervals[num_intervals].size_bytes = sz;
            num_intervals++;
        }
    }

    // Sort by birth_wave
    for (uint32_t i = 1; i < num_intervals; i++) {
        lancius_interval key = intervals[i];
        int j = i - 1;
        while (j >= 0 && intervals[j].birth_wave > key.birth_wave) {
            intervals[j + 1] = intervals[j];
            j--;
        }
        intervals[j + 1] = key;
    }

    // 3. Linear Scan
    lancius_memory_plan* plan = (lancius_memory_plan*)calloc(1, sizeof(lancius_memory_plan));
    plan->offsets = (size_t*)calloc(max_id, sizeof(size_t));
    plan->is_pooled = (uint8_t*)calloc(max_id, sizeof(uint8_t));

    free_block* free_list = (free_block*)malloc(sizeof(free_block));
    free_list->offset = 0;
    free_list->size = SIZE_MAX;
    free_list->next = NULL;

    size_t peak_memory = 0;
    lancius_interval* active = (lancius_interval*)calloc(num_intervals, sizeof(lancius_interval));
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < num_intervals; i++) {
        lancius_interval* curr = &intervals[i];

        uint32_t new_active_count = 0;
        for (uint32_t a = 0; a < active_count; a++) {
            if (active[a].death_wave < curr->birth_wave) {
                free_block* fb = (free_block*)malloc(sizeof(free_block));
                fb->offset = active[a].offset;
                fb->size = active[a].size_bytes;
                fb->next = free_list;
                free_list = fb;
            } else {
                active[new_active_count++] = active[a];
            }
        }
        active_count = new_active_count;

        free_block* prev = NULL;
        free_block* curr_fb = free_list;
        bool found = false;

        while (curr_fb) {
            if (curr_fb->size >= curr->size_bytes) {
                plan->offsets[curr->node_id] = curr_fb->offset;
                plan->is_pooled[curr->node_id] = 1;

                size_t end_addr = curr_fb->offset + curr->size_bytes;
                if (end_addr > peak_memory) peak_memory = end_addr;

                if (curr_fb->size > curr->size_bytes) {
                    curr_fb->offset += curr->size_bytes;
                    curr_fb->size -= curr->size_bytes;
                } else {
                    if (prev) prev->next = curr_fb->next;
                    else free_list = curr_fb->next;
                    free(curr_fb);
                }
                found = true;
                break;
            }
            prev = curr_fb;
            curr_fb = curr_fb->next;
        }

        if (!found) {
            plan->offsets[curr->node_id] = peak_memory;
            plan->is_pooled[curr->node_id] = 1;
            peak_memory += curr->size_bytes;
        }

        active[active_count++] = *curr;
    }

    plan->peak_memory = peak_memory;

    free(birth); free(death); free(intervals); free(active); free(is_input_to_others);
    return plan;
}

void lancius_memory_plan_destroy(lancius_memory_plan* plan) {
    if (!plan) return;
    free(plan->offsets);
    free(plan->is_pooled);
    free(plan);
}
