#!/usr/bin/env python3
"""Phase 2.8: Fix scheduler cycle detection to return NULL instead of partial schedule."""
import os, sys, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_harness"))
import safe_edit as se

FILE = "src/runtime/lancius_scheduler.c"
MARKER = "v12R1-208"

def main():
    print("[208] scheduler cycle detection fix")
    content = se.read(se.p(FILE))
    if MARKER in content:
        print(f"  [skip] {FILE}: already applied")
        return

    # Replace the cycle detection block
    old_cycle = (
        "        if (q_tail == 0) {\n"
        "            fprintf(stderr, \"[SCHEDULER FATAL] Cycle or disconnect! Processed %u / %u\\n\", processed, g->node_count);\n"
        "            break;\n"
        "        }"
    )
    new_cycle = (
        "        if (q_tail == 0) {\n"
        f"            /* {MARKER}: cycle detected - destroy partial schedule and return NULL */\n"
        "            fprintf(stderr, \"[SCHEDULER FATAL] Cycle or disconnect! Processed %u / %u\\n\", processed, g->node_count);\n"
        "            lancius_set_error(LANCIUS_ERROR_GRAPH_CYCLE);\n"
        "            /* Destroy partial schedule */\n"
        "            for (uint32_t w = 0; w < sched->wave_count; w++) free(sched->waves[w].nodes);\n"
        "            free(sched->waves);\n"
        "            free(sched);\n"
        "            free(in_degree); free(queue);\n"
        "            return NULL;\n"
        "        }"
    )
    content = se.replace_once(content, old_cycle, new_cycle)

    # Also add allocation failure checks for the initial allocations
    old_alloc = (
        "    lancius_schedule* sched = (lancius_schedule*)calloc(1, sizeof(lancius_schedule));\n"
        "    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));\n"
        "    lancius_node** queue = (lancius_node**)malloc(sizeof(lancius_node*) * g->node_count);"
    )
    new_alloc = (
        f"    /* {MARKER}: check all initial allocations */\n"
        "    lancius_schedule* sched = (lancius_schedule*)calloc(1, sizeof(lancius_schedule));\n"
        "    uint32_t* in_degree = (uint32_t*)calloc(g->next_id, sizeof(uint32_t));\n"
        "    lancius_node** queue = (lancius_node**)malloc(sizeof(lancius_node*) * g->node_count);\n"
        "    if (!sched || !in_degree || !queue) {\n"
        "        free(sched); free(in_degree); free(queue);\n"
        "        lancius_set_error(LANCIUS_ERROR_OOM);\n"
        "        return NULL;\n"
        "    }"
    )
    content = se.replace_once(content, old_alloc, new_alloc)

    # Also check the wave allocation
    old_wave_alloc = (
        "        if (sched->wave_count >= wave_cap) {\n"
        "            wave_cap *= 2;\n"
        "            sched->waves = (lancius_wave*)realloc(sched->waves, sizeof(lancius_wave) * wave_cap);\n"
        "        }"
    )
    new_wave_alloc = (
        f"        /* {MARKER}: check realloc failure */\n"
        "        if (sched->wave_count >= wave_cap) {\n"
        "            wave_cap *= 2;\n"
        "            lancius_wave* new_waves = (lancius_wave*)realloc(sched->waves, sizeof(lancius_wave) * wave_cap);\n"
        "            if (!new_waves) {\n"
        "                for (uint32_t w = 0; w < sched->wave_count; w++) free(sched->waves[w].nodes);\n"
        "                free(sched->waves); free(sched);\n"
        "                free(in_degree); free(queue);\n"
        "                lancius_set_error(LANCIUS_ERROR_OOM);\n"
        "                return NULL;\n"
        "            }\n"
        "            sched->waves = new_waves;\n"
        "        }"
    )
    content = se.replace_once(content, old_wave_alloc, new_wave_alloc)

    se.backup(se.p(FILE))
    se.write(se.p(FILE), content)
    print(f"  [edit] {FILE}")
    print("[208] done")

if __name__ == "__main__":
    main()
