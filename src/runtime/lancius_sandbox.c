#include "lancius/lancius_sandbox.h"
#include "lancius/lancius_scheduler.h"
#include "lancius/lancius_error.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int sandbox_scan_f64(const double *p, size_t n)
{
    size_t i;
    if (p == NULL) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        double v = p[i];
        if (isnan(v) || isinf(v)) {
            return -1;
        }
    }
    return 0;
}

static int sandbox_scan_f32(const float *p, size_t n)
{
    size_t i;
    if (p == NULL) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        float v = p[i];
        if (isnan(v) || isinf(v)) {
            return -1;
        }
    }
    return 0;
}

int lancius_sandbox_check_graph(const lancius_graph *g,
                                const lancius_sandbox_caps *caps)
{
    uint32_t i;
    size_t sum_bytes = 0;

    lancius_clear_error();

    if (g == NULL || caps == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return -1;
    }
    if (g->node_count > 0u && g->nodes == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return -1;
    }
    if (g->node_count > caps->max_nodes) {
        lancius_set_error(LANCIUS_ERROR_LIMIT);
        return -1;
    }
    if (g->node_count > caps->max_steps) {
        lancius_set_error(LANCIUS_ERROR_LIMIT);
        return -1;
    }

    for (i = 0; i < g->node_count; i++) {
        const lancius_node *n = g->nodes[i];
        size_t elems = 0;
        size_t bytes = 0;

        if (n == NULL) {
            lancius_set_error(LANCIUS_ERROR_NULL_PTR);
            return -1;
        }
        if (n->ndim > 4u) {
            lancius_set_error(LANCIUS_ERROR_INVALID_RANK);
            return -1;
        }
        if (!lancius_node_elements_checked(n, &elems)) {
            lancius_set_error(LANCIUS_ERROR_LIMIT);
            return -1;
        }
        if (!lancius_node_bytes_checked(n, &bytes)) {
            lancius_set_error(LANCIUS_ERROR_LIMIT);
            return -1;
        }
        if (isnan(n->attr_val) || isinf(n->attr_val)) {
            lancius_set_error(LANCIUS_ERROR_NUMERICAL);
            return -1;
        }
        if (bytes > SIZE_MAX - sum_bytes) {
            lancius_set_error(LANCIUS_ERROR_OVERFLOW);
            return -1;
        }
        sum_bytes += bytes;
    }

    if (g->node_count == 0u) {
        if (sum_bytes > caps->max_bytes) {
            lancius_set_error(LANCIUS_ERROR_LIMIT);
            return -1;
        }
        return 0;
    }

    {
        lancius_liveness_profile prof =
            lancius_analyze_liveness((lancius_graph *)g);
        lancius_error aerr = lancius_get_error();

        if (aerr != LANCIUS_ERROR_OK) {
            return -1;
        }
        if (prof.peak_memory_bytes > caps->max_bytes) {
            lancius_set_error(LANCIUS_ERROR_LIMIT);
            return -1;
        }
    }

    return 0;
}

int lancius_sandbox_check_weights(const lancius_graph *g)
{
    uint32_t i;

    lancius_clear_error();

    if (g == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return -1;
    }
    if (g->node_count > 0u && g->nodes == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return -1;
    }

    for (i = 0; i < g->node_count; i++) {
        const lancius_node *n = g->nodes[i];
        size_t elems = 0;
        const double *rt_buf = NULL;
        const float *rt_f32 = NULL;

        if (n == NULL) {
            lancius_set_error(LANCIUS_ERROR_NULL_PTR);
            return -1;
        }
        if (!lancius_node_elements_checked(n, &elems)) {
            lancius_set_error(LANCIUS_ERROR_LIMIT);
            return -1;
        }
        if (elems == 0u) {
            continue;
        }
        if (n->runtime_data != NULL &&
            sandbox_scan_f64(n->runtime_data, elems) != 0) {
            lancius_set_error(LANCIUS_ERROR_NUMERICAL);
            return -1;
        }
        if (n->runtime_data_f32 != NULL &&
            sandbox_scan_f32(n->runtime_data_f32, elems) != 0) {
            lancius_set_error(LANCIUS_ERROR_NUMERICAL);
            return -1;
        }
        if (n->rt != NULL) {
            rt_buf = (const double *)n->rt->buffer;
            rt_f32 = (const float *)n->rt->buffer_f32;
            if (rt_buf != NULL && rt_buf != n->runtime_data &&
                sandbox_scan_f64(rt_buf, elems) != 0) {
                lancius_set_error(LANCIUS_ERROR_NUMERICAL);
                return -1;
            }
            if (rt_f32 != NULL && rt_f32 != n->runtime_data_f32 &&
                sandbox_scan_f32(rt_f32, elems) != 0) {
                lancius_set_error(LANCIUS_ERROR_NUMERICAL);
                return -1;
            }
        }
    }

    return 0;
}
