#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int checks = 0;
static int failures = 0;

#define CHECK(cond, msg) \
    do { \
        checks++; \
        if (!(cond)) { \
            printf("  ❌ FAIL: %s\n", msg); \
            failures++; \
        } \
    } while (0)

static int close_f(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

static void report(const char* name, int before) {
    if (failures == before) printf("  ✅ %s\n", name);
}

static void test_kernel_matmul_f32(void) {
    float a[6] = {-1.0f, 2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
    float b[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float c[4];

    double ad[6], bd[6], cd[4];

    for (int i = 0; i < 6; i++) {
        ad[i] = (double)a[i];
        bd[i] = (double)b[i];
    }

    kernel_matmul_f32(c, a, b, 2, 3, 2);
    kernel_matmul(cd, ad, bd, 2, 3, 2);

    for (int i = 0; i < 4; i++) {
        CHECK(close_f(c[i], (float)cd[i], 1e-6f), "kernel_matmul_f32 parity");
    }
}

static void test_scheduler_fp32_matmul(void) {
    lancius_graph* g = lancius_graph_create();

    lancius_node* A = lancius_input(g, 2, 3);
    lancius_node* B = lancius_input(g, 3, 2);
    lancius_node* C = lancius_matmul(g, A, B);

    CHECK(C != NULL, "fp32 matmul graph created");

    if (!C) {
        lancius_graph_destroy(g);
        return;
    }

    C->dtype = LANCIUS_DTYPE_FP32;
    if (C->rt) C->rt->dtype = LANCIUS_DTYPE_FP32;

    float a[6] = {-1.0f, 2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
    float b[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    lancius_node_bind_external_f32(A, a);
    lancius_node_bind_external_f32(B, b);

    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(1024 * 1024);

    CHECK(s != NULL, "fp32 schedule created");
    CHECK(scratch != NULL, "fp32 scratch created");

    if (!s || !scratch) {
        if (s) lancius_schedule_destroy(s);
        if (scratch) lancius_arena_destroy(scratch);
        lancius_graph_destroy(g);
        return;
    }

    lancius_schedule_execute(s, scratch);

    CHECK(C->runtime_data_f32 != NULL, "fp32 output allocated");

    if (C->runtime_data_f32) {
        CHECK(close_f(C->runtime_data_f32[0], 4.0f, 1e-6f), "fp32 matmul [0]");
        CHECK(close_f(C->runtime_data_f32[1], 4.0f, 1e-6f), "fp32 matmul [1]");
        CHECK(close_f(C->runtime_data_f32[2], 5.0f, 1e-6f), "fp32 matmul [2]");
        CHECK(close_f(C->runtime_data_f32[3], 5.0f, 1e-6f), "fp32 matmul [3]");
    }

    lancius_schedule_destroy(s);
    lancius_arena_destroy(scratch);
    lancius_graph_destroy(g);
}

static void test_serialization_fp32(void) {
    const char* path = "fp32_roundtrip.lancius";

    lancius_graph* g = lancius_graph_create();

    lancius_node* X = lancius_input(g, 2, 2);
    lancius_node* W = lancius_input(g, 2, 2);
    lancius_node* M = lancius_matmul(g, X, W);
    (void)M;

    float* wmem = (float*)malloc(4 * sizeof(float));
    CHECK(wmem != NULL, "fp32 weight allocation");

    if (!wmem) {
        lancius_graph_destroy(g);
        return;
    }

    wmem[0] = 1.5f;
    wmem[1] = -2.5f;
    wmem[2] = 3.25f;
    wmem[3] = 4.75f;

    lancius_node_bind_owned_heap_f32(W, wmem);

    lancius_graph_save(g, path);
    lancius_graph_destroy(g);

    lancius_graph* gl = lancius_graph_load(path);
    CHECK(gl != NULL, "fp32 model loaded");

    if (gl) {
        lancius_node* Wl = NULL;

        for (uint32_t i = 0; i < gl->node_count; i++) {
            lancius_node* n = gl->nodes[i];
            if (n->op == LANCIUS_OP_INPUT &&
                n->ndim == 2 &&
                n->shape[0] == 2 &&
                n->shape[1] == 2 &&
                n->dtype == LANCIUS_DTYPE_FP32 &&
                n->runtime_data_f32 != NULL) {
                Wl = n;
                break;
            }
        }

        CHECK(Wl != NULL, "fp32 weight tensor loaded");

        if (Wl && Wl->runtime_data_f32) {
            CHECK(close_f(Wl->runtime_data_f32[0], 1.5f, 0.0f), "fp32 weight [0]");
            CHECK(close_f(Wl->runtime_data_f32[1], -2.5f, 0.0f), "fp32 weight [1]");
            CHECK(close_f(Wl->runtime_data_f32[2], 3.25f, 0.0f), "fp32 weight [2]");
            CHECK(close_f(Wl->runtime_data_f32[3], 4.75f, 0.0f), "fp32 weight [3]");
        }

        lancius_graph_destroy(gl);
    }

    remove(path);
}

int main(void) {
    printf("================================================================\n");
    printf("  LANCIUS v11A2: FP32 PATH AUDIT\n");
    printf("================================================================\n");

    int before;

    before = failures; test_kernel_matmul_f32();    report("FP32 matmul kernel", before);
    before = failures; test_scheduler_fp32_matmul(); report("FP32 scheduler execution", before);
    before = failures; test_serialization_fp32();    report("FP32 serialization roundtrip", before);

    printf("================================================================\n");
    printf("  FP32 AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    printf("================================================================\n");

    return failures ? 1 : 0;
}
