#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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

static int close_d(double a, double b, double tol) {
    return fabs(a - b) <= tol;
}

static void report(const char* name, int before) {
    if (failures == before) printf("  ✅ %s\n", name);
}

static int wr_u32_le(FILE* f, uint32_t v) {
    uint8_t b[4];
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
    return fwrite(b, 1, 4, f) == 4;
}

static int wr_u64_le(FILE* f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; i++) {
        b[i] = (uint8_t)((v >> (8 * i)) & 0xFF);
    }
    return fwrite(b, 1, 8, f) == 8;
}

static void test_ownership_contract(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* ext = lancius_input(g, 2, 2);
    lancius_node* owned = lancius_input(g, 2, 2);

    double ext_buf[4] = {1.0, 2.0, 3.0, 4.0};
    double* owned_buf = (double*)malloc(4 * sizeof(double));

    for (int i = 0; i < 4; i++) owned_buf[i] = (double)i;

    lancius_node_bind_external(ext, ext_buf);
    lancius_node_bind_owned_heap(owned, owned_buf);

    CHECK(lancius_node_get_owner(ext) == LANCIUS_MEMORY_EXTERNAL,
          "external buffer owner is external");

    CHECK(lancius_node_get_owner(owned) == LANCIUS_MEMORY_OWNED_HEAP,
          "owned buffer owner is owned_heap");

    lancius_graph_release_owned(g);

    CHECK(ext->runtime_data == ext_buf,
          "external buffer preserved after release_owned");

    CHECK(owned->runtime_data == NULL,
          "owned buffer cleared after release_owned");

    CHECK(lancius_node_get_owner(owned) == LANCIUS_MEMORY_EXTERNAL,
          "owned buffer owner reset after release_owned");

    lancius_graph_destroy(g);

    lancius_graph* g2 = lancius_graph_create();
    lancius_node* owned8 = lancius_input(g2, 1, 4);
    int8_t* buf8 = (int8_t*)malloc(4);

    for (int i = 0; i < 4; i++) buf8[i] = (int8_t)i;

    lancius_node_bind_owned_heap_int8(owned8, buf8);

    CHECK(owned8->rt && owned8->rt->int8_owner == LANCIUS_MEMORY_OWNED_HEAP,
          "int8 owned buffer owner is owned_heap");

    lancius_graph_release_owned(g2);

    CHECK(owned8->runtime_data_int8 == NULL,
          "int8 owned buffer cleared after release_owned");

    CHECK(owned8->rt && owned8->rt->int8_owner == LANCIUS_MEMORY_EXTERNAL,
          "int8 owned buffer owner reset after release_owned");

    lancius_graph_destroy(g2);
}

static void test_serialization_roundtrip(void) {
    const char* path = "regression_roundtrip.lancius";

    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 2);
    lancius_node* W = lancius_input(g, 2, 2);
    lancius_node* M = lancius_matmul(g, X, W);
    lancius_node* Y = lancius_relu(g, M);

    (void)M;

    double* wmem = (double*)malloc(4 * sizeof(double));
    wmem[0] = 1.0;
    wmem[1] = 2.0;
    wmem[2] = 3.0;
    wmem[3] = 4.0;

    lancius_node_bind_owned_heap(W, wmem);

    lancius_graph_save(g, path);

    FILE* f = fopen(path, "rb");
    CHECK(f != NULL, "saved model exists");

    uint8_t magic[4] = {0, 0, 0, 0};

    if (f) {
        size_t rd = fread(magic, 1, 4, f);
        CHECK(rd == 4, "read model magic");
        fclose(f);
    }

    CHECK(magic[0] == 0x41 && magic[1] == 0x4E && magic[2] == 0x43 && magic[3] == 0x32,
          "model uses v2 magic ANC2");

    double xbuf[4] = {1.0, -2.0, 3.0, -4.0};
    lancius_node_bind_external(X, xbuf);

    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(1024 * 1024);

    lancius_schedule_execute(s, scratch);

    double out_orig[4] = {0.0, 0.0, 0.0, 0.0};

    CHECK(Y && Y->runtime_data, "original output allocated");

    if (Y && Y->runtime_data) {
        memcpy(out_orig, Y->runtime_data, sizeof(out_orig));
    }

    uint32_t node_count = g->node_count;

    lancius_schedule_destroy(s);
    lancius_arena_destroy(scratch);
    lancius_graph_destroy(g);

    lancius_graph* gl = lancius_graph_load(path);

    CHECK(gl != NULL, "model loaded");

    if (gl) {
        CHECK(gl->node_count == node_count, "loaded node count matches");

        lancius_node* Xl = NULL;
        lancius_node* Yl = NULL;

        for (uint32_t i = 0; i < gl->node_count; i++) {
            lancius_node* n = gl->nodes[i];

            if (n->op == LANCIUS_OP_INPUT &&
                n->runtime_data == NULL &&
                n->shape[0] == 2 &&
                n->shape[1] == 2) {
                Xl = n;
            }

            if (n->op == LANCIUS_OP_RELU) {
                Yl = n;
            }
        }

        CHECK(Xl != NULL, "loaded input found");
        CHECK(Yl != NULL, "loaded output found");

        if (Xl && Yl) {
            lancius_node_bind_external(Xl, xbuf);

            lancius_schedule* sl = lancius_ir_schedule(gl);
            lancius_arena* scratchl = lancius_arena_create(1024 * 1024);

            lancius_schedule_execute(sl, scratchl);

            CHECK(Yl->runtime_data != NULL, "loaded output allocated");

            int match = 1;

            if (Yl->runtime_data) {
                for (int i = 0; i < 4; i++) {
                    if (!close_d(out_orig[i], Yl->runtime_data[i], 1e-12)) {
                        match = 0;
                        break;
                    }
                }
            } else {
                match = 0;
            }

            CHECK(match, "loaded graph output matches original graph output");

            lancius_schedule_destroy(sl);
            lancius_arena_destroy(scratchl);
        }

        lancius_graph_destroy(gl);
    }

    remove(path);
}

static void test_malformed_models(void) {
    lancius_graph* g = NULL;

    const char* bad_magic_path = "regression_bad_magic.lancius";
    FILE* f = fopen(bad_magic_path, "wb");
    CHECK(f != NULL, "create bad magic file");
    if (f) {
        CHECK(wr_u32_le(f, 0x12345678u), "write bad magic");
        fclose(f);
    }
    g = lancius_graph_load(bad_magic_path);
    CHECK(g == NULL, "reject bad magic");
    if (g) lancius_graph_destroy(g);
    remove(bad_magic_path);

    const char* trunc_path = "regression_trunc_v2.lancius";
    f = fopen(trunc_path, "wb");
    CHECK(f != NULL, "create truncated v2 file");
    if (f) {
        CHECK(wr_u32_le(f, 0x32434E41u), "write v2 magic");
        CHECK(wr_u32_le(f, 2u), "write v2 version");
        CHECK(wr_u32_le(f, 3u), "write v2 flags");
        fclose(f);
    }
    g = lancius_graph_load(trunc_path);
    CHECK(g == NULL, "reject truncated v2 header");
    if (g) lancius_graph_destroy(g);
    remove(trunc_path);

    const char* huge_v1_path = "regression_huge_v1.lancius";
    f = fopen(huge_v1_path, "wb");
    CHECK(f != NULL, "create huge v1 file");
    if (f) {
        CHECK(wr_u32_le(f, 0x21434E41u), "write v1 magic");
        CHECK(wr_u32_le(f, 0xFFFFFFFFu), "write huge v1 node count");
        fclose(f);
    }
    g = lancius_graph_load(huge_v1_path);
    CHECK(g == NULL, "reject huge v1 node count");
    if (g) lancius_graph_destroy(g);
    remove(huge_v1_path);

    const char* huge_v2_path = "regression_huge_v2.lancius";
    f = fopen(huge_v2_path, "wb");
    CHECK(f != NULL, "create huge v2 file");
    if (f) {
        CHECK(wr_u32_le(f, 0x32434E41u), "write v2 magic");
        CHECK(wr_u32_le(f, 2u), "write v2 version");
        CHECK(wr_u32_le(f, 3u), "write v2 flags");
        CHECK(wr_u32_le(f, 0xFFFFFFFFu), "write huge v2 node count");
        CHECK(wr_u32_le(f, 0u), "write tensor_count");
        CHECK(wr_u32_le(f, 0u), "write attribute_count");
        CHECK(wr_u32_le(f, 48u), "write header_size");
        CHECK(wr_u32_le(f, 0u), "write reserved0");
        CHECK(wr_u64_le(f, 0u), "write weight_block_offset");
        CHECK(wr_u32_le(f, 0u), "write checksum");
        CHECK(wr_u32_le(f, 0u), "write reserved1");
        fclose(f);
    }
    g = lancius_graph_load(huge_v2_path);
    CHECK(g == NULL, "reject huge v2 node count");
    if (g) lancius_graph_destroy(g);
    remove(huge_v2_path);
}

static void test_deterministic_reexecute(void) {
    double A[16];
    double B[16];

    for (int i = 0; i < 16; i++) {
        A[i] = (double)(i - 7);
        B[i] = (double)(i % 5) - 2.0;
    }

    for (int iter = 0; iter < 6; iter++) {
        lancius_graph* g = lancius_graph_create();
        lancius_node* nA = lancius_input(g, 4, 4);
        lancius_node* nB = lancius_input(g, 4, 4);

        lancius_node* Y = NULL;

        if (iter % 3 == 0) {
            Y = lancius_add(g, nA, nB);
        } else if (iter % 3 == 1) {
            Y = lancius_mul(g, nA, nB);
        } else {
            Y = lancius_relu(g, nA);
        }

        lancius_node_bind_external(nA, A);
        lancius_node_bind_external(nB, B);

        lancius_schedule* s = lancius_ir_schedule(g);
        lancius_arena* scratch = lancius_arena_create(1024 * 1024);

        lancius_schedule_execute(s, scratch);

        double first[16] = {0};

        if (Y && Y->runtime_data) {
            memcpy(first, Y->runtime_data, sizeof(first));
        } else {
            CHECK(0, "determinism output allocated on first run");
        }

        lancius_schedule_execute(s, scratch);

        int match = 1;

        if (Y && Y->runtime_data) {
            for (int i = 0; i < 16; i++) {
                if (!close_d(first[i], Y->runtime_data[i], 0.0)) {
                    match = 0;
                    break;
                }
            }
        } else {
            match = 0;
        }

        CHECK(match, "re-execution is deterministic");

        lancius_schedule_destroy(s);
        lancius_arena_destroy(scratch);
        lancius_graph_destroy(g);
    }
}

int main(void) {
    printf("================================================================\n");
    printf("  LANCIUS v11A1: REGRESSION HARDENING AUDIT (TASK 13C)\n");
    printf("================================================================\n");

    int before;

    before = failures;
    test_ownership_contract();
    report("Ownership contract", before);

    before = failures;
    test_serialization_roundtrip();
    report("Serialization roundtrip", before);

    before = failures;
    test_malformed_models();
    report("Malformed model rejection", before);

    before = failures;
    test_deterministic_reexecute();
    report("Deterministic re-execution", before);

    printf("================================================================\n");
    printf("  REGRESSION AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    printf("================================================================\n");

    return failures ? 1 : 0;
}
