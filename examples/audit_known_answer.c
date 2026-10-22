#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int checks = 0;
static int failures = 0;
static lancius_arena* scratch = NULL;

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

static lancius_schedule* run(lancius_graph* g) {
    lancius_schedule* s = lancius_ir_schedule(g);
    lancius_schedule_execute(s, scratch);
    return s;
}

static void finish(lancius_schedule* s, lancius_graph* g) {
    lancius_schedule_destroy(s);
    lancius_graph_destroy(g);
    lancius_arena_reset(scratch);
}

static void test_matmul(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* A = lancius_input(g, 2, 3);
    lancius_node* B = lancius_input(g, 3, 2);
    lancius_node* C = lancius_matmul(g, A, B);

    double a[6] = {-1.0, 2.0, 3.0, 4.0, -5.0, 6.0};
    double b[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    lancius_node_bind_external(A, a);
    lancius_node_bind_external(B, b);

    lancius_schedule* s = run(g);

    CHECK(C && C->runtime_data, "matmul output allocated");
    if (C && C->runtime_data) {
        CHECK(close_d(C->runtime_data[0], 4.0, 1e-12), "matmul [0] == 4");
        CHECK(close_d(C->runtime_data[1], 4.0, 1e-12), "matmul [1] == 4");
        CHECK(close_d(C->runtime_data[2], 5.0, 1e-12), "matmul [2] == 5");
        CHECK(close_d(C->runtime_data[3], 5.0, 1e-12), "matmul [3] == 5");
    }

    finish(s, g);
}

static void test_elementwise(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* A = lancius_input(g, 2, 2);
    lancius_node* B = lancius_input(g, 2, 2);

    lancius_node* add = lancius_add(g, A, B);
    lancius_node* sub = lancius_sub(g, A, B);
    lancius_node* mul = lancius_mul(g, A, B);
    lancius_node* rel = lancius_relu(g, A);

    double a[4] = {1.0, -2.0, 3.0, -4.0};
    double b[4] = {10.0, 20.0, 30.0, 40.0};

    lancius_node_bind_external(A, a);
    lancius_node_bind_external(B, b);

    lancius_schedule* s = run(g);

    CHECK(add && add->runtime_data, "add output allocated");
    CHECK(sub && sub->runtime_data, "sub output allocated");
    CHECK(mul && mul->runtime_data, "mul output allocated");
    CHECK(rel && rel->runtime_data, "relu output allocated");

    if (add && add->runtime_data) {
        CHECK(close_d(add->runtime_data[0], 11.0, 1e-12), "add [0]");
        CHECK(close_d(add->runtime_data[1], 18.0, 1e-12), "add [1]");
        CHECK(close_d(add->runtime_data[2], 33.0, 1e-12), "add [2]");
        CHECK(close_d(add->runtime_data[3], 36.0, 1e-12), "add [3]");
    }

    if (sub && sub->runtime_data) {
        CHECK(close_d(sub->runtime_data[0], -9.0, 1e-12), "sub [0]");
        CHECK(close_d(sub->runtime_data[1], -22.0, 1e-12), "sub [1]");
        CHECK(close_d(sub->runtime_data[2], -27.0, 1e-12), "sub [2]");
        CHECK(close_d(sub->runtime_data[3], -44.0, 1e-12), "sub [3]");
    }

    if (mul && mul->runtime_data) {
        CHECK(close_d(mul->runtime_data[0], 10.0, 1e-12), "mul [0]");
        CHECK(close_d(mul->runtime_data[1], -40.0, 1e-12), "mul [1]");
        CHECK(close_d(mul->runtime_data[2], 90.0, 1e-12), "mul [2]");
        CHECK(close_d(mul->runtime_data[3], -160.0, 1e-12), "mul [3]");
    }

    if (rel && rel->runtime_data) {
        CHECK(close_d(rel->runtime_data[0], 1.0, 1e-12), "relu [0]");
        CHECK(close_d(rel->runtime_data[1], 0.0, 1e-12), "relu [1]");
        CHECK(close_d(rel->runtime_data[2], 3.0, 1e-12), "relu [2]");
        CHECK(close_d(rel->runtime_data[3], 0.0, 1e-12), "relu [3]");
    }

    finish(s, g);
}

static void test_broadcast(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* A = lancius_input(g, 2, 2);
    lancius_node* bias = lancius_input(g, 1, 2);
    lancius_node* bc = lancius_broadcast(g, bias, 2, 2);
    lancius_node* out = lancius_add(g, A, bc);

    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double bias_data[2] = {100.0, 200.0};

    lancius_node_bind_external(A, a);
    lancius_node_bind_external(bias, bias_data);

    lancius_schedule* s = run(g);

    CHECK(out && out->runtime_data, "broadcast output allocated");
    if (out && out->runtime_data) {
        CHECK(close_d(out->runtime_data[0], 101.0, 1e-12), "broadcast [0]");
        CHECK(close_d(out->runtime_data[1], 202.0, 1e-12), "broadcast [1]");
        CHECK(close_d(out->runtime_data[2], 103.0, 1e-12), "broadcast [2]");
        CHECK(close_d(out->runtime_data[3], 204.0, 1e-12), "broadcast [3]");
    }

    finish(s, g);
}

static void test_softmax(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 1, 2);
    lancius_node* S = lancius_softmax(g, X);

    double x[2] = {0.0, 0.0};
    lancius_node_bind_external(X, x);

    lancius_schedule* s = run(g);

    CHECK(S && S->runtime_data, "softmax output allocated");
    if (S && S->runtime_data) {
        double sum = S->runtime_data[0] + S->runtime_data[1];
        CHECK(close_d(S->runtime_data[0], 0.5, 1e-12), "softmax [0] == 0.5");
        CHECK(close_d(S->runtime_data[1], 0.5, 1e-12), "softmax [1] == 0.5");
        CHECK(close_d(sum, 1.0, 1e-12), "softmax sum == 1");
    }

    finish(s, g);
}

static void test_transpose(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 2, 3);
    lancius_node* T = lancius_transpose(g, X);

    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    lancius_node_bind_external(X, x);

    lancius_schedule* s = run(g);

    CHECK(T && T->runtime_data, "transpose output allocated");
    if (T && T->runtime_data) {
        CHECK(T->shape[0] == 3 && T->shape[1] == 2, "transpose shape == [3,2]");
        CHECK(close_d(T->runtime_data[0], 1.0, 1e-12), "transpose [0]");
        CHECK(close_d(T->runtime_data[1], 4.0, 1e-12), "transpose [1]");
        CHECK(close_d(T->runtime_data[2], 2.0, 1e-12), "transpose [2]");
        CHECK(close_d(T->runtime_data[3], 5.0, 1e-12), "transpose [3]");
        CHECK(close_d(T->runtime_data[4], 3.0, 1e-12), "transpose [4]");
        CHECK(close_d(T->runtime_data[5], 6.0, 1e-12), "transpose [5]");
    }

    finish(s, g);
}

static void test_permute(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, 1, 2, 2, 1);
    lancius_node* P = lancius_permute(g, X, 0, 2, 1, 3);

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    lancius_node_bind_external(X, x);

    lancius_schedule* s = run(g);

    CHECK(P && P->runtime_data, "permute output allocated");
    if (P && P->runtime_data) {
        CHECK(P->shape[0] == 1 && P->shape[1] == 2 && P->shape[2] == 2 && P->shape[3] == 1,
              "permute shape == [1,2,2,1]");
        CHECK(close_d(P->runtime_data[0], 1.0, 1e-12), "permute [0]");
        CHECK(close_d(P->runtime_data[1], 3.0, 1e-12), "permute [1]");
        CHECK(close_d(P->runtime_data[2], 2.0, 1e-12), "permute [2]");
        CHECK(close_d(P->runtime_data[3], 4.0, 1e-12), "permute [3]");
    }

    finish(s, g);
}

static void test_all_permutations(void) {
    /* One hand-written permutation (above) can miss an index-coordinate bug
     * that only appears when two axes have distinct extents. Exhaust all 24
     * permutations of [2,3,4,5] and compare every output value with a
     * separately computed row-major reference. */
    const size_t S[4] = {2, 3, 4, 5};
    double x[120];
    for (int i = 0; i < 120; i++) x[i] = (double)(i + 1);

    int perm_count = 0;
    for (uint32_t a0 = 0; a0 < 4; a0++) {
        for (uint32_t a1 = 0; a1 < 4; a1++) {
            if (a1 == a0) continue;
            for (uint32_t a2 = 0; a2 < 4; a2++) {
                if (a2 == a0 || a2 == a1) continue;
                for (uint32_t a3 = 0; a3 < 4; a3++) {
                    if (a3 == a0 || a3 == a1 || a3 == a2) continue;
                    uint32_t axes[4] = {a0, a1, a2, a3};
                    size_t O[4] = {S[a0], S[a1], S[a2], S[a3]};
                    char msg[128];
                    snprintf(msg, sizeof msg, "permute [%u,%u,%u,%u]", a0, a1, a2, a3);

                    lancius_graph* g = lancius_graph_create();
                    lancius_node* X = lancius_input_4d(g, S[0], S[1], S[2], S[3]);
                    lancius_node* P = lancius_permute(g, X, a0, a1, a2, a3);
                    lancius_node_bind_external(X, x);
                    lancius_schedule* s = run(g);

                    int shape_ok = P && P->runtime_data;
                    for (int d = 0; d < 4 && shape_ok; d++) shape_ok = (P->shape[d] == O[d]);
                    int vals_ok = shape_ok;
                    if (shape_ok) {
                        for (size_t p = 0; p < 120 && vals_ok; p++) {
                            size_t rem = p, oc[4];
                            for (int d = 3; d >= 0; d--) { oc[d] = rem % O[d]; rem /= O[d]; }
                            size_t ic[4] = {0, 0, 0, 0};
                            for (int d = 0; d < 4; d++) ic[axes[d]] = oc[d];
                            size_t ii = ((ic[0] * S[1] + ic[1]) * S[2] + ic[2]) * S[3] + ic[3];
                            if (!close_d(P->runtime_data[p], x[ii], 1e-12)) vals_ok = 0;
                        }
                    }
                    CHECK(shape_ok, msg);
                    snprintf(msg, sizeof msg, "permute values [%u,%u,%u,%u]", a0, a1, a2, a3);
                    CHECK(vals_ok, msg);

                    finish(s, g);
                    perm_count++;
                }
            }
        }
    }
    CHECK(perm_count == 24, "all 24 permutations enumerated");
}

static void test_reshape(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, 1, 2, 3, 4);
    lancius_node* R = lancius_reshape(g, X, 4, 1, 4, 3, 2);
    double x[24];
    for (int i = 0; i < 24; i++) x[i] = (double)(i + 1);
    lancius_node_bind_external(X, x);
    lancius_schedule* s = run(g);

    CHECK(R && R->runtime_data, "reshape output allocated");
    if (R && R->runtime_data) {
        CHECK(R->shape[0] == 1 && R->shape[1] == 4 && R->shape[2] == 3 && R->shape[3] == 2,
              "reshape shape == [1,4,3,2]");
        for (int i = 0; i < 24; i++) {
            CHECK(close_d(R->runtime_data[i], x[i], 1e-12), "reshape preserves flat values");
        }
    }
    finish(s, g);
}

static void test_matmul_batched(void) {
    lancius_graph* g = lancius_graph_create();

    lancius_node* A = lancius_input(g, 2, 2);
    A->ndim = 3;
    A->shape[0] = 2;
    A->shape[1] = 1;
    A->shape[2] = 2;

    lancius_node* B = lancius_input(g, 2, 2);
    B->ndim = 3;
    B->shape[0] = 2;
    B->shape[1] = 2;
    B->shape[2] = 1;

    lancius_node* C = lancius_matmul_batched(g, A, B);

    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4] = {5.0, 6.0, 7.0, 8.0};

    lancius_node_bind_external(A, a);
    lancius_node_bind_external(B, b);

    lancius_schedule* s = run(g);

    CHECK(C && C->runtime_data, "matmul_batched output allocated");
    if (C && C->runtime_data) {
        CHECK(close_d(C->runtime_data[0], 17.0, 1e-12), "batched matmul [0] == 17");
        CHECK(close_d(C->runtime_data[1], 53.0, 1e-12), "batched matmul [1] == 53");
    }

    finish(s, g);
}

static void test_conv2d(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, 1, 1, 2, 2);
    lancius_node* W = lancius_input_4d(g, 1, 1, 2, 2);
    lancius_node* C = lancius_conv2d(g, X, W, 1, 0);

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double w[4] = {1.0, 1.0, 1.0, 1.0};

    lancius_node_bind_external(X, x);
    lancius_node_bind_external(W, w);

    lancius_schedule* s = run(g);

    CHECK(C && C->runtime_data, "conv2d output allocated");
    if (C && C->runtime_data) {
        CHECK(close_d(C->runtime_data[0], 10.0, 1e-12), "conv2d sum == 10");
    }

    finish(s, g);
}

static void test_maxpool2d(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, 1, 1, 2, 2);
    lancius_node* P = lancius_maxpool2d(g, X, 2, 2);

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    lancius_node_bind_external(X, x);

    lancius_schedule* s = run(g);

    CHECK(P && P->runtime_data, "maxpool output allocated");
    if (P && P->runtime_data) {
        CHECK(close_d(P->runtime_data[0], 4.0, 1e-12), "maxpool == 4");
    }

    finish(s, g);
}

static void test_flatten(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, 1, 2, 2, 2);
    lancius_node* F = lancius_flatten(g, X);

    double x[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    lancius_node_bind_external(X, x);

    lancius_schedule* s = run(g);

    CHECK(F && F->runtime_data, "flatten output allocated");
    if (F && F->runtime_data) {
        CHECK(F->shape[0] == 1 && F->shape[1] == 8, "flatten shape == [1,8]");
        for (int i = 0; i < 8; i++) {
            CHECK(close_d(F->runtime_data[i], (double)(i + 1), 1e-12), "flatten value");
        }
    }

    finish(s, g);
}

static void test_cross_entropy_bwd(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* logits = lancius_input(g, 2, 3);
    lancius_node* targets = lancius_input(g, 2, 3);
    lancius_node* grad = lancius_input(g, 1, 1);
    lancius_node* dx = lancius_cross_entropy_bwd(g, logits, targets, grad);

    double l[6] = {1.0, 2.0, 3.0, 0.5, -1.0, 0.25};
    double t[6] = {0.0, 1.0, 0.0, 1.0, 0.0, 0.0};
    double go[1] = {0.5};
    lancius_node_bind_external(logits, l);
    lancius_node_bind_external(targets, t);
    lancius_node_bind_external(grad, go);

    lancius_schedule* s = run(g);

    CHECK(dx && dx->runtime_data, "cross_entropy_bwd output allocated");
    if (dx && dx->runtime_data) {
        CHECK(dx->shape[0] == 2 && dx->shape[1] == 3, "cross_entropy_bwd shape == [2,3]");
        for (int r = 0; r < 2; r++) {
            double max_val = l[r * 3];
            for (int c = 1; c < 3; c++) if (l[r * 3 + c] > max_val) max_val = l[r * 3 + c];
            double sum_exp = 0.0;
            for (int c = 0; c < 3; c++) sum_exp += exp(l[r * 3 + c] - max_val);
            for (int c = 0; c < 3; c++) {
                double sm = exp(l[r * 3 + c] - max_val) / sum_exp;
                double want = (sm - t[r * 3 + c]) * (go[0] / 2.0);
                CHECK(close_d(dx->runtime_data[r * 3 + c], want, 1e-12),
                      "cross_entropy_bwd matches (softmax - target) * scale");
            }
        }
    }

    finish(s, g);

    /* Corrupt labels must fail loud, mirroring the forward op. */
    lancius_clear_error();
    g = lancius_graph_create();
    logits = lancius_input(g, 1, 2);
    targets = lancius_input(g, 1, 2);
    grad = lancius_input(g, 1, 1);
    dx = lancius_cross_entropy_bwd(g, logits, targets, grad);
    double bad_l[2] = {0.0, 0.0};
    double bad_t[2] = {-1.0, 2.0};
    double one[1] = {1.0};
    lancius_node_bind_external(logits, bad_l);
    lancius_node_bind_external(targets, bad_t);
    lancius_node_bind_external(grad, one);
    s = run(g);
    CHECK(lancius_get_error() == LANCIUS_ERROR_NUMERICAL,
          "cross_entropy_bwd rejects negative targets");
    finish(s, g);
    lancius_clear_error();
}

static void test_mse_bwd(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* pred = lancius_input(g, 2, 3);
    lancius_node* target = lancius_input(g, 2, 3);
    lancius_node* grad = lancius_input(g, 1, 1);
    lancius_node* dx = lancius_mse_bwd(g, pred, target, grad);

    double p[6] = {1.0, 2.0, 3.0, -1.0, 0.5, 4.0};
    double t[6] = {0.0, 2.5, 1.0, 0.5, -1.0, 3.0};
    double go[1] = {0.25};
    lancius_node_bind_external(pred, p);
    lancius_node_bind_external(target, t);
    lancius_node_bind_external(grad, go);

    lancius_schedule* s = run(g);

    CHECK(dx && dx->runtime_data, "mse_bwd output allocated");
    if (dx && dx->runtime_data) {
        CHECK(dx->shape[0] == 2 && dx->shape[1] == 3, "mse_bwd shape == [2,3]");
        for (int k = 0; k < 6; k++) {
            double want = (2.0 * go[0] / 6.0) * (p[k] - t[k]);
            CHECK(close_d(dx->runtime_data[k], want, 1e-12), "mse_bwd matches 2*g/pe*(p-t)");
        }
    }

    finish(s, g);
}

/* v12R2 generic primitives: bounded activation + regression loss.
 * Framework only: tanh saturates the fluid scale, MSE grounds step labels. */
static void test_tanh_mse(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input(g, 1, 3);
    lancius_node* Y = lancius_tanh(g, X);
    lancius_node* P = lancius_input(g, 1, 3);
    lancius_node* T = lancius_input(g, 1, 3);
    lancius_node* L = lancius_mse(g, P, T);

    double x[3] = {0.0, 1.0, -1.0};
    double p[3] = {1.0, 2.0, 3.0};
    double t[3] = {1.0, 2.0, 4.0};
    lancius_node_bind_external(X, x);
    lancius_node_bind_external(P, p);
    lancius_node_bind_external(T, t);

    lancius_schedule* s = run(g);

    CHECK(Y && Y->runtime_data, "tanh output allocated");
    if (Y && Y->runtime_data) {
        CHECK(close_d(Y->runtime_data[0], 0.0, 1e-12), "tanh(0) == 0");
        CHECK(close_d(Y->runtime_data[1], 0.7615941559557649, 1e-12), "tanh(1)");
        CHECK(close_d(Y->runtime_data[2], -0.7615941559557649, 1e-12), "tanh(-1)");
    }
    CHECK(L && L->runtime_data, "mse output allocated");
    if (L && L->runtime_data) {
        CHECK(close_d(L->runtime_data[0], 1.0 / 3.0, 1e-12), "mse == 1/3");
    }

    finish(s, g);
}

static void test_cross_entropy(void) {
    lancius_graph* g = lancius_graph_create();
    lancius_node* logits = lancius_input(g, 1, 2);
    lancius_node* targets = lancius_input(g, 1, 2);
    lancius_node* loss = lancius_cross_entropy(g, logits, targets);

    double l[2] = {0.0, 0.0};
    double t[2] = {1.0, 0.0};

    lancius_node_bind_external(logits, l);
    lancius_node_bind_external(targets, t);

    lancius_schedule* s = run(g);

    CHECK(loss && loss->runtime_data, "cross_entropy output allocated");
    if (loss && loss->runtime_data) {
        CHECK(close_d(loss->runtime_data[0], log(2.0), 1e-12), "cross_entropy == log(2)");
    }

    finish(s, g);
}

int main(void) {
    printf("================================================================\n");
    printf("  LANCIUS v11A1: KNOWN-ANSWER CORRECTNESS AUDIT\n");
    printf("================================================================\n");

    scratch = lancius_arena_create(64 * 1024 * 1024);
    if (!scratch) {
        printf("FATAL: could not create scratch arena\n");
        return 1;
    }

    int before;

    before = failures; test_matmul();          report("MatMul", before);
    before = failures; test_elementwise();     report("Add/Sub/Mul/ReLU", before);
    before = failures; test_broadcast();       report("Broadcast", before);
    before = failures; test_softmax();         report("Softmax", before);
    before = failures; test_transpose();       report("Transpose", before);
    before = failures; test_permute();         report("Permute", before);
    before = failures; test_all_permutations(); report("All 24 Permutations", before);
    before = failures; test_reshape();         report("Reshape", before);
    before = failures; test_matmul_batched();  report("MatMulBatched", before);
    before = failures; test_conv2d();          report("Conv2D", before);
    before = failures; test_maxpool2d();       report("MaxPool2D", before);
    before = failures; test_flatten();         report("Flatten", before);
    before = failures; test_cross_entropy();   report("CrossEntropy", before);
    before = failures; test_cross_entropy_bwd(); report("CrossEntropyBwd", before);
    before = failures; test_mse_bwd();         report("MSEBwd", before);
    before = failures; test_tanh_mse();        report("Tanh/MSE", before);

    lancius_arena_destroy(scratch);

    printf("================================================================\n");
    printf("  KNOWN-ANSWER AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    printf("================================================================\n");

    return failures ? 1 : 0;
}
