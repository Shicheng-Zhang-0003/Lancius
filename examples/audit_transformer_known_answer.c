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

/*
 * Deterministic test RNG.
 */
static uint32_t rng_state = 123456789u;

static double frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((double)(rng_state >> 8) / (double)(1u << 24)) * 2.0 - 1.0;
}

/*
 * Naive causal attention reference.
 */
static void naive_attention(
    double* out,
    const double* q,
    const double* k,
    const double* v,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim
) {
    double scale = 1.0 / sqrt((double)head_dim);
    double* scores = (double*)malloc(seq_len * sizeof(double));

    if (!scores) {
        fprintf(stderr, "FATAL: naive_attention OOM\n");
        exit(1);
    }

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t h = 0; h < n_heads; h++) {
            double max_s = -1e9;

            for (size_t j = 0; j < seq_len; j++) {
                double s = 0.0;

                for (size_t d = 0; d < head_dim; d++) {
                    size_t q_idx = (i * n_heads * head_dim) + (h * head_dim) + d;
                    size_t k_idx = (j * n_heads * head_dim) + (h * head_dim) + d;
                    s += q[q_idx] * k[k_idx];
                }

                s *= scale;

                if (j > i) {
                    s = -1e9;
                }

                scores[j] = s;

                if (s > max_s) {
                    max_s = s;
                }
            }

            double sum_exp = 0.0;

            for (size_t j = 0; j < seq_len; j++) {
                scores[j] = exp(scores[j] - max_s);
                sum_exp += scores[j];
            }

            for (size_t j = 0; j < seq_len; j++) {
                scores[j] /= sum_exp;
            }

            for (size_t d = 0; d < head_dim; d++) {
                double val = 0.0;

                for (size_t j = 0; j < seq_len; j++) {
                    size_t v_idx = (j * n_heads * head_dim) + (h * head_dim) + d;
                    val += scores[j] * v[v_idx];
                }

                size_t out_idx = (i * n_heads * head_dim) + (h * head_dim) + d;
                out[out_idx] = val;
            }
        }
    }

    free(scores);
}

static double ref_gelu(double x) {
    if (x > 100.0) x = 100.0;
    if (x < -100.0) x = -100.0;

    const double sqrt_2_over_pi = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}

static double ref_silu(double x) {
    return x / (1.0 + exp(-x));
}

static void test_layernorm(void) {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double gamma[4] = {1.0, 1.0, 1.0, 1.0};
    double beta[4] = {0.0, 0.0, 0.0, 0.0};
    double y[4];

    kernel_layernorm(y, x, gamma, beta, 1, 4, 1e-5);

    double mean = 2.5;
    double var = 1.25;
    double inv = 1.0 / sqrt(var + 1e-5);

    for (int i = 0; i < 4; i++) {
        CHECK(close_d(y[i], (x[i] - mean) * inv, 1e-12), "layernorm value");
    }
}

static void test_rmsnorm(void) {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double gamma[4] = {1.0, 1.0, 1.0, 1.0};
    double y[4];

    kernel_rmsnorm(y, x, gamma, 1, 4, 1e-5);

    double rms = sqrt(7.5 + 1e-5);

    for (int i = 0; i < 4; i++) {
        CHECK(close_d(y[i], x[i] / rms, 1e-12), "rmsnorm value");
    }
}

static void test_gelu(void) {
    double x[6] = {-2.0, -1.0, 0.0, 0.5, 1.0, 2.0};
    double y[6];

    kernel_gelu(y, x, 6);

    for (int i = 0; i < 6; i++) {
        CHECK(close_d(y[i], ref_gelu(x[i]), 1e-12), "gelu value");
    }
}

static void test_swiglu(void) {
    double gate[4] = {-1.0, 0.0, 1.0, 2.0};
    double up[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4];

    kernel_swiglu(y, gate, up, 4);

    for (int i = 0; i < 4; i++) {
        double expected = ref_silu(gate[i]) * up[i];
        CHECK(close_d(y[i], expected, 1e-12), "swiglu value");
    }
}

static void test_rope(void) {
    size_t n_heads = 2;
    size_t head_dim = 4;
    size_t hidden = n_heads * head_dim;
    int pos = 3;

    double q[8], k[8];
    double qr[8], kr[8];

    for (size_t i = 0; i < hidden; i++) {
        q[i] = frand();
        k[i] = frand();
    }

    memcpy(qr, q, sizeof(q));
    memcpy(kr, k, sizeof(k));

    kernel_rope(q, k, 1, 1, n_heads, head_dim, pos);

    for (size_t h = 0; h < n_heads; h++) {
        for (size_t d = 0; d < head_dim; d += 2) {
            size_t idx = h * head_dim + d;

            double freq = 1.0 / pow(10000.0, (double)d / (double)head_dim);
            double angle = (double)pos * freq;
            double c = cos(angle);
            double s = sin(angle);

            double q0 = qr[idx];
            double q1 = qr[idx + 1];
            double k0 = kr[idx];
            double k1 = kr[idx + 1];

            qr[idx]     = q0 * c - q1 * s;
            qr[idx + 1] = q0 * s + q1 * c;

            kr[idx]     = k0 * c - k1 * s;
            kr[idx + 1] = k0 * s + k1 * c;
        }
    }

    for (size_t i = 0; i < hidden; i++) {
        CHECK(close_d(q[i], qr[i], 1e-12), "rope q value");
        CHECK(close_d(k[i], kr[i], 1e-12), "rope k value");
    }
}

static void test_attention_full(void) {
    size_t seq = 8;
    size_t n_heads = 2;
    size_t head_dim = 4;
    size_t hidden = n_heads * head_dim;
    size_t total = seq * hidden;

    double* q = (double*)malloc(total * sizeof(double));
    double* k = (double*)malloc(total * sizeof(double));
    double* v = (double*)malloc(total * sizeof(double));
    double* out_naive = (double*)malloc(total * sizeof(double));
    double* out_kernel = (double*)malloc(total * sizeof(double));

    if (!q || !k || !v || !out_naive || !out_kernel) {
        CHECK(0, "attention allocation");
        free(q); free(k); free(v); free(out_naive); free(out_kernel);
        return;
    }

    for (size_t i = 0; i < total; i++) {
        q[i] = frand();
        k[i] = frand();
        v[i] = frand();
    }

    naive_attention(out_naive, q, k, v, seq, n_heads, head_dim);
    kernel_attention(out_kernel, q, k, v, seq, n_heads, head_dim);

    for (size_t i = 0; i < total; i++) {
        CHECK(close_d(out_kernel[i], out_naive[i], 1e-9), "full attention parity");
    }

    free(q);
    free(k);
    free(v);
    free(out_naive);
    free(out_kernel);
}

static void test_kv_cache_step_parity(void) {
    size_t seq = 8;
    size_t n_heads = 2;
    size_t head_dim = 4;
    size_t hidden = n_heads * head_dim;
    size_t total = seq * hidden;

    double* q = (double*)malloc(total * sizeof(double));
    double* k = (double*)malloc(total * sizeof(double));
    double* v = (double*)malloc(total * sizeof(double));
    double* full_out = (double*)malloc(total * sizeof(double));
    double* step_out = (double*)malloc(hidden * sizeof(double));

    if (!q || !k || !v || !full_out || !step_out) {
        CHECK(0, "kv-cache allocation");
        free(q); free(k); free(v); free(full_out); free(step_out);
        return;
    }

    for (size_t i = 0; i < total; i++) {
        q[i] = frand();
        k[i] = frand();
        v[i] = frand();
    }

    kernel_attention(full_out, q, k, v, seq, n_heads, head_dim);

    lancius_kv_cache* cache = lancius_kv_cache_create(
        seq,
        n_heads,
        head_dim,
        LANCIUS_DTYPE_FP64
    );

    CHECK(cache != NULL, "kv-cache created");

    if (!cache) {
        free(q); free(k); free(v); free(full_out); free(step_out);
        return;
    }

    for (size_t t = 0; t < seq; t++) {
        int rc = lancius_kv_cache_append(
            cache,
            k + t * hidden,
            v + t * hidden,
            1
        );

        CHECK(rc == 0, "kv-cache append");

        size_t active_seq = lancius_kv_cache_seq_len(cache);
        const double* k_cache = (const double*)lancius_kv_cache_k_buffer(cache, NULL);
        const double* v_cache = (const double*)lancius_kv_cache_v_buffer(cache, NULL);

        CHECK(k_cache != NULL, "kv-cache k buffer");
        CHECK(v_cache != NULL, "kv-cache v buffer");

        kernel_attention_kv_cache(
            step_out,
            q + t * hidden,
            k_cache,
            v_cache,
            active_seq,
            n_heads,
            head_dim
        );

        for (size_t i = 0; i < hidden; i++) {
            CHECK(
                close_d(step_out[i], full_out[t * hidden + i], 1e-9),
                "kv-cache step parity"
            );
        }
    }

    lancius_kv_cache_destroy(cache);

    free(q);
    free(k);
    free(v);
    free(full_out);
    free(step_out);
}

static void test_prefill_generation_parity(void) {
    size_t seq = 5;
    size_t prompt_len = 4;
    size_t n_heads = 2;
    size_t head_dim = 4;
    size_t hidden = n_heads * head_dim;
    size_t total = seq * hidden;

    double* q = (double*)malloc(total * sizeof(double));
    double* k = (double*)malloc(total * sizeof(double));
    double* v = (double*)malloc(total * sizeof(double));
    double* full_out = (double*)malloc(total * sizeof(double));
    double* step_out = (double*)malloc(hidden * sizeof(double));

    if (!q || !k || !v || !full_out || !step_out) {
        CHECK(0, "prefill allocation");
        free(q); free(k); free(v); free(full_out); free(step_out);
        return;
    }

    for (size_t i = 0; i < total; i++) {
        q[i] = frand();
        k[i] = frand();
        v[i] = frand();
    }

    kernel_attention(full_out, q, k, v, seq, n_heads, head_dim);

    lancius_kv_cache* cache = lancius_kv_cache_create(
        seq,
        n_heads,
        head_dim,
        LANCIUS_DTYPE_FP64
    );

    CHECK(cache != NULL, "prefill cache created");

    if (!cache) {
        free(q); free(k); free(v); free(full_out); free(step_out);
        return;
    }

    /*
     * Prefill prompt tokens 0..3 in one shot.
     */
    int rc = lancius_kv_cache_append(cache, k, v, prompt_len);
    CHECK(rc == 0, "prefill append");

    /*
     * Generate token 4.
     */
    rc = lancius_kv_cache_append(
        cache,
        k + prompt_len * hidden,
        v + prompt_len * hidden,
        1
    );
    CHECK(rc == 0, "generation append");

    size_t active_seq = lancius_kv_cache_seq_len(cache);
    const double* k_cache = (const double*)lancius_kv_cache_k_buffer(cache, NULL);
    const double* v_cache = (const double*)lancius_kv_cache_v_buffer(cache, NULL);

    CHECK(active_seq == seq, "prefill active sequence length");
    CHECK(k_cache != NULL, "prefill k buffer");
    CHECK(v_cache != NULL, "prefill v buffer");

    kernel_attention_kv_cache(
        step_out,
        q + prompt_len * hidden,
        k_cache,
        v_cache,
        active_seq,
        n_heads,
        head_dim
    );

    for (size_t i = 0; i < hidden; i++) {
        CHECK(
            close_d(step_out[i], full_out[prompt_len * hidden + i], 1e-9),
            "prefill + generation parity"
        );
    }

    lancius_kv_cache_destroy(cache);

    free(q);
    free(k);
    free(v);
    free(full_out);
    free(step_out);
}

static void test_gqa_parity(void) {
    size_t seq = 4;
    size_t n_heads_q = 4;
    size_t n_heads_kv = 2;
    size_t head_dim = 4;

    size_t hidden_q = n_heads_q * head_dim;
    size_t hidden_kv = n_heads_kv * head_dim;

    size_t total_q = seq * hidden_q;
    size_t total_kv = seq * hidden_kv;

    double* q = (double*)malloc(total_q * sizeof(double));
    double* k = (double*)malloc(total_kv * sizeof(double));
    double* v = (double*)malloc(total_kv * sizeof(double));

    double* k_exp = (double*)malloc(total_q * sizeof(double));
    double* v_exp = (double*)malloc(total_q * sizeof(double));

    double* out_naive = (double*)malloc(total_q * sizeof(double));
    double* out_kernel = (double*)malloc(total_q * sizeof(double));

    if (!q || !k || !v || !k_exp || !v_exp || !out_naive || !out_kernel) {
        CHECK(0, "gqa allocation");
        free(q); free(k); free(v);
        free(k_exp); free(v_exp);
        free(out_naive); free(out_kernel);
        return;
    }

    for (size_t i = 0; i < total_q; i++) {
        q[i] = frand();
    }

    for (size_t i = 0; i < total_kv; i++) {
        k[i] = frand();
        v[i] = frand();
    }

    size_t group_size = n_heads_q / n_heads_kv;

    for (size_t s = 0; s < seq; s++) {
        for (size_t hq = 0; hq < n_heads_q; hq++) {
            size_t hk = hq / group_size;

            for (size_t d = 0; d < head_dim; d++) {
                size_t q_idx = s * hidden_q + hq * head_dim + d;
                size_t kv_idx = s * hidden_kv + hk * head_dim + d;

                k_exp[q_idx] = k[kv_idx];
                v_exp[q_idx] = v[kv_idx];
            }
        }
    }

    naive_attention(out_naive, q, k_exp, v_exp, seq, n_heads_q, head_dim);
    kernel_gqa(out_kernel, q, k, v, seq, n_heads_q, n_heads_kv, head_dim);

    for (size_t i = 0; i < total_q; i++) {
        CHECK(close_d(out_kernel[i], out_naive[i], 1e-9), "gqa parity");
    }

    free(q);
    free(k);
    free(v);
    free(k_exp);
    free(v_exp);
    free(out_naive);
    free(out_kernel);
}

int main(void) {
    printf("================================================================\n");
    printf("  LANCIUS v11A2: TRANSFORMER KNOWN-ANSWER AUDIT\n");
    printf("================================================================\n");

    int before;

    before = failures; test_layernorm();              report("LayerNorm", before);
    before = failures; test_rmsnorm();               report("RMSNorm", before);
    before = failures; test_gelu();                  report("GELU", before);
    before = failures; test_swiglu();                report("SwiGLU", before);
    before = failures; test_rope();                  report("RoPE", before);
    before = failures; test_attention_full();        report("Full causal attention", before);
    before = failures; test_kv_cache_step_parity();  report("KV-cache step parity", before);
    before = failures; test_prefill_generation_parity(); report("Prefill + generation parity", before);
    before = failures; test_gqa_parity();            report("GQA parity", before);

    printf("================================================================\n");
    printf("  TRANSFORMER AUDIT COMPLETE: %d checks, %d failures\n", checks, failures);
    printf("================================================================\n");

    return failures ? 1 : 0;
}
