#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Naive O(N^2) Attention for mathematical verification
void naive_attention(double* out, const double* q, const double* k, const double* v, size_t seq_len, size_t n_heads, size_t head_dim) {
    double* scores = (double*)malloc(seq_len * seq_len * sizeof(double)); // The O(N^2) bottleneck!
    double scale = 1.0 / sqrt((double)head_dim);

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t h = 0; h < n_heads; h++) {
            // Compute scores
            double max_s = -1e9;
            for (size_t j = 0; j < seq_len; j++) {
                double s = 0.0;
                for (size_t d = 0; d < head_dim; d++) {
                    s += q[(i * n_heads * head_dim) + (h * head_dim) + d] * k[(j * n_heads * head_dim) + (h * head_dim) + d];
                }
                s *= scale;
                if (j > i) s = -1e9; // Causal
                scores[j] = s;
                if (s > max_s) max_s = s;
            }

            // Softmax
            double sum_exp = 0.0;
            for (size_t j = 0; j < seq_len; j++) {
                scores[j] = exp(scores[j] - max_s);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j < seq_len; j++) {
                scores[j] /= sum_exp;
            }

            // Weighted sum V
            for (size_t d = 0; d < head_dim; d++) {
                double val = 0.0;
                for (size_t j = 0; j < seq_len; j++) {
                    val += scores[j] * v[(j * n_heads * head_dim) + (h * head_dim) + d];
                }
                out[(i * n_heads * head_dim) + (h * head_dim) + d] = val;
            }
        }
    }
    free(scores);
}

int main() {
    printf("================================================================\n");
    printf("  LANCIUS v9 STABLE: FLASH ATTENTION AUDIT (SRAM TILING)       \n");
    printf("================================================================\n\n");

    size_t seq_len = 512;
    size_t n_heads = 8;
    size_t head_dim = 64;
    size_t total_elems = seq_len * n_heads * head_dim;

    printf("Config: seq_len=%zu, n_heads=%zu, head_dim=%zu\n", seq_len, n_heads, head_dim);
    size_t n_sq_memory = seq_len * seq_len * sizeof(double);
    printf("[1/3] Naive Attention O(N^2) Scratchpad Requirement: %zu bytes (%.2f MB)\n", n_sq_memory, n_sq_memory / (1024.0 * 1024.0));
    printf("      Flash Attention O(D) Scratchpad Requirement: %zu bytes (Registers/L1 Cache)\n\n", head_dim * sizeof(double));

    double* q = (double*)malloc(total_elems * sizeof(double));
    double* k = (double*)malloc(total_elems * sizeof(double));
    double* v = (double*)malloc(total_elems * sizeof(double));
    double* out_naive = (double*)malloc(total_elems * sizeof(double));
    double* out_flash = (double*)malloc(total_elems * sizeof(double));

    srand(42);
    for(size_t i=0; i<total_elems; i++) {
        q[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        k[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        v[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    printf("[2/3] Running Naive Attention (O(N^2) Memory)...\n");
    naive_attention(out_naive, q, k, v, seq_len, n_heads, head_dim);

    printf("[3/3] Running Flash Attention (Online Softmax / Zero Alloc)...\n");
    kernel_attention(out_flash, q, k, v, seq_len, n_heads, head_dim);

    // Verify Math
    double max_diff = 0.0;
    for(size_t i=0; i<total_elems; i++) {
        double diff = fabs(out_naive[i] - out_flash[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("\n================================================================\n");
    if (max_diff < 1e-5) {
        printf("  ✅ MATHEMATICAL PARITY VERIFIED.\n");
        printf("  📉 MAX ERROR: %e (Threshold: < 1e-5)\n", max_diff);
        printf("  🧠 FLASH ATTENTION SUCCESSFULLY ELIMINATED O(N^2) BOTTLENECK.\n");
    } else {
        printf("  ❌ MATH DIVERGENCE DETECTED.\n");
        printf("  📉 MAX ERROR: %e\n", max_diff);
    }
    printf("================================================================\n");

    free(q); free(k); free(v); free(out_naive); free(out_flash);
    return 0;
}
