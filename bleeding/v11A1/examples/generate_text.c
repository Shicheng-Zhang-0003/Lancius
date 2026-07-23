#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

const char* VOCAB[] = {"[PAD]", "Hello", " world", "!", " Lancius", " is", " fast", ".", "\n", " AI"};
#define VOCAB_SIZE 10

int main() {
    printf("================================================================\n");
    printf("  Lancius v10S: AUTOREGRESSIVE LLM GENERATION (KV-Cache)      \n");
    printf("================================================================\n\n");

    size_t n_heads = 2;
    size_t head_dim = 4;
    size_t hidden_size = n_heads * head_dim;
    size_t max_seq_len = 32;

    // 1. Allocate Stateful KV-Cache (Outside the Arena!)
    double* k_cache = (double*)calloc(max_seq_len * hidden_size, sizeof(double));
    double* v_cache = (double*)calloc(max_seq_len * hidden_size, sizeof(double));

    // Build a mock 1-layer Transformer Graph
    lancius_graph* g = lancius_graph_create();

    // Inputs: Q (1 token), K (cache), V (cache)
    lancius_node* Q_in = lancius_input(g, 1, hidden_size);
    Q_in->ndim = 3; Q_in->shape[0] = 1; Q_in->shape[1] = n_heads; Q_in->shape[2] = head_dim;

    lancius_node* K_in = lancius_input(g, 1, hidden_size); // Shape updated dynamically
    K_in->ndim = 3; K_in->shape[0] = 1; K_in->shape[1] = n_heads; K_in->shape[2] = head_dim;

    lancius_node* V_in = lancius_input(g, 1, hidden_size);
    V_in->ndim = 3; V_in->shape[0] = 1; V_in->shape[1] = n_heads; V_in->shape[2] = head_dim;

    // Attention Node
    lancius_node* attn = lancius_attention(g, Q_in, K_in, V_in);

    // LayerNorm & GELU
    lancius_node* gamma = lancius_input(g, 1, hidden_size);
    lancius_node* beta = lancius_input(g, 1, hidden_size);
    lancius_node* ln = lancius_layernorm(g, attn, gamma, beta);
    lancius_node* gelu = lancius_gelu(g, ln);
    (void)gelu; // Suppress unused variable warning

    // Allocate persistent weights for LN
    gamma->runtime_data = (double*)malloc(hidden_size * sizeof(double));
    beta->runtime_data = (double*)calloc(hidden_size, sizeof(double));
    for(size_t i=0; i<hidden_size; i++) gamma->runtime_data[i] = 1.0;

    printf("[1/2] Compiling Transformer Schedule...\n");
    lancius_schedule* sched = lancius_ir_schedule(g);
    lancius_arena* scratch = lancius_arena_create(1024 * 1024); // 1MB scratch

    // Mock initial prompt: "Hello" (token 1)
    int prompt[] = {1};
    size_t seq_len = 0;

    printf("Prompt: %s", VOCAB[prompt[0]]);

    // 2. Autoregressive Generation Loop
    for(int step=0; step<8; step++) {
        // Mock Projections for current token
        double* q = (double*)malloc(hidden_size * sizeof(double));
        double* k = (double*)malloc(hidden_size * sizeof(double));
        double* v = (double*)malloc(hidden_size * sizeof(double));

        for(size_t i=0; i<hidden_size; i++) {
            q[i] = 0.5 + 0.1 * step;
            k[i] = 0.5 - 0.1 * step;
            v[i] = 1.0;
        }

        // Apply RoPE to Q and K
        kernel_rope(q, k, 1, 1, n_heads, head_dim, seq_len);

        // Bounds check to prevent heap overflow
        if (seq_len >= max_seq_len) {
            printf("\n[WARN] Max sequence length (%zu) reached. Stopping generation.\n", max_seq_len);
            break;
        }
        // Append K and V to Stateful Cache
        memcpy(k_cache + seq_len * hidden_size, k, hidden_size * sizeof(double));
        memcpy(v_cache + seq_len * hidden_size, v, hidden_size * sizeof(double));
        seq_len++;

        // Update IR Graph Inputs to point to persistent cache
        Q_in->runtime_data = q;
        K_in->runtime_data = k_cache;
        V_in->runtime_data = v_cache;

        // DYNAMIC SHAPE UPDATE: Tell the scheduler the new sequence length
        K_in->shape[0] = seq_len;
        V_in->shape[0] = seq_len;
        attn->shape[0] = 1; // Output is always 1 token
        attn->shape[1] = n_heads;
        attn->shape[2] = head_dim;

        // Nullify intermediates to force Arena allocation
        for(uint32_t w=0; w<sched->wave_count; w++) {
            for(uint32_t i=0; i<sched->waves[w].node_count; i++) {
                lancius_node* n = sched->waves[w].nodes[i];
                if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
            }
        }

        // Execute Forward Pass
        lancius_schedule_execute(sched, scratch);

        // Mock Output Projection to Logits (VOCAB_SIZE)
        double* logits = (double*)calloc(VOCAB_SIZE, sizeof(double));
        for(int i=0; i<VOCAB_SIZE; i++) {
            if (i == (step + 2) % VOCAB_SIZE) logits[i] = 5.0; // Favor specific token
            else logits[i] = -1.0;
        }

        // Argmax Sampling
        int next_token = 0;
        double max_logit = -1e9;
        for(int i=0; i<VOCAB_SIZE; i++) {
            if (logits[i] > max_logit) { max_logit = logits[i]; next_token = i; }
        }

        printf("%s", VOCAB[next_token]);
        fflush(stdout);

        free(q); free(k); free(v); free(logits);
        lancius_arena_reset(scratch); // O(1) Teardown for intermediates!
    }

    printf("\n\n================================================================\n");
    printf("  LANCIUS v10 AUTOREGRESSIVE GENERATION COMPLETE.\n");
    printf("  Final Sequence Length: %zu tokens\n", seq_len);
    printf("================================================================\n");

    free(k_cache); free(v_cache);
    free(gamma->runtime_data); free(beta->runtime_data);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    lancius_arena_destroy(scratch);
    return 0;
}
