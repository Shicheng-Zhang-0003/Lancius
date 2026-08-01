#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

const char* VOCAB[] = {"[PAD]", "Hello", " world", "!", " Lancius", " is", " fast", ".", "\n", " AI"};
#define VOCAB_SIZE 10

int main(void) {
    printf("================================================================\n");
    printf("  Lancius v11A2: EXPLICIT PREFILL + GENERATION DEMO\n");
    printf("================================================================\n");

    size_t n_heads = 2;
    size_t head_dim = 4;
    size_t hidden_size = n_heads * head_dim;
    size_t prompt_len = 2;
    size_t max_seq_len = 16;
    size_t gen_steps = 6;

    /*
     * v11A2 Section 2:
     * Stateful KV-cache runtime object.
     */
    lancius_kv_cache* cache = lancius_kv_cache_create(
        max_seq_len,
        n_heads,
        head_dim,
        LANCIUS_DTYPE_FP64
    );

    if (!cache) {
        printf("FATAL: could not create KV-cache\n");
        return 1;
    }

    /*
     * Prompt buffers.
     */
    size_t prompt_elems = prompt_len * hidden_size;

    double* q_prompt = (double*)malloc(prompt_elems * sizeof(double));
    double* k_prompt = (double*)malloc(prompt_elems * sizeof(double));
    double* v_prompt = (double*)malloc(prompt_elems * sizeof(double));

    if (!q_prompt || !k_prompt || !v_prompt) {
        printf("FATAL: OOM allocating prompt buffers\n");
        return 1;
    }

    for (size_t t = 0; t < prompt_len; t++) {
        for (size_t i = 0; i < hidden_size; i++) {
            q_prompt[t * hidden_size + i] = 0.25 + 0.11 * t;
            k_prompt[t * hidden_size + i] = 0.31 - 0.07 * t;
            v_prompt[t * hidden_size + i] = 1.0 + 0.05 * t;
        }
    }

    /*
     * v11A2 Section 6:
     * Prompt RoPE uses positions 0..prompt_len-1.
     */
    kernel_rope(
        q_prompt,
        k_prompt,
        1,
        prompt_len,
        n_heads,
        head_dim,
        0
    );

    int prompt_tokens[] = {1, 4};

    printf("Prompt:");
    for (size_t i = 0; i < prompt_len; i++) {
        printf(" %s", VOCAB[prompt_tokens[i]]);
    }
    printf("\n");

    /*
     * v11A2 Section 4:
     * Explicit prefill path.
     *
     * Prefill executes full causal attention over the prompt, then the
     * prompt K/V tensors are appended to the KV-cache in one operation.
     */
    printf("[1/3] Running explicit prefill path...\n");

    lancius_graph* gp = lancius_graph_create();

    lancius_node* Qp = lancius_input_3d(gp, prompt_len, n_heads, head_dim);
    lancius_node* Kp = lancius_input_3d(gp, prompt_len, n_heads, head_dim);
    lancius_node* Vp = lancius_input_3d(gp, prompt_len, n_heads, head_dim);

    lancius_node* attn_p = lancius_attention(gp, Qp, Kp, Vp);
    (void)attn_p;

    lancius_node_bind_external(Qp, q_prompt);
    lancius_node_bind_external(Kp, k_prompt);
    lancius_node_bind_external(Vp, v_prompt);

    lancius_schedule* sp = lancius_ir_schedule(gp);
    lancius_arena* scratch = lancius_arena_create(1024 * 1024);

    if (!sp || !scratch) {
        printf("FATAL: could not compile prefill schedule\n");
        return 1;
    }

    lancius_schedule_execute(sp, scratch);

    if (lancius_kv_cache_prefill(cache, k_prompt, v_prompt, prompt_len) != 0) {
        printf("FATAL: prefill cache append failed\n");
        return 1;
    }

    printf("  Prefill complete. Cache seq_len = %zu\n",
           lancius_kv_cache_seq_len(cache));

    lancius_arena_reset(scratch);
    lancius_schedule_destroy(sp);
    lancius_graph_destroy(gp);

    /*
     * v11A2 Section 4:
     * Explicit generation path.
     *
     * The generation graph is static.
     * Q is always one token.
     * K/V are cache-backed external buffers.
     * Active sequence length is read from the cache object.
     */
    printf("[2/3] Compiling generation graph...\n");

    lancius_graph* g = lancius_graph_create();

    lancius_node* Q_in = lancius_input_3d(g, 1, n_heads, head_dim);
    lancius_node* K_in = lancius_input_3d(g, max_seq_len, n_heads, head_dim);
    lancius_node* V_in = lancius_input_3d(g, max_seq_len, n_heads, head_dim);

    lancius_node* attn = lancius_attention(g, Q_in, K_in, V_in);

    /*
     * v11A2 Section 3:
     * Bind transformer runtime state to the attention node.
     */
    lancius_node_bind_transformer_state(attn, cache);

    lancius_node* gamma = lancius_input(g, 1, hidden_size);
    lancius_node* beta = lancius_input(g, 1, hidden_size);

    lancius_node* ln = lancius_layernorm(g, attn, gamma, beta);
    lancius_node* gelu = lancius_gelu(g, ln);
    (void)gelu;

    double* gamma_data = (double*)malloc(hidden_size * sizeof(double));
    double* beta_data = (double*)calloc(hidden_size, sizeof(double));

    if (!gamma_data || !beta_data) {
        printf("FATAL: OOM allocating LayerNorm parameters\n");
        return 1;
    }

    for (size_t i = 0; i < hidden_size; i++) {
        gamma_data[i] = 1.0;
    }

    lancius_node_bind_external(gamma, gamma_data);
    lancius_node_bind_external(beta, beta_data);

    /*
     * Bind K/V graph inputs to cache buffers.
     */
    lancius_node_bind_external(K_in, (void*)lancius_kv_cache_k_buffer(cache, NULL));
    lancius_node_bind_external(V_in, (void*)lancius_kv_cache_v_buffer(cache, NULL));

    lancius_schedule* sched = lancius_ir_schedule(g);

    if (!sched) {
        printf("FATAL: could not compile generation schedule\n");
        return 1;
    }

    double* q = (double*)malloc(hidden_size * sizeof(double));
    double* k = (double*)malloc(hidden_size * sizeof(double));
    double* v = (double*)malloc(hidden_size * sizeof(double));

    if (!q || !k || !v) {
        printf("FATAL: OOM allocating token buffers\n");
        return 1;
    }

    printf("[3/3] Generating tokens...\n");

    for (size_t step = 0; step < gen_steps; step++) {
        /*
         * v11A2 Section 6:
         * Position comes from the cache, not from demo bookkeeping.
         */
        size_t position = lancius_kv_cache_next_position(cache);

        if (position >= max_seq_len) {
            printf("\n[WARN] max sequence length reached. Stopping generation.\n");
            break;
        }

        for (size_t i = 0; i < hidden_size; i++) {
            q[i] = 0.51 + 0.03 * step;
            k[i] = 0.44 - 0.02 * step;
            v[i] = 1.0;
        }

        if (lancius_transformer_apply_rope_token(cache, q, k, (int)position) != 0) {
            printf("\n[WARN] RoPE helper failed. Stopping generation.\n");
            break;
        }

        if (lancius_kv_cache_append_generation_token(cache, k, v) != 0) {
            printf("\n[WARN] KV-cache append failed. Stopping generation.\n");
            break;
        }

        /*
         * Static graph.
         * No IR shape mutation.
         * No manual intermediate nulling.
         */
        lancius_node_bind_external(Q_in, q);
        lancius_schedule_execute(sched, scratch);

        /*
         * Mock output projection / sampling.
         */
        double* logits = (double*)calloc(VOCAB_SIZE, sizeof(double));
        if (!logits) {
            printf("\nFATAL: OOM allocating logits\n");
            break;
        }

        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (i == (int)((step + 2) % VOCAB_SIZE)) {
                logits[i] = 5.0;
            } else {
                logits[i] = -1.0;
            }
        }

        int next_token = 0;
        double max_logit = -1e9;

        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token = i;
            }
        }

        printf("%s", VOCAB[next_token]);
        fflush(stdout);

        free(logits);
        lancius_arena_reset(scratch);
    }

    printf("\n================================================================\n");
    printf("  LANCIUS v11A2 PREFILL/GENERATION DEMO COMPLETE.\n");
    printf("  Final Sequence Length: %zu tokens\n",
           lancius_kv_cache_seq_len(cache));
    printf("================================================================\n");

    free(q);
    free(k);
    free(v);

    free(q_prompt);
    free(k_prompt);
    free(v_prompt);

    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    lancius_arena_destroy(scratch);

    free(gamma_data);
    free(beta_data);

    lancius_kv_cache_destroy(cache);

    return 0;
}
