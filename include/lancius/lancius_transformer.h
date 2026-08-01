#ifndef LANCIUS_TRANSFORMER_H
#define LANCIUS_TRANSFORMER_H

#include <stddef.h>
#include "lancius/lancius_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * v11A2 transformer runtime support.
 *
 * This introduces the first real stateful transformer runtime object:
 * the KV-cache.
 *
 * A2 contract:
 *   - graph shapes remain static
 *   - active sequence length lives in the cache object
 *   - attention execution may consult the cache instead of mutated IR shapes
 */

typedef struct lancius_kv_cache lancius_kv_cache;

lancius_kv_cache* lancius_kv_cache_create(
    size_t max_seq_len,
    size_t n_heads,
    size_t head_dim,
    lancius_dtype dtype
);

void lancius_kv_cache_destroy(lancius_kv_cache* cache);

void lancius_kv_cache_reset(lancius_kv_cache* cache);

/*
 * Append num_tokens tokens to the cache.
 *
 * For v11A2 initially only FP64 is supported.
 * k and v must each contain:
 *
 *   num_tokens * n_heads * head_dim
 *
 * elements.
 *
 * Returns 0 on success, non-zero on failure.
 */
int lancius_kv_cache_append(
    lancius_kv_cache* cache,
    const void* k,
    const void* v,
    size_t num_tokens
);

size_t lancius_kv_cache_seq_len(const lancius_kv_cache* cache);
size_t lancius_kv_cache_max_seq_len(const lancius_kv_cache* cache);
size_t lancius_kv_cache_n_heads(const lancius_kv_cache* cache);
size_t lancius_kv_cache_head_dim(const lancius_kv_cache* cache);
size_t lancius_kv_cache_hidden_size(const lancius_kv_cache* cache);

const void* lancius_kv_cache_k_buffer(
    const lancius_kv_cache* cache,
    size_t* active_seq_len
);

const void* lancius_kv_cache_v_buffer(
    const lancius_kv_cache* cache,
    size_t* active_seq_len
);

/*
 * Attach a transformer runtime object to a node.
 *
 * For v11A2 this is used to bind a KV-cache to an ATTENTION node.
 */
struct lancius_node;
void lancius_node_bind_transformer_state(struct lancius_node* n, void* state);


/*
 * v11A2 Section 6:
 * RoPE position contract.
 *
 * The cache's active sequence length is the authoritative position source
 * for autoregressive generation.
 */
size_t lancius_kv_cache_next_position(const lancius_kv_cache* cache);

int lancius_transformer_apply_rope_token(
    lancius_kv_cache* cache,
    double* q,
    double* k,
    int position
);


/*
 * v11A2 Section 4:
 * Explicit prefill and generation cache paths.
 */
int lancius_kv_cache_prefill(
    lancius_kv_cache* cache,
    const void* k,
    const void* v,
    size_t prompt_len
);

int lancius_kv_cache_append_generation_token(
    lancius_kv_cache* cache,
    const void* k,
    const void* v
);

#ifdef __cplusplus
}
#endif

#endif /* LANCIUS_TRANSFORMER_H */
