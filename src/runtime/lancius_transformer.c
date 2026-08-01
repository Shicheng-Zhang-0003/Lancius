#include "lancius/lancius_transformer.h"
#include "lancius/lancius_ir.h"
#include "lancius/lancius_kernels.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

struct lancius_kv_cache {
    size_t max_seq_len;
    size_t seq_len;
    size_t n_heads;
    size_t head_dim;
    size_t hidden_size;
    lancius_dtype dtype;

    /*
     * v11A2 initial implementation:
     * FP64 contiguous buffers:
     *
     *   [seq_len, n_heads, head_dim]
     */
    double* k;
    double* v;
};

lancius_kv_cache* lancius_kv_cache_create(
    size_t max_seq_len,
    size_t n_heads,
    size_t head_dim,
    lancius_dtype dtype
) {
    if (max_seq_len == 0 || n_heads == 0 || head_dim == 0) {
        return NULL;
    }

    /*
     * A2 initial KV-cache contract:
     * FP64 only. FP32 comes later in the A2 FP32 path.
     */
    if (dtype != LANCIUS_DTYPE_FP64) {
        return NULL;
    }

    /*
     * v11A2 Section 6:
     * RoPE requires an even head dimension.
     */
    if ((head_dim % 2) != 0) {
        return NULL;
    }

    if (n_heads > SIZE_MAX / head_dim) {
        return NULL;
    }

    size_t hidden_size = n_heads * head_dim;

    if (hidden_size == 0 || max_seq_len > SIZE_MAX / hidden_size) {
        return NULL;
    }

    lancius_kv_cache* cache = (lancius_kv_cache*)calloc(1, sizeof(lancius_kv_cache));
    if (!cache) {
        return NULL;
    }

    cache->max_seq_len = max_seq_len;
    cache->seq_len = 0;
    cache->n_heads = n_heads;
    cache->head_dim = head_dim;
    cache->hidden_size = hidden_size;
    cache->dtype = dtype;

    size_t total_elems = max_seq_len * hidden_size;

    cache->k = (double*)calloc(total_elems, sizeof(double));
    cache->v = (double*)calloc(total_elems, sizeof(double));

    if (!cache->k || !cache->v) {
        free(cache->k);
        free(cache->v);
        free(cache);
        return NULL;
    }

    return cache;
}

void lancius_kv_cache_destroy(lancius_kv_cache* cache) {
    if (!cache) {
        return;
    }

    free(cache->k);
    free(cache->v);
    free(cache);
}

void lancius_kv_cache_reset(lancius_kv_cache* cache) {
    if (!cache) {
        return;
    }

    cache->seq_len = 0;
}

int lancius_kv_cache_append(
    lancius_kv_cache* cache,
    const void* k,
    const void* v,
    size_t num_tokens
) {
    if (!cache || !k || !v || num_tokens == 0) {
        return -1;
    }

    if (cache->dtype != LANCIUS_DTYPE_FP64) {
        return -1;
    }

    if (num_tokens > cache->max_seq_len) {
        return -1;
    }

    if (cache->seq_len > cache->max_seq_len - num_tokens) {
        return -1;
    }

    if (cache->hidden_size > SIZE_MAX / num_tokens) {
        return -1;
    }

    size_t elems = num_tokens * cache->hidden_size;
    size_t bytes = elems * sizeof(double);

    double* k_dst = cache->k + (cache->seq_len * cache->hidden_size);
    double* v_dst = cache->v + (cache->seq_len * cache->hidden_size);

    memcpy(k_dst, k, bytes);
    memcpy(v_dst, v, bytes);

    cache->seq_len += num_tokens;

    return 0;
}

size_t lancius_kv_cache_seq_len(const lancius_kv_cache* cache) {
    return cache ? cache->seq_len : 0;
}

size_t lancius_kv_cache_max_seq_len(const lancius_kv_cache* cache) {
    return cache ? cache->max_seq_len : 0;
}

size_t lancius_kv_cache_n_heads(const lancius_kv_cache* cache) {
    return cache ? cache->n_heads : 0;
}

size_t lancius_kv_cache_head_dim(const lancius_kv_cache* cache) {
    return cache ? cache->head_dim : 0;
}

size_t lancius_kv_cache_hidden_size(const lancius_kv_cache* cache) {
    return cache ? cache->hidden_size : 0;
}

const void* lancius_kv_cache_k_buffer(
    const lancius_kv_cache* cache,
    size_t* active_seq_len
) {
    if (!cache) {
        return NULL;
    }

    if (active_seq_len) {
        *active_seq_len = cache->seq_len;
    }

    return cache->k;
}

const void* lancius_kv_cache_v_buffer(
    const lancius_kv_cache* cache,
    size_t* active_seq_len
) {
    if (!cache) {
        return NULL;
    }

    if (active_seq_len) {
        *active_seq_len = cache->seq_len;
    }

    return cache->v;
}

void lancius_node_bind_transformer_state(lancius_node* n, void* state) {
    if (!n || !n->rt) {
        return;
    }

    n->rt->transformer_state = state;
}

size_t lancius_kv_cache_next_position(const lancius_kv_cache* cache) {
    return cache ? cache->seq_len : 0;
}

int lancius_transformer_apply_rope_token(
    lancius_kv_cache* cache,
    double* q,
    double* k,
    int position
) {
    if (!cache || !q || !k || position < 0) {
        return -1;
    }

    if (cache->dtype != LANCIUS_DTYPE_FP64) {
        return -1;
    }

    if (cache->head_dim == 0 || (cache->head_dim % 2) != 0) {
        return -1;
    }

    kernel_rope(
        q,
        k,
        1,
        1,
        cache->n_heads,
        cache->head_dim,
        position
    );

    return 0;
}

int lancius_kv_cache_prefill(
    lancius_kv_cache* cache,
    const void* k,
    const void* v,
    size_t prompt_len
) {
    if (!cache || !k || !v || prompt_len == 0) {
        return -1;
    }

    return lancius_kv_cache_append(cache, k, v, prompt_len);
}

int lancius_kv_cache_append_generation_token(
    lancius_kv_cache* cache,
    const void* k,
    const void* v
) {
    if (!cache || !k || !v) {
        return -1;
    }

    return lancius_kv_cache_append(cache, k, v, 1);
}
