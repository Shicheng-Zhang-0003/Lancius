#include "lancius/lancius_validate.h"
#include "lancius/lancius_checked.h"
#include "lancius/lancius_ir.h"

lancius_error lancius_validate_rank(uint8_t ndim) {
    if (ndim == 0 || ndim > 4) return LANCIUS_ERROR_INVALID_RANK;
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_shape(const size_t* shape, uint8_t ndim) {
    lancius_error e = lancius_validate_rank(ndim);
    if (e != LANCIUS_ERROR_OK) return e;
    if (!shape) return LANCIUS_ERROR_NULL_PTR;
    for (uint8_t i = 0; i < ndim; i++) {
        if (shape[i] == 0) return LANCIUS_ERROR_INVALID_SHAPE;
    }
    size_t elems = 0;
    if (!lancius_checked_product_shape(shape, ndim, &elems)) return LANCIUS_ERROR_LIMIT;
    if (elems > LANCIUS_MAX_TENSOR_ELEMS) return LANCIUS_ERROR_LIMIT;
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_permutation(const uint32_t* axes, uint8_t rank) {
    if (!axes) return LANCIUS_ERROR_NULL_PTR;
    if (rank == 0 || rank > 4) return LANCIUS_ERROR_INVALID_RANK;
    uint8_t seen[4] = {0, 0, 0, 0};
    for (uint8_t i = 0; i < rank; i++) {
        uint32_t a = axes[i];
        if (a >= rank) return LANCIUS_ERROR_INVALID_PERMUTATION;
        if (seen[a]) return LANCIUS_ERROR_INVALID_PERMUTATION;
        seen[a] = 1;
    }
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_matmul(const size_t* a_shape, uint8_t a_ndim,
                                      const size_t* b_shape, uint8_t b_ndim) {
    if (!a_shape || !b_shape) return LANCIUS_ERROR_NULL_PTR;
    if (a_ndim < 2 || b_ndim < 2) return LANCIUS_ERROR_INVALID_RANK;
    if (a_shape[a_ndim - 1] != b_shape[b_ndim - 2]) return LANCIUS_ERROR_SHAPE_MISMATCH;
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_reshape(size_t in_elems, const size_t* out_shape, uint8_t out_ndim) {
    if (!out_shape) return LANCIUS_ERROR_NULL_PTR;
    lancius_error e = lancius_validate_rank(out_ndim);
    if (e != LANCIUS_ERROR_OK) return e;
    size_t out_elems = 0;
    if (!lancius_checked_product_shape(out_shape, out_ndim, &out_elems)) return LANCIUS_ERROR_LIMIT;
    if (out_elems != in_elems) return LANCIUS_ERROR_RESHAPE_MISMATCH;
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_binary_same_shape(const size_t* a, const size_t* b, uint8_t ndim) {
    if (!a || !b) return LANCIUS_ERROR_NULL_PTR;
    lancius_error e = lancius_validate_rank(ndim);
    if (e != LANCIUS_ERROR_OK) return e;
    for (uint8_t i = 0; i < ndim; i++) {
        if (a[i] != b[i]) return LANCIUS_ERROR_SHAPE_MISMATCH;
    }
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_conv2d(size_t H_in, size_t W_in, size_t K_h, size_t K_w,
                                      uint32_t stride, uint32_t pad) {
    if (stride == 0) return LANCIUS_ERROR_INVALID_STRIDE;
    if (K_h == 0 || K_w == 0) return LANCIUS_ERROR_INVALID_SHAPE;
    /* Checked: H_in + 2*pad must not wrap and must cover kernel. */
    if (pad > (SIZE_MAX - H_in) / 2) return LANCIUS_ERROR_OVERFLOW;
    if (pad > (SIZE_MAX - W_in) / 2) return LANCIUS_ERROR_OVERFLOW;
    if (H_in + 2 * (size_t)pad < K_h) return LANCIUS_ERROR_INVALID_SHAPE;
    if (W_in + 2 * (size_t)pad < K_w) return LANCIUS_ERROR_INVALID_SHAPE;
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_gqa(uint32_t n_heads_q, uint32_t n_heads_kv) {
    if (n_heads_kv == 0) return LANCIUS_ERROR_INVALID_SHAPE;
    if (n_heads_q < n_heads_kv) return LANCIUS_ERROR_INVALID_SHAPE;
    if (n_heads_q % n_heads_kv != 0) return LANCIUS_ERROR_INVALID_SHAPE;
    return LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_stride_nonzero(uint32_t stride) {
    return stride == 0 ? LANCIUS_ERROR_INVALID_STRIDE : LANCIUS_ERROR_OK;
}

lancius_error lancius_validate_binary_broadcast(const lancius_node* a, const lancius_node* b) {
    if (!a || !b) return LANCIUS_ERROR_NULL_PTR;
    if (a->ndim != b->ndim) return LANCIUS_ERROR_SHAPE_MISMATCH;
    for (uint8_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] && a->shape[i] != 1 && b->shape[i] != 1) {
            return LANCIUS_ERROR_SHAPE_MISMATCH;
        }
    }
    return LANCIUS_ERROR_OK;
}
