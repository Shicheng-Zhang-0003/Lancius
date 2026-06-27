#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    double* t5;
    double* t6;
    double* t7;
    double* t8;
    double* t9;
} GhostContext;

void ghost_weight_forward(double* X, double* W1, double* b1, double* W2, double* b2, GhostContext* ctx, double* out) {
    ctx->t5 = (double*)malloc(4 * 8 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) {
        double sum = 0.0;
        for(size_t k=0; k<2; k++) sum += X[i*2 + k] * W1[k*8 + j];
        ctx->t5[i*8 + j] = sum;
    }
    ctx->t6 = (double*)malloc(4 * 8 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) ctx->t6[i*8 + j] = ctx->t5[i*8 + j] + b1[j];
    ctx->t7 = (double*)malloc(4 * 8 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) { double v = ctx->t6[i*8 + j]; ctx->t7[i*8 + j] = v > 0.0 ? v : 0.0; }
    ctx->t8 = (double*)malloc(4 * 1 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<1; j++) {
        double sum = 0.0;
        for(size_t k=0; k<8; k++) sum += ctx->t7[i*8 + k] * W2[k*1 + j];
        ctx->t8[i*1 + j] = sum;
    }
    ctx->t9 = (double*)malloc(4 * 1 * sizeof(double));
    for(size_t i=0; i<4; i++) for(size_t j=0; j<1; j++) ctx->t9[i*1 + j] = ctx->t8[i*1 + j] + b2[j];
    memcpy(out, ctx->t9, 4 * 1 * sizeof(double));
}

void ghost_weight_backward(double* X, double* W1, double* b1, double* W2, double* b2, GhostContext* ctx, double* grad_out, double* grad_X, double* grad_W1, double* grad_b1, double* grad_W2, double* grad_b2) {
    double* grad_t5 = (double*)calloc(4 * 8, sizeof(double));
    double* grad_t6 = (double*)calloc(4 * 8, sizeof(double));
    double* grad_t7 = (double*)calloc(4 * 8, sizeof(double));
    double* grad_t8 = (double*)calloc(4 * 1, sizeof(double));
    double* grad_t9 = grad_out;
    for(size_t i=0; i<4; i++) for(size_t j=0; j<1; j++) grad_t8[i*1+j] += grad_t9[i*1+j];
    for(size_t j=0; j<1; j++) { double s=0; for(size_t i=0; i<4; i++) s += grad_t9[i*1+j]; grad_b2[j] += s; }
    for(size_t i=0; i<4; i++) for(size_t k=0; k<8; k++) { double s=0; for(size_t j=0; j<1; j++) s += grad_t8[i*1+j] * W2[k*1+j]; grad_t7[i*8+k] += s; }
    for(size_t k=0; k<8; k++) for(size_t j=0; j<1; j++) { double s=0; for(size_t i=0; i<4; i++) s += ctx->t7[i*8+k] * grad_t8[i*1+j]; grad_W2[k*1+j] += s; }
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) grad_t6[i*8+j] += grad_t7[i*8+j] * (ctx->t7[i*8+j] > 0.0 ? 1.0 : 0.0);
    for(size_t i=0; i<4; i++) for(size_t j=0; j<8; j++) grad_t5[i*8+j] += grad_t6[i*8+j];
    for(size_t j=0; j<8; j++) { double s=0; for(size_t i=0; i<4; i++) s += grad_t6[i*8+j]; grad_b1[j] += s; }
    for(size_t i=0; i<4; i++) for(size_t k=0; k<2; k++) { double s=0; for(size_t j=0; j<8; j++) s += grad_t5[i*8+j] * W1[k*8+j]; grad_X[i*2+k] += s; }
    for(size_t k=0; k<2; k++) for(size_t j=0; j<8; j++) { double s=0; for(size_t i=0; i<4; i++) s += X[i*2+k] * grad_t5[i*8+j]; grad_W1[k*8+j] += s; }
    free(grad_t5);
    free(grad_t6);
    free(grad_t7);
    free(grad_t8);
}

void ghost_free_context(GhostContext* ctx) {
    free(ctx->t5);
    free(ctx->t6);
    free(ctx->t7);
    free(ctx->t8);
    free(ctx->t9);
}
