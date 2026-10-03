#ifndef LANCIUS_TRAIN_H
#define LANCIUS_TRAIN_H

#include <math.h>
#include <stddef.h>

/*
 * Minimal honest C training library: SGD / SGDM / AdamW / grad-clip / LR schedules.
 *
 * Contract (all functions):
 * - Pure CPU, double precision, no abort/exit/assert, no allocation, no statics.
 * - Bad pointers / bad hypers call lancius_set_error() and return safely
 *   (void fns: no-op; clip: return 0.0 on NULL, else old norm unscaled;
 *   schedules: return lr_min).
 * - n == 0 with valid pointers is a silent no-op (clip returns 0.0).
 * - NaN/Inf inside w/m/v/g arrays is propagated, never hidden.
 */

void lancius_sgd_step(double *w, const double *g, size_t n, double lr);

void lancius_sgdm_step(double *w, double *m, const double *g, size_t n,
                       double lr, double momentum);

void lancius_adamw_step(double *w, double *m, double *v, const double *g, size_t n,
                        double lr, double b1, double b2, double eps, double wd, long t);

/* Scales g in place iff norm > max_norm. Returns old (pre-scale) norm. */
double lancius_clip_grad_norm(double *g, size_t n, double max_norm);

/* R3-3 global-norm clip over ntensors gradient arrays: total norm
 * sqrt(sum ||g_i||^2); scales every array in place iff total > max_norm.
 * Returns old total norm. Same contract: no alloc/abort, NULL-safe
 * (NULL array or ns with valid ntensors sets error, returns 0.0),
 * ntensors == 0 is a no-op returning 0.0, NaN/Inf never rescaled. */
double lancius_clip_global_norm(double **gs, const size_t *ns, size_t ntensors,
                                 double max_norm);

/* lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*step/total)), step clamped to [0,total]. */
double lancius_lr_cosine(int step, int total, double lr_max, double lr_min);

/*
 * Linear warmup then cosine:
 *   step < warmup : lr_min + (lr_max-lr_min)*step/warmup
 *   step >= warmup: lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*(step-warmup)/(total-warmup)))
 * warmup <= 0 degenerates to lancius_lr_cosine(). step clamped to [0,total].
 */
double lancius_lr_warmup_cosine(int step, int warmup, int total,
                                double lr_max, double lr_min);

#endif /* LANCIUS_TRAIN_H */
