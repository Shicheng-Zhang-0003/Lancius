#include "lancius/lancius_train.h"
#include "lancius/lancius_error.h"

#include <math.h>
#include <stddef.h>

/* Private pi literal: M_PI is not standard C11. */
static const double LANCIUS_TRAIN_PI = 3.14159265358979323846;

/* LAPACK-style scaled sum of squares. `scale * sqrt(ssq)` is the norm, while
 * the pair remains useful even when that product exceeds DBL_MAX. */
static void lassq_add(double x, double *scale, double *ssq)
{
    double ax = fabs(x);
    if (ax == 0.0) return;
    if (*scale < ax) {
        double r = *scale / ax;
        *ssq = 1.0 + *ssq * r * r;
        *scale = ax;
    } else {
        double r = ax / *scale;
        *ssq += r * r;
    }
}

static double lassq_norm(double scale, double ssq)
{
    return scale == 0.0 ? 0.0 : scale * sqrt(ssq);
}

void lancius_sgd_step(double *w, const double *g, size_t n, double lr)
{
    size_t i;

    if (w == NULL || g == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return;
    }
    if (n == 0) {
        return;
    }
    if (!isfinite(lr)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return;
    }
    /* Exact: w -= lr*g */
    for (i = 0; i < n; ++i) {
        w[i] -= lr * g[i];
    }
}

void lancius_sgdm_step(double *w, double *m, const double *g, size_t n,
                       double lr, double momentum)
{
    size_t i;

    if (w == NULL || m == NULL || g == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return;
    }
    if (n == 0) {
        return;
    }
    if (!isfinite(lr) || !isfinite(momentum)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return;
    }
    /* Exact: m = momentum*m + g, w -= lr*m */
    for (i = 0; i < n; ++i) {
        m[i] = momentum * m[i] + g[i];
        w[i] -= lr * m[i];
    }
}

void lancius_adamw_step(double *w, double *m, double *v, const double *g, size_t n,
                        double lr, double b1, double b2, double eps, double wd, long t)
{
    double a1, a2, bc1, bc2;
    size_t i;

    if (w == NULL || m == NULL || v == NULL || g == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return;
    }
    if (n == 0) {
        return;
    }
    if (!isfinite(lr) || !isfinite(b1) || !isfinite(b2) ||
        !isfinite(eps) || !isfinite(wd)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return;
    }
    if (!(b1 >= 0.0 && b1 < 1.0) || !(b2 >= 0.0 && b2 < 1.0) ||
        !(eps > 0.0) || t < 1L) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return;
    }

    a1 = 1.0 - b1;
    a2 = 1.0 - b2;
    bc1 = 1.0 - pow(b1, (double)t);
    bc2 = 1.0 - pow(b2, (double)t);
    if (!(bc1 > 0.0) || !(bc2 > 0.0) || !isfinite(bc1) || !isfinite(bc2)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return;
    }

    for (i = 0; i < n; ++i) {
        double gi = g[i];
        double m_hat, v_hat;
        m[i] = b1 * m[i] + a1 * gi;
        v[i] = b2 * v[i] + a2 * gi * gi;
        m_hat = m[i] / bc1;
        v_hat = v[i] / bc2;
        /* Decoupled weight decay: w -= lr*m_hat/(sqrt(v_hat)+eps) + lr*wd*w */
        w[i] -= lr * (m_hat / (sqrt(v_hat) + eps)) + lr * wd * w[i];
    }
}

double lancius_clip_grad_norm(double *g, size_t n, double max_norm)
{
    double scale = 0.0, ssq = 1.0, norm;
    int has_nan = 0, has_inf = 0;
    size_t i;

    if (g == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return 0.0;
    }
    if (n == 0) {
        return 0.0;
    }
    for (i = 0; i < n; ++i) {
        if (isnan(g[i])) has_nan = 1;
        else if (isinf(g[i])) has_inf = 1;
        else lassq_add(g[i], &scale, &ssq);
    }
    norm = has_nan ? NAN : (has_inf ? INFINITY : lassq_norm(scale, ssq));
    if (!isfinite(max_norm) || !(max_norm > 0.0)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return norm;
    }
    if (has_nan || has_inf) return norm;
    if (!(scale > 0.0)) return norm;
    /* Compare without forming a possibly overflowing norm. */
    {
        double root_ssq = sqrt(ssq);
        if (scale > max_norm / root_ssq) {
            double clipped_scale = max_norm / root_ssq;
            /* Divide before multiplying so max_norm/scale cannot underflow
             * for a finite gradient near DBL_MAX and a small clip limit. */
            for (i = 0; i < n; ++i) g[i] = (g[i] / scale) * clipped_scale;
        }
    }
    return norm;
}

double lancius_clip_global_norm(double **gs, const size_t *ns, size_t ntensors,
                                 double max_norm)
{
    double scale = 0.0, ssq = 1.0, norm;
    int has_nan = 0, has_inf = 0;
    size_t t, i;

    if (ntensors == 0) {
        return 0.0;
    }
    if (gs == NULL || ns == NULL) {
        lancius_set_error(LANCIUS_ERROR_NULL_PTR);
        return 0.0;
    }
    for (t = 0; t < ntensors; ++t) {
        if (gs[t] == NULL && ns[t] != 0) {
            lancius_set_error(LANCIUS_ERROR_NULL_PTR);
            return 0.0;
        }
        for (i = 0; i < ns[t]; ++i) {
            if (isnan(gs[t][i])) has_nan = 1;
            else if (isinf(gs[t][i])) has_inf = 1;
            else lassq_add(gs[t][i], &scale, &ssq);
        }
    }
    norm = has_nan ? NAN : (has_inf ? INFINITY : lassq_norm(scale, ssq));
    if (!isfinite(max_norm) || !(max_norm > 0.0)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return norm;
    }
    if (has_nan || has_inf) return norm;
    if (!(scale > 0.0)) return norm;
    /* Compare without forming a possibly overflowing norm. */
    {
        double root_ssq = sqrt(ssq);
        if (scale > max_norm / root_ssq) {
            double clipped_scale = max_norm / root_ssq;
            for (t = 0; t < ntensors; ++t) {
                for (i = 0; i < ns[t]; ++i) {
                    gs[t][i] = (gs[t][i] / scale) * clipped_scale;
                }
            }
        }
    }
    return norm;
}

double lancius_lr_cosine(int step, int total, double lr_max, double lr_min)
{
    double progress;

    if (!isfinite(lr_max) || !isfinite(lr_min)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return lr_min;
    }
    if (total <= 0) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return lr_min;
    }
    if (step < 0) {
        step = 0;
    } else if (step > total) {
        step = total;
    }
    /* Exact: lr_min + 0.5*(max-min)*(1+cos(pi*step/total)) */
    progress = (double)step / (double)total;
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos(LANCIUS_TRAIN_PI * progress));
}

double lancius_lr_warmup_cosine(int step, int warmup, int total,
                                double lr_max, double lr_min)
{
    double progress;

    if (!isfinite(lr_max) || !isfinite(lr_min)) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return lr_min;
    }
    if (total <= 0) {
        lancius_set_error(LANCIUS_ERROR_NUMERICAL);
        return lr_min;
    }
    if (warmup < 0) {
        warmup = 0;
    }
    if (step < 0) {
        step = 0;
    }
    if (step >= total) {
        return lr_min;
    }
    /* A warmup that consumes the whole schedule has no cosine phase. Use the
     * well-defined cosine schedule rather than ramping to the peak then
     * jumping abruptly to lr_min at `total`. */
    if (warmup >= total) {
        return lancius_lr_cosine(step, total, lr_max, lr_min);
    }
    /* Linear warmup: lr_min -> lr_max over [0, warmup). */
    if (warmup > 0 && step < warmup) {
        return lr_min + (lr_max - lr_min) * ((double)step / (double)warmup);
    }
    /* Cosine decay over [warmup, total]: pi*(step-warmup)/(total-warmup). */
    progress = (double)(step - warmup) / (double)(total - warmup);
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos(LANCIUS_TRAIN_PI * progress));
}
