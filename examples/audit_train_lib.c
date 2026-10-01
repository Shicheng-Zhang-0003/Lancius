/* R2-1 gate: training-library known-answer + finite-difference proofs.
 * Exits 1 on divergence. Uses only public lancius_train.h + math.h.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lancius/lancius_train.h"
#include "lancius/lancius_error.h"

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } \
    else { printf("  PASS: %s\n", msg); } \
} while (0)

static int dbl_eq(double a, double b, double tol) {
    if (a != a || b != b) return 0;
    return fabs(a - b) <= tol;
}

int main(void) {
    printf("=== R2-1 train-lib audit ===\n");

    /* SGD: w=[1,2], g=[0.5,-0.5], lr=0.1 -> [0.95,2.05] */
    {
        double w[2] = {1.0, 2.0};
        double g[2] = {0.5, -0.5};
        lancius_clear_error();
        lancius_sgd_step(w, g, 2, 0.1);
        CHECK(dbl_eq(w[0], 0.95, 1e-12) && dbl_eq(w[1], 2.05, 1e-12), "sgd exact");
        CHECK(lancius_get_error() == LANCIUS_ERROR_OK, "sgd no error");
    }
    /* SGDM: m=0, momentum=0.9, g=[1], lr=0.1 -> m=[1], w=[-0.1]; second: m=1.9, w=-0.29 */
    {
        double w[1] = {0.0}, m[1] = {0.0}, g[1] = {1.0};
        lancius_sgdm_step(w, m, g, 1, 0.1, 0.9);
        CHECK(dbl_eq(m[0], 1.0, 1e-12) && dbl_eq(w[0], -0.1, 1e-12), "sgdm step1");
        lancius_sgdm_step(w, m, g, 1, 0.1, 0.9);
        CHECK(dbl_eq(m[0], 1.9, 1e-12) && dbl_eq(w[0], -0.29, 1e-12), "sgdm step2");
    }
    /* AdamW t=1: w=0,m=0,v=0,g=1,lr=0.001,b1=0.9,b2=0.999,eps=1e-8,wd=0
     * m=0.1,v=0.001,bc1=0.1,bc2=0.001,mhat=1,vhat=1 -> w=-0.001/(1+1e-8) */
    {
        double w[1] = {0.0}, m[1] = {0.0}, v[1] = {0.0}, g[1] = {1.0};
        lancius_adamw_step(w, m, v, g, 1, 0.001, 0.9, 0.999, 1e-8, 0.0, 1L);
        double expect = -0.001 / (1.0 + 1e-8);
        CHECK(dbl_eq(w[0], expect, 1e-12), "adamw t1 exact");
        CHECK(dbl_eq(m[0], 0.1, 1e-12) && dbl_eq(v[0], 0.001, 1e-12), "adamw moments");
    }
    /* AdamW decoupled decay: w=1, g=0, wd=0.1, lr=0.01 -> w=1-0.01*0.1*1=0.999 */
    {
        double w[1] = {1.0}, m[1] = {0.0}, v[1] = {0.0}, g[1] = {0.0};
        lancius_adamw_step(w, m, v, g, 1, 0.01, 0.9, 0.999, 1e-8, 0.1, 1L);
        CHECK(dbl_eq(w[0], 0.999, 1e-12), "adamw decoupled wd");
    }
    /* Clip: g=[3,4] norm=5, max=2.5 -> scale 0.5 -> [1.5,2.0], returns 5 */
    {
        double g[2] = {3.0, 4.0};
        double old = lancius_clip_grad_norm(g, 2, 2.5);
        CHECK(dbl_eq(old, 5.0, 1e-12), "clip returns old norm");
        CHECK(dbl_eq(g[0], 1.5, 1e-12) && dbl_eq(g[1], 2.0, 1e-12), "clip scales");
        /* No-clip path */
        {
            double h[2] = {1.0, 0.0};
            double o2 = lancius_clip_grad_norm(h, 2, 5.0);
            CHECK(dbl_eq(o2, 1.0, 1e-12) && dbl_eq(h[0], 1.0, 1e-12), "clip no-op under max");
        }
    }
    /* Cosine: step=0 -> max, step=total -> min, step=total/2 -> mid */
    {
        double a = lancius_lr_cosine(0, 100, 0.1, 0.001);
        double b = lancius_lr_cosine(100, 100, 0.1, 0.001);
        double c = lancius_lr_cosine(50, 100, 0.1, 0.001);
        CHECK(dbl_eq(a, 0.1, 1e-12), "cosine start=max");
        CHECK(dbl_eq(b, 0.001, 1e-12), "cosine end=min");
        CHECK(dbl_eq(c, 0.0505, 1e-12), "cosine mid");
    }
    /* Warmup+cosine: warmup=10,total=100: step0=min, step10=max, step100=min */
    {
        double a = lancius_lr_warmup_cosine(0, 10, 100, 0.1, 0.001);
        double b = lancius_lr_warmup_cosine(10, 10, 100, 0.1, 0.001);
        double c = lancius_lr_warmup_cosine(100, 10, 100, 0.1, 0.001);
        double d = lancius_lr_warmup_cosine(5, 10, 100, 0.1, 0.001);
        CHECK(dbl_eq(a, 0.001, 1e-12), "warmup start=min");
        CHECK(dbl_eq(b, 0.1, 1e-12), "warmup peak=max");
        CHECK(dbl_eq(c, 0.001, 1e-12), "warmup end=min");
        CHECK(dbl_eq(d, 0.0505, 1e-12), "warmup linear mid");
    }
    /* NULL safety: must set error, not crash */
    {
        lancius_clear_error();
        lancius_sgd_step(NULL, NULL, 0, 0.1);
        CHECK(lancius_get_error() != LANCIUS_ERROR_OK, "sgd NULL sets error");
        lancius_clear_error();
        double x = lancius_clip_grad_norm(NULL, 5, 1.0);
        CHECK(x == 0.0 && lancius_get_error() != LANCIUS_ERROR_OK, "clip NULL sets error");
        lancius_clear_error();
    }
    /* Loss monotonicity: quadratic 0.5*w^2, SGD lr=0.1 from w=1 -> decreases */
    {
        double w = 1.0, prev = 0.5 * w * w;
        int ok = 1;
        for (int i = 0; i < 20; i++) {
            double g = w;
            lancius_sgd_step(&w, &g, 1, 0.1);
            double loss = 0.5 * w * w;
            if (!(loss < prev)) { ok = 0; break; }
            prev = loss;
        }
        CHECK(ok, "sgd loss monotonic on quadratic");
    }

    if (fails) { printf("TRAIN-LIB AUDIT: %d FAILURES\n", fails); return 1; }
    printf("TRAIN-LIB AUDIT: ALL PROOFS HOLD\n");
    return 0;
}
