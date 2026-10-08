/* Bottom-up audit, layer 1: the four train-lib entry points that section 17
 * never checked against a primary source (sgd_step, sgdm_step, clip_grad_norm,
 * lr_warmup_cosine). Emits results for tools/audit/trainlib_oracle.py to check
 * against torch and HuggingFace, neither of which links or imports Lancius.
 */
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "temp/scratch/trainlib";
    char p[512];
    snprintf(p, sizeof p, "%s.meta", out);
    FILE* m = fopen(p, "w"); if (!m) { fprintf(stderr, "cannot write %s\n", p); return 2; }
    fprintf(m, "cases 1\n");

    /* --- SGD: plain and momentum, over a fixed gradient sweep --- */
    {
        snprintf(p, sizeof p, "%s.sgd.w", out);
        FILE* f = fopen(p, "w");
        enum { N = 6, T = 8 };
        double w0[N] = {0.5, -1.25, 3.0, 0.0, -0.75, 2.5};
        const double LRS[3] = {0.1, 0.01, 1.0};
        const double MOMS[3] = {0.0, 0.9, 0.99};
        for (int li = 0; li < 3; li++) {
          for (int mi = 0; mi < 3; mi++) {
                double lr = LRS[li], mom = MOMS[mi];
                double w[N], buf[N];
                for (int i = 0; i < N; i++) { w[i] = w0[i]; buf[i] = 0.0; }
                for (int t = 0; t < T; t++) {
                    double g[N];
                    for (int i = 0; i < N; i++)
                        g[i] = sin(1.7 * (i + 1) + 0.3 * t) * (1.0 + 0.1 * i);
                    if (mom == 0.0) lancius_sgd_step(w, g, N, lr);
                    else            lancius_sgdm_step(w, buf, g, N, lr, mom);
                }
                /* emit the ELEMENT INDEX too: the first version wrote all N
                 * elements without it and the oracle compared every row
                 * against element N-1, which manufactured large fake failures
                 * for the configs whose weights diverge fastest. */
                for (int i = 0; i < N; i++)
                    fprintf(f, "%.17g %.17g %.17g %d %d %d\n", lr, mom, w[i],
                            (int)(lr * 100), (int)(mom * 100), i);
          }
        }
        fclose(f);
    }

    /* --- clip_grad_norm: sweep gradients across the max_norm boundary --- */
    {
        snprintf(p, sizeof p, "%s.clip", out);
        FILE* f = fopen(p, "w");
        enum { CN = 5, CT = 12 };
        for (int t = 0; t < CT; t++) {
            double g[CN]; double pre = 0.0;
            /* sweep the pre-clip norm across max_norm so the boundary is hit */
            double scale = pow(10.0, -1.0 + 0.2 * t);
            for (int i = 0; i < CN; i++) {
                g[i] = scale * cos(2.3 * (i + 1) + 0.7 * t);
                pre += g[i] * g[i];
            }
            pre = sqrt(pre);
            double max_norm = 1.0;
            /* lancius_clip_grad_norm clips IN PLACE, so keep a pre-clip copy.
             * Without this the dump reports pre-clip norms alongside post-clip
             * gradients, and an external oracle handed those gradients would
             * compare the wrong quantity -- my first version did exactly that
             * and produced four large, entirely fictional failures. */
            double g0[CN];
            for (int i = 0; i < CN; i++) g0[i] = g[i];
            double returned = lancius_clip_grad_norm(g, CN, max_norm);
            double post = 0.0;
            for (int i = 0; i < CN; i++) post += g[i] * g[i];
            post = sqrt(post);
            fprintf(f, "%.17g %.17g %.17g %d\n", pre, returned, post, t);
            for (int i = 0; i < CN; i++) fprintf(f, "g%d %.17g\n", i, g0[i]);
        }
        fclose(f);
    }

    /* --- lr_warmup_cosine: dense sweep, two lr_min settings --- */
    {
        snprintf(p, sizeof p, "%s.lr", out);
        FILE* f = fopen(p, "w");
        enum { warmup = 5, total = 20 };
        const double lrmax = 1e-3, lrmin = 1e-5;
        for (int step = 0; step <= total + 2; step++)
            fprintf(f, "%d %.17g\n", step,
                    lancius_lr_warmup_cosine(step, warmup, total, lrmax, lrmin));
        /* lr_min == 0 is the common case and the one HF's default matches */
        for (int step = 0; step <= total + 2; step++)
            fprintf(f, "z%d %.17g\n", step,
                    lancius_lr_warmup_cosine(step, warmup, total, lrmax, 0.0));
        fclose(f);
    }
    fclose(m);
    printf("trainlib dump written to %s\n", out);
    return 0;
}