/* External-oracle dump probe.
 *
 * Calls every public math kernel DIRECTLY (no graph, no IR, no arena) and
 * writes inputs + outputs as raw little-endian binary under argv[1].
 * The Python oracle recomputes each quantity from NumPy / PyTorch and
 * compares. No Lancius math is reused on the oracle side.
 *
 * RNG: splitmix64, seeded per-block with an explicit integer recorded in the
 * .meta file, so Python can regenerate bit-identical inputs:
 *   state += 0x9E3779B97F4A7C15; z = state; z ^= z>>30; z *= 0xBF58476D1CE4E5B9;
 *   z ^= z>>27; z *= 0x94D049BB133111EB; z ^= z>>31;
 *   u = (z >> 11) * 2^-53   in [0,1)
 *   value = lo + (hi-lo)*u
 */
#include <lancius/lancius_kernels.h>
#include <lancius/lancius_error.h>
#include <lancius/lancius_train.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static const char* DIR = ".";

static FILE* of(const char* name, const char* suffix) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s%s", DIR, name, suffix);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "FATAL: cannot write %s\n", path); exit(2); }
    return f;
}

static void wmeta(const char* name, const size_t* dims, int nd) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.meta", DIR, name);
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "FATAL: cannot write %s\n", path); exit(2); }
    for (int i = 0; i < nd; i++) fprintf(f, "%zu%s", dims[i], i + 1 < nd ? " " : "\n");
    fclose(f);
}

static void wdbl(const char* name, const double* v, size_t n) {
    FILE* f = of(name, ".in.bin");
    fwrite(v, sizeof(double), n, f);
    fclose(f);
}

static void wbytes(const char* name, const void* v, size_t n) {
    FILE* f = of(name, ".in.bin");
    fwrite(v, 1, n, f);
    fclose(f);
}

static void odbl(const char* name, const double* v, size_t n) {
    FILE* f = of(name, ".out.bin");
    fwrite(v, sizeof(double), n, f);
    fclose(f);
}

static uint64_t sm_state;
static void rseed(uint64_t s) { sm_state = s; }
static double u01(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
static double* rndarr(size_t n, double lo, double hi) {
    double* v = (double*)malloc(n * sizeof(double));
    if (!v) { fprintf(stderr, "FATAL: oom\n"); exit(2); }
    for (size_t i = 0; i < n; i++) v[i] = lo + (hi - lo) * u01();
    return v;
}
static int8_t* rndi8(size_t n, int lo, int hi) {
    int8_t* v = (int8_t*)malloc(n);
    if (!v) { fprintf(stderr, "FATAL: oom\n"); exit(2); }
    for (size_t i = 0; i < n; i++) v[i] = (int8_t)(lo + (int)(u01() * (double)(hi - lo + 1)));
    return v;
}
static double* calloc_d(size_t n) {
    double* v = (double*)calloc(n ? n : 1, sizeof(double));
    if (!v) { fprintf(stderr, "FATAL: oom\n"); exit(2); }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <outdir>\n", argv[0]); return 2; }
    DIR = argv[1];

    /* ---- kernel_matmul ---- */
    {
        size_t M=7, K=5, N=11;
        rseed(1001);
        double* a = rndarr(M*K, -2, 2), *b = rndarr(K*N, -2, 2);
        double* o = calloc_d(M*N);
        kernel_matmul(o, a, b, M, K, N);
        wmeta("matmul", (size_t[]){M,K,N,M,N}, 5);
        wdbl("matmul_a", a, M*K); wdbl("matmul_b", b, K*N);
        odbl("matmul", o, M*N);
        free(a); free(b); free(o);
    }

    /* ---- kernel_matmul_f32 (FP32 in/out, FP64 accumulate) ---- */
    {
        size_t M=6, K=4, N=9;
        rseed(1002);
        double* ad = rndarr(M*K, -2, 2), *bd = rndarr(K*N, -2, 2);
        float* af = (float*)malloc(M*K*sizeof(float));
        float* bf = (float*)malloc(K*N*sizeof(float));
        float* o = (float*)malloc(M*N*sizeof(float));
        for (size_t i=0;i<M*K;i++) af[i]=(float)ad[i];
        for (size_t i=0;i<K*N;i++) bf[i]=(float)bd[i];
        kernel_matmul_f32(o, af, bf, M, K, N);
        wmeta("matmul_f32", (size_t[]){M,K,N,M,N}, 5);
        wdbl("matmul_f32_a", ad, M*K); wdbl("matmul_f32_b", bd, K*N);
        FILE* f = of("matmul_f32", ".out.bin"); fwrite(o, sizeof(float), M*N, f); fclose(f);
        free(ad); free(bd); free(af); free(bf); free(o);
    }

    /* ---- kernel_conv2d_fwd: stride 1 pad 1 ---- */
    {
        size_t N=2,Cin=3,Hin=7,Win=7,Cout=4,Kh=3,Kw=3,st=1,pad=1;
        size_t Hout=(Hin+2*pad-Kh)/st+1, Wout=(Win+2*pad-Kw)/st+1;
        rseed(1003);
        double* in = rndarr(N*Cin*Hin*Win, -1, 1), *w = rndarr(Cout*Cin*Kh*Kw, -1, 1);
        double* o = calloc_d(N*Cout*Hout*Wout);
        kernel_conv2d_fwd(o, in, w, N, Cin, Hin, Win, Cout, Kh, Kw, st, pad);
        wmeta("conv2d", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wdbl("conv2d_in", in, N*Cin*Hin*Win); wdbl("conv2d_w", w, Cout*Cin*Kh*Kw);
        odbl("conv2d", o, N*Cout*Hout*Wout);
        free(in); free(w); free(o);
    }

    /* ---- kernel_conv2d_fwd: stride 2 pad 0 ---- */
    {
        size_t N=1,Cin=2,Hin=8,Win=8,Cout=3,Kh=3,Kw=3,st=2,pad=0;
        size_t Hout=(Hin-Kh)/st+1, Wout=(Win-Kw)/st+1;
        rseed(1004);
        double* in = rndarr(N*Cin*Hin*Win, -1, 1), *w = rndarr(Cout*Cin*Kh*Kw, -1, 1);
        double* o = calloc_d(N*Cout*Hout*Wout);
        kernel_conv2d_fwd(o, in, w, N, Cin, Hin, Win, Cout, Kh, Kw, st, pad);
        wmeta("conv2d_s2", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wdbl("conv2d_s2_in", in, N*Cin*Hin*Win); wdbl("conv2d_s2_w", w, Cout*Cin*Kh*Kw);
        odbl("conv2d_s2", o, N*Cout*Hout*Wout);
        free(in); free(w); free(o);
    }

    /* ---- conv2d backwards: outputs to be checked by central differences ---- */
    {
        size_t N=2,Cin=2,Hin=5,Win=5,Cout=3,Kh=3,Kw=3,st=1,pad=1;
        size_t Hout=(Hin+2*pad-Kh)/st+1, Wout=(Win+2*pad-Kw)/st+1;
        rseed(1005);
        double* in = rndarr(N*Cin*Hin*Win, -1, 1), *w = rndarr(Cout*Cin*Kh*Kw, -1, 1);
        double* gout = rndarr(N*Cout*Hout*Wout, -1, 1);
        double* din = calloc_d(N*Cin*Hin*Win);
        double* dw  = calloc_d(Cout*Cin*Kh*Kw);
        kernel_conv2d_bwd_in(din, gout, w, N, Cin, Hin, Win, Cout, Hout, Wout, Kh, Kw, st, pad);
        kernel_conv2d_bwd_w(dw, gout, in, N, Cin, Hin, Win, Cout, Hout, Wout, Kh, Kw, st, pad);
        wmeta("conv_bwd", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wdbl("conv_bwd_in", in, N*Cin*Hin*Win);
        wdbl("conv_bwd_w_in", w, Cout*Cin*Kh*Kw);
        wdbl("conv_bwd_g", gout, N*Cout*Hout*Wout);
        odbl("conv_bwd_din", din, N*Cin*Hin*Win);
        odbl("conv_bwd_dw", dw, Cout*Cin*Kh*Kw);
        free(in); free(w); free(gout); free(din); free(dw);
    }

    /* ---- kernel_conv2d_relu_fwd ---- */
    {
        size_t N=2,Cin=2,Hin=6,Win=6,Cout=3,Kh=3,Kw=3,st=1,pad=1;
        size_t Hout=(Hin+2*pad-Kh)/st+1, Wout=(Win+2*pad-Kw)/st+1;
        rseed(1006);
        double* in = rndarr(N*Cin*Hin*Win, -1, 1), *w = rndarr(Cout*Cin*Kh*Kw, -1, 1);
        double* o = calloc_d(N*Cout*Hout*Wout);
        kernel_conv2d_relu_fwd(o, in, w, N, Cin, Hin, Win, Cout, Kh, Kw, st, pad);
        wmeta("conv_relu", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wdbl("conv_relu_in", in, N*Cin*Hin*Win); wdbl("conv_relu_w", w, Cout*Cin*Kh*Kw);
        odbl("conv_relu", o, N*Cout*Hout*Wout);
        free(in); free(w); free(o);
    }

    /* ---- kernel_conv2d_int8_fwd ---- */
    {
        size_t N=1,Cin=3,Hin=5,Win=5,Cout=4,Kh=3,Kw=3,st=1,pad=1;
        size_t Hout=(Hin+2*pad-Kh)/st+1, Wout=(Win+2*pad-Kw)/st+1;
        rseed(1007);
        int8_t* in = rndi8(N*Cin*Hin*Win, -100, 100);
        int8_t* w  = rndi8(Cout*Cin*Kh*Kw, -100, 100);
        double si = 0.021, sw = 0.037;
        double* o = calloc_d(N*Cout*Hout*Wout);
        kernel_conv2d_int8_fwd(o, in, w, si, sw, N, Cin, Hin, Win, Cout, Kh, Kw, st, pad);
        wmeta("conv_int8", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wbytes("conv_int8_in", in, N*Cin*Hin*Win);
        wbytes("conv_int8_w", w, Cout*Cin*Kh*Kw);
        { double s2[2] = {si, sw}; FILE* f = of("conv_int8_scales", ".in.bin"); fwrite(s2, sizeof(double), 2, f); fclose(f); }
        odbl("conv_int8", o, N*Cout*Hout*Wout);
        free(in); free(w); free(o);
    }

    /* ---- kernel_layernorm ---- */
    {
        size_t B=5,Hd=8;
        rseed(1008);
        double* x = rndarr(B*Hd,-2,2), *g = rndarr(Hd,0.5,1.5), *b = rndarr(Hd,-0.5,0.5);
        double* o = (double*)malloc(B*Hd*sizeof(double));
        kernel_layernorm(o, x, g, b, B, Hd, LANCIUS_NORM_EPS);
        wmeta("layernorm", (size_t[]){B,Hd}, 2);
        wdbl("layernorm_x", x, B*Hd); wdbl("layernorm_g", g, Hd); wdbl("layernorm_b", b, Hd);
        odbl("layernorm", o, B*Hd);
        free(x); free(g); free(b); free(o);
    }

    /* ---- kernel_layernorm_bwd / _gamma / _beta ---- */
    {
        size_t B=5,Hd=8;
        rseed(1009);
        double* x = rndarr(B*Hd,-2,2), *g = rndarr(Hd,0.5,1.5), *gy = rndarr(B*Hd,-1,1);
        double* dx = (double*)malloc(B*Hd*sizeof(double));
        double* dg = (double*)malloc(Hd*sizeof(double));
        double* db = (double*)malloc(Hd*sizeof(double));
        kernel_layernorm_bwd(dx, gy, x, g, B, Hd, LANCIUS_NORM_EPS);
        kernel_layernorm_bwd_gamma(dg, gy, x, g, B, Hd, LANCIUS_NORM_EPS);
        kernel_layernorm_bwd_beta(db, gy, B, Hd);
        wmeta("layernorm_bwd", (size_t[]){B,Hd}, 2);
        wdbl("lnb_x", x, B*Hd); wdbl("lnb_g", g, Hd); wdbl("lnb_gy", gy, B*Hd);
        odbl("layernorm_bwd_dx", dx, B*Hd);
        odbl("layernorm_bwd_dg", dg, Hd);
        odbl("layernorm_bwd_db", db, Hd);
        free(x); free(g); free(gy); free(dx); free(dg); free(db);
    }

    /* ---- kernel_rmsnorm + backwards ---- */
    {
        size_t B=4,Hd=7;
        rseed(1010);
        double* x = rndarr(B*Hd,-2,2), *g = rndarr(Hd,0.5,1.5), *gy = rndarr(B*Hd,-1,1);
        double* o = (double*)malloc(B*Hd*sizeof(double));
        double* dx = (double*)malloc(B*Hd*sizeof(double));
        double* dg = (double*)malloc(Hd*sizeof(double));
        kernel_rmsnorm(o, x, g, B, Hd, LANCIUS_NORM_EPS);
        kernel_rmsnorm_bwd(dx, gy, x, g, B, Hd, LANCIUS_NORM_EPS);
        kernel_rmsnorm_bwd_gamma(dg, gy, x, g, B, Hd, LANCIUS_NORM_EPS);
        wmeta("rmsnorm", (size_t[]){B,Hd}, 2);
        wdbl("rms_x", x, B*Hd); wdbl("rms_g", g, Hd); wdbl("rms_gy", gy, B*Hd);
        odbl("rmsnorm", o, B*Hd);
        odbl("rmsnorm_dx", dx, B*Hd);
        odbl("rmsnorm_dg", dg, Hd);
        free(x); free(g); free(gy); free(o); free(dx); free(dg);
    }

    /* ---- kernel_gelu + bwd over the clamp boundaries ---- */
    {
        size_t N=129;
        rseed(1011);
        double* x = (double*)malloc(N*sizeof(double));
        double* gy = rndarr(N,-1,1);
        for (size_t i=0;i<N;i++) x[i] = -12.0 + 24.0*(double)i/(double)(N-1);
        double* o = (double*)malloc(N*sizeof(double));
        double* gx = (double*)malloc(N*sizeof(double));
        kernel_gelu(o, x, N);
        kernel_gelu_bwd(gx, gy, x, N);
        wmeta("gelu", (size_t[]){N}, 1);
        wdbl("gelu_x", x, N); wdbl("gelu_gy", gy, N);
        odbl("gelu", o, N);
        odbl("gelu_bwd", gx, N);
        free(x); free(gy); free(o); free(gx);
    }

    /* ---- kernel_swiglu ---- */
    {
        size_t N=64;
        rseed(1012);
        double* gt = rndarr(N,-8,8), *up = rndarr(N,-2,2);
        double* o = (double*)malloc(N*sizeof(double));
        kernel_swiglu(o, gt, up, N);
        wmeta("swiglu", (size_t[]){N}, 1);
        wdbl("swiglu_g", gt, N); wdbl("swiglu_u", up, N);
        odbl("swiglu", o, N);
        free(gt); free(up); free(o);
    }

    /* ---- kernel_rope (in-place, non-zero position offset) ---- */
    {
        size_t B=2,S=4,H=3,D=8;
        int off = 3;
        rseed(1013);
        double* q0 = rndarr(B*S*H*D,-1,1), *k0 = rndarr(B*S*H*D,-1,1);
        double* q = (double*)malloc(B*S*H*D*sizeof(double));
        double* k = (double*)malloc(B*S*H*D*sizeof(double));
        memcpy(q, q0, B*S*H*D*sizeof(double));
        memcpy(k, k0, B*S*H*D*sizeof(double));
        kernel_rope(q, k, B, S, H, D, off);
        wmeta("rope", (size_t[]){B,S,H,D,(size_t)off}, 5);
        wdbl("rope_q0", q0, B*S*H*D); wdbl("rope_k0", k0, B*S*H*D);
        odbl("rope_q", q, B*S*H*D);
        odbl("rope_k", k, B*S*H*D);
        free(q0); free(k0); free(q); free(k);
    }

    /* ---- kernel_attention (causal, online-softmax flash) ---- */
    {
        size_t S=6,H=3,D=4;
        rseed(1014);
        double* q = rndarr(S*H*D,-1,1), *k = rndarr(S*H*D,-1,1), *v = rndarr(S*H*D,-1,1);
        double* o = (double*)malloc(S*H*D*sizeof(double));
        kernel_attention(o, q, k, v, S, H, D);
        wmeta("attention", (size_t[]){S,H,D}, 3);
        wdbl("attn_q", q, S*H*D); wdbl("attn_k", k, S*H*D); wdbl("attn_v", v, S*H*D);
        odbl("attention", o, S*H*D);
        free(q); free(k); free(v); free(o);
    }

    /* ---- kernel_gqa ---- */
    {
        size_t S=5,HQ=4,HKV=2,D=4;
        rseed(1015);
        double* q = rndarr(S*HQ*D,-1,1), *k = rndarr(S*HKV*D,-1,1), *v = rndarr(S*HKV*D,-1,1);
        double* o = (double*)malloc(S*HQ*D*sizeof(double));
        kernel_gqa(o, q, k, v, S, HQ, HKV, D);
        wmeta("gqa", (size_t[]){S,HQ,HKV,D}, 4);
        wdbl("gqa_q", q, S*HQ*D); wdbl("gqa_k", k, S*HKV*D); wdbl("gqa_v", v, S*HKV*D);
        odbl("gqa", o, S*HQ*D);
        free(q); free(k); free(v); free(o);
    }

    /* ---- kernel_attention_kv_cache ---- */
    {
        size_t S=7,H=3,D=4;
        rseed(1016);
        double* q = rndarr(H*D,-1,1), *kc = rndarr(S*H*D,-1,1), *vc = rndarr(S*H*D,-1,1);
        double* o = (double*)malloc(H*D*sizeof(double));
        kernel_attention_kv_cache(o, q, kc, vc, S, H, D);
        wmeta("kvcache", (size_t[]){S,H,D}, 3);
        wdbl("kvc_q", q, H*D); wdbl("kvc_k", kc, S*H*D); wdbl("kvc_v", vc, S*H*D);
        odbl("kvcache", o, H*D);
        free(q); free(kc); free(vc); free(o);
    }

    /* ---- trainers: SGD / SGDM / AdamW / global-norm clip / schedules ---- */
    {
        size_t n = 6;
        rseed(1017);
        double* w0 = rndarr(n,-1,1), *g = rndarr(n,-1,1);
        double* m = calloc_d(n), *v = calloc_d(n);
        double* w = (double*)malloc(n*sizeof(double));
        memcpy(w, w0, n*sizeof(double));
        for (int t=1;t<=20;t++)
            lancius_adamw_step(w, m, v, g, n, 0.001, 0.9, 0.999, 1e-8, 0.01, t);
        wmeta("adamw", (size_t[]){n}, 1);
        wdbl("adamw_w0", w0, n); wdbl("adamw_g", g, n);
        odbl("adamw_w", w, n);

        double* ws = (double*)malloc(n*sizeof(double));
        memcpy(ws, w0, n*sizeof(double));
        lancius_sgd_step(ws, g, n, 0.1);
        odbl("sgd_w", ws, n);

        double* ms = calloc_d(n);
        double* ws2 = (double*)malloc(n*sizeof(double));
        memcpy(ws2, w0, n*sizeof(double));
        lancius_sgdm_step(ws2, ms, g, n, 0.1, 0.9);
        odbl("sgdm_w", ws2, n);

        double* a1 = rndarr(4,-1,1);
        double* a2 = rndarr(3,-1,1);
        double* ptrs[2] = {a1, a2};
        size_t ns[2] = {4,3};
        double before = lancius_clip_global_norm(ptrs, ns, 2, 1.0);
        { double b1[1] = {before}; FILE* f = of("clip_norm", ".scalars"); fwrite(b1, sizeof(double), 1, f); fclose(f); }
        wdbl("clip_pre1", a1, 4);   /* inputs regenerated by the oracle; see oracle */
        odbl("clip_a1", a1, 4);
        odbl("clip_a2", a2, 3);

        free(w); free(w0); free(g); free(m); free(v); free(ws); free(ws2); free(ms);
        free(a1); free(a2);
    }
    {
        FILE* f = of("lr_cos", ".out.bin");
        for (int s=0;s<=10;s++){ double x=lancius_lr_cosine(s,10,0.1,0.0); fwrite(&x,sizeof(double),1,f); }
        for (int s=0;s<=10;s++){ double x=lancius_lr_warmup_cosine(s,3,10,0.1,0.0); fwrite(&x,sizeof(double),1,f); }
        for (int s=0;s<=10;s++){ double x=lancius_lr_cosine(s,10,0.1,0.01); fwrite(&x,sizeof(double),1,f); }
        for (int s=0;s<=10;s++){ double x=lancius_lr_warmup_cosine(s,0,10,0.1,0.01); fwrite(&x,sizeof(double),1,f); }
        fclose(f);
        wmeta("lr_cos", (size_t[]){44}, 1);
    }

    /* ---- GQA with all kv heads forced equal: proves the group mapping ---- */
    {
        size_t S=5,HQ=4,HKV=2,D=4;
        rseed(1018);
        double* q = rndarr(S*HQ*D,-1,1);
        double* k = rndarr(S*HKV*D,-1,1);
        double* v = rndarr(S*HKV*D,-1,1);
        /* force EVERY kv head (and position) to the same vector */
        for (size_t s=0;s<S;s++) for (size_t h=0;h<HKV;h++) for (size_t d=0;d<D;d++){
            k[(s*HKV+h)*D+d]=k[d]; v[(s*HKV+h)*D+d]=v[d];
        }
        double* o = (double*)malloc(S*HQ*D*sizeof(double));
        kernel_gqa(o, q, k, v, S, HQ, HKV, D);
        wmeta("gqa_uniform", (size_t[]){S,HQ,HKV,D}, 4);
        wdbl("gqa_uniform_q", q, S*HQ*D);
        wdbl("gqa_uniform_k", k, S*HKV*D);
        wdbl("gqa_uniform_v", v, S*HKV*D);
        odbl("gqa_uniform", o, S*HQ*D);
        free(q); free(k); free(v); free(o);
    }

    /* ---- single decode step vs the last row of full causal attention ---- */
    {
        size_t S=7,H=3,D=4;
        rseed(1019);
        double* qall = rndarr(S*H*D,-1,1), *k = rndarr(S*H*D,-1,1), *v = rndarr(S*H*D,-1,1);
        double* ofull = (double*)malloc(S*H*D*sizeof(double));
        kernel_attention(ofull, qall, k, v, S, H, D);
        double* ostep = (double*)malloc(H*D*sizeof(double));
        kernel_attention_kv_cache(ostep, qall + (S-1)*H*D, k, v, S, H, D);
        wmeta("kv_step", (size_t[]){S,H,D}, 3);
        wdbl("kv_step_q", qall + (S-1)*H*D, H*D);
        wdbl("kv_step_k", k, S*H*D);
        wdbl("kv_step_v", v, S*H*D);
        odbl("kv_step", ostep, H*D);
        odbl("kv_full", ofull, S*H*D);
        free(qall); free(k); free(v); free(ofull); free(ostep);
    }

    printf("kernel_oracle_dump: dumps in %s, final err=%d\n", DIR, (int)lancius_get_error());
    return 0;
}