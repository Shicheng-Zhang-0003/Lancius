/* Graph-level oracle dump: drives the public graph API (IR builders +
 * scheduler + vision ops + autodiff) and writes every input/output so the
 * Python oracle can compare against NumPy / PyTorch.
 *
 * Every op under test is exercised through lancius_graph_* + lancius_schedule,
 * which is the path the CLI, the trainers and the audits all use.
 *
 * Buffer ownership follows the documented pattern (examples/audit_known_answer.c):
 * inputs are bound EXTERNALLY, execution uses one arena that is never reset
 * mid-dump, and every output is written to disk before the graph is destroyed.
 */
#include <lancius.h>
#include <lancius/lancius_autodiff.h>
#include <lancius/lancius_kernels.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static const char* D;
static lancius_arena* SCRATCH = NULL;

static void wf(const char* name, const void* v, size_t bytes) {
    char p[512];
    snprintf(p, sizeof(p), "%s/%s", D, name);
    FILE* f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "FATAL: %s\n", p); exit(2); }
    fwrite(v, 1, bytes, f);
    fclose(f);
}
static void wd(const char* name, const double* v, size_t n) { wf(name, v, n * sizeof(double)); }
static void wi(const char* name, const int64_t* v, size_t n) { wf(name, v, n * sizeof(int64_t)); }
static void wm(const char* name, const size_t* dims, int nd) {
    char p[512];
    snprintf(p, sizeof(p), "%s/%s", D, name);
    FILE* f = fopen(p, "w");
    if (!f) { fprintf(stderr, "FATAL: %s\n", p); exit(2); }
    for (int i = 0; i < nd; i++) fprintf(f, "%zu%s", dims[i], i + 1 < nd ? " " : "\n");
    fclose(f);
}

static uint64_t sm;
static void rseed(uint64_t s) { sm = s; }
static double u01(void) {
    uint64_t z = (sm += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}
static double* rndarr(size_t n, double lo, double hi) {
    double* v = (double*)malloc(n * sizeof(double));
    if (!v) { fprintf(stderr, "FATAL oom\n"); exit(2); }
    for (size_t i = 0; i < n; i++) v[i] = lo + (hi - lo) * u01();
    return v;
}

static int checks = 0, failures = 0;
#define REQUIRE(cond, msg) do { checks++; if(!(cond)) { printf("  PROBE FAIL: %s (err=%d)\n", msg, (int)lancius_get_error()); failures++; } } while (0)

static void bind(lancius_node* n, const double* src) {
    lancius_node_bind_external(n, (void*)src);
}

static int run(lancius_graph* g) {
    if (!SCRATCH) SCRATCH = lancius_arena_create(16u * 1024u * 1024u);
    lancius_clear_error();
    lancius_schedule* s = lancius_ir_schedule(g);
    if (!s) return 0;
    lancius_schedule_execute(s, SCRATCH);
    int err = (int)lancius_get_error();
    lancius_schedule_destroy(s);
    return err == 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <outdir>\n", argv[0]); return 2; }
    D = argv[1];

    /* ---------------------------------------------------------- softmax */
    fprintf(stderr, "[block] softmax\n");
    {
        size_t R = 4, C = 6;
        rseed(2001);
        double* z  = rndarr(R*C, -3, 3);
        double* z2 = (double*)malloc(R*C*sizeof(double));
        for (size_t i=0;i<R*C;i++) z2[i] = z[i] + 7.5;
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input(g, R, C);
        lancius_node* sm = lancius_softmax(g, in);
        REQUIRE(sm != NULL, "softmax node built");
        bind(in, z);
        REQUIRE(run(g), "softmax executed");
        wd("softmax_in.bin", z, R*C);
        wd("softmax_out.bin", sm->runtime_data, R*C);
        lancius_graph_destroy(g);
        {
            /* fresh graph: shift-invariance must hold for the SAME logits+7.5 */
            lancius_graph* g2 = lancius_graph_create();
            lancius_node* i2 = lancius_input(g2, R, C);
            lancius_node* s2 = lancius_softmax(g2, i2);
            bind(i2, z2);
            REQUIRE(run(g2), "softmax executed (shifted)");
            wd("softmax_out2.bin", s2->runtime_data, R*C);
            lancius_graph_destroy(g2);
        }
        free(z); free(z2);
    }

    /* ---------------------------------------------------- cross entropy */
    fprintf(stderr, "[block] cross_entropy\n");
    {
        size_t R = 5, C = 3;
        rseed(2002);
        double* lg = rndarr(R*C, -2, 2);
        int64_t lab[5] = {0, 2, 1, 2, 0};
        double* lb = (double*)malloc(R*sizeof(double));
        for (size_t i=0;i<R;i++) lb[i] = (double)lab[i];
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input(g, R, C);
        /* targets are one-hot [R,C] -- CROSS_ENTROPY requires matching shape */
        double* oh = (double*)calloc(R*C, sizeof(double));
        for (size_t i=0;i<R;i++) oh[i*C + lab[i]] = 1.0;
        (void)lb;
        lancius_node* tn = lancius_input(g, R, C);
        bind(in, lg); bind(tn, oh);
        lancius_node* ce = lancius_cross_entropy(g, in, tn);
        REQUIRE(ce != NULL, "cross_entropy node built");
        REQUIRE(run(g), "cross_entropy executed");
        wd("ce_logits.bin", lg, R*C);
        wi("ce_labels.bin", lab, R);
        wd("ce_loss.bin", ce->runtime_data, 1);
        lancius_graph_destroy(g);
        free(lg); free(lb); free(oh);
    }

    /* -------------------------------------------------------------- mse */
    fprintf(stderr, "[block] mse\n");
    {
        size_t R = 4, C = 5;
        rseed(2003);
        double* p = rndarr(R*C,-1,1), *y = rndarr(R*C,-1,1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* pn = lancius_input(g, R, C), *yn = lancius_input(g, R, C);
        bind(pn, p); bind(yn, y);
        lancius_node* m = lancius_mse(g, pn, yn);
        REQUIRE(m != NULL, "mse node built");
        REQUIRE(run(g), "mse executed");
        wd("mse_p.bin", p, R*C); wd("mse_y.bin", y, R*C);
        wd("mse_loss.bin", m->runtime_data, 1);
        lancius_graph_destroy(g);
        free(p); free(y);
    }

    /* --------------------------------------------- broadcast arithmetic */
    fprintf(stderr, "[block] broadcast\n");
    {
        rseed(2004);
        double* a = rndarr(8,-1,1), *b = rndarr(24,-1,1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* an = lancius_input_3d(g, 2, 1, 4);
        lancius_node* bn = lancius_input_3d(g, 2, 3, 4);
        bind(an, a); bind(bn, b);
        lancius_node* o = lancius_add(g, an, bn);
        REQUIRE(o != NULL, "3D broadcast add built");
        REQUIRE(run(g), "3D broadcast add executed");
        wd("bcast_a.bin", a, 8); wd("bcast_b.bin", b, 24);
        wd("bcast_out.bin", o->runtime_data, 24);
        lancius_graph_destroy(g); free(a); free(b);
    }
    {
        size_t N=3, C=4;
        rseed(2005);
        double* a = rndarr(N*C,-1,1), *b = rndarr(N*C,-1,1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* an = lancius_input(g, N, C), *bn = lancius_input(g, N, C);
        bind(an, a); bind(bn, b);
        lancius_node* o = lancius_mul(g, an, bn);
        REQUIRE(o != NULL, "2D mul built");
        REQUIRE(run(g), "2D mul executed");
        wd("bcast2_a.bin", a, N*C); wd("bcast2_b.bin", b, N*C);
        wd("bcast2_out.bin", o->runtime_data, N*C);
        lancius_graph_destroy(g); free(a); free(b);
    }
    {
        size_t N=2,C=3,H=4,W=5;
        rseed(2006);
        double* a = rndarr(N*W,-1,1), *b = rndarr(N*C*H*W,-1,1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* an = lancius_input_4d(g, N, 1, 1, W);
        lancius_node* bn = lancius_input_4d(g, N, C, H, W);
        bind(an, a); bind(bn, b);
        lancius_node* o = lancius_sub(g, an, bn);
        REQUIRE(o != NULL, "4D broadcast sub built");
        REQUIRE(run(g), "4D broadcast sub executed");
        wd("bcast4_a.bin", a, N*W); wd("bcast4_b.bin", b, N*C*H*W);
        wd("bcast4_out.bin", o->runtime_data, N*C*H*W);
        lancius_graph_destroy(g); free(a); free(b);
    }

    /* ------------------------------------------------------- SUM_AXIS_ND */
    fprintf(stderr, "[block] sum_axis_nd\n");
    {
        struct { const char* tag; size_t dims[4]; int nd; uint32_t axis; } cases[] = {
            {"sumax0",  {5,4,1,1}, 2, 0},
            {"sumax1",  {5,4,1,1}, 2, 1},
            {"sumax3d", {2,3,4,1}, 3, 2},
            {"sumax4d", {2,3,4,5}, 4, 3},
        };
        for (unsigned ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
            size_t nelem = 1, oe = 1;
            for (int i = 0; i < cases[ci].nd; i++) nelem *= cases[ci].dims[i];
            for (int i = 0; i < cases[ci].nd; i++)
                oe *= (i == (int)cases[ci].axis) ? 1 : cases[ci].dims[i];
            rseed(3000 + ci);
            double* x = rndarr(nelem, -2, 2);
            lancius_graph* g = lancius_graph_create();
            lancius_node* in;
            if (cases[ci].nd == 2)      in = lancius_input(g, cases[ci].dims[0], cases[ci].dims[1]);
            else if (cases[ci].nd == 3) in = lancius_input_3d(g, cases[ci].dims[0], cases[ci].dims[1], cases[ci].dims[2]);
            else                        in = lancius_input_4d(g, cases[ci].dims[0], cases[ci].dims[1], cases[ci].dims[2], cases[ci].dims[3]);
            bind(in, x);
            lancius_node* o = lancius_sum_axis_nd(g, in, cases[ci].axis);
            REQUIRE(o != NULL, cases[ci].tag);
            REQUIRE(run(g), cases[ci].tag);
            char nm[128];
            snprintf(nm, sizeof(nm), "%s_in.bin", cases[ci].tag);  wd(nm, x, nelem);
            snprintf(nm, sizeof(nm), "%s_out.bin", cases[ci].tag); wd(nm, o->runtime_data, oe);
            lancius_graph_destroy(g); free(x);
        }
    }

    /* --------------------------------------------------- matmul batched */
    fprintf(stderr, "[block] matmul_batched\n");
    {
        size_t B=3,M=4,K=5,N=6;
        rseed(2007);
        double* a = rndarr(B*M*K,-1,1), *b = rndarr(B*K*N,-1,1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* an = lancius_input_3d(g, B, M, K);
        lancius_node* bn = lancius_input_3d(g, B, K, N);
        bind(an, a); bind(bn, b);
        lancius_node* o = lancius_matmul_batched(g, an, bn);
        REQUIRE(o != NULL, "matmul_batched built");
        REQUIRE(run(g), "matmul_batched executed");
        wd("mmb_a.bin", a, B*M*K); wd("mmb_w.bin", b, B*K*N);
        wd("mmb_out.bin", o->runtime_data, B*M*N);
        wm("mmb.meta", (size_t[]){B,M,K,N}, 4);
        lancius_graph_destroy(g); free(a); free(b);
    }

    /* --------------------------------------------------------- permute */
    fprintf(stderr, "[block] permute\n");
    {
        size_t N=2,C=3,H=4,W=5;
        rseed(2008);
        double* x = rndarr(N*C*H*W,-1,1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input_4d(g, N, C, H, W);
        bind(in, x);
        lancius_node* o = lancius_permute(g, in, 0, 2, 1, 3);
        REQUIRE(o != NULL, "permute built");
        REQUIRE(run(g), "permute executed");
        wd("perm_in.bin", x, N*C*H*W);
        wd("perm_out.bin", o->runtime_data, N*C*H*W);
        lancius_graph_destroy(g); free(x);
    }

    /* --------------------------------------------- conv backward (FD) */
    fprintf(stderr, "[block] conv_bwd\n");
    {
        size_t N=1,Cin=2,Hin=5,Win=5,Cout=3,Kh=3,Kw=3,st=1,pad=1;
        size_t Hout=(Hin+2*pad-Kh)/st+1, Wout=(Win+2*pad-Kw)/st+1;
        rseed(2009);
        double* in = rndarr(N*Cin*Hin*Win,-1,1);
        double* w  = rndarr(Cout*Cin*Kh*Kw,-1,1);
        double* g  = rndarr(N*Cout*Hout*Wout,-1,1);
        lancius_graph* gph = lancius_graph_create();
        lancius_node* in_n = lancius_input_4d(gph, N, Cin, Hin, Win);
        lancius_node* w_n  = lancius_input_4d(gph, Cout, Cin, Kh, Kw);
        bind(in_n, in); bind(w_n, w);
        /* the incoming gradient is an ordinary node bound to g */
        lancius_node* gn = lancius_input_4d(gph, N, Cout, Hout, Wout);
        bind(gn, g);
        lancius_node* dgi = lancius_conv2d_bwd(gph, gn, in_n, w_n, st, pad);
        lancius_node* dgw = lancius_conv2d_bwd_w(gph, gn, in_n, Kh, Kw, st, pad);
        REQUIRE(dgi != NULL, "conv2d_bwd built");
        REQUIRE(dgw != NULL, "conv2d_bwd_w built");
        REQUIRE(run(gph), "conv bwd graph executed");
        wm("convbwd.meta", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wd("convbwd_in.bin", in, N*Cin*Hin*Win);
        wd("convbwd_w.bin", w, Cout*Cin*Kh*Kw);
        wd("convbwd_g.bin", g, N*Cout*Hout*Wout);
        wd("convbwd_din.bin", dgi->runtime_data, N*Cin*Hin*Win);
        wd("convbwd_dw.bin", dgw->runtime_data, Cout*Cin*Kh*Kw);
        lancius_graph_destroy(gph); free(in); free(w); free(g);
    }

    /* ------------------------------------------------------- maxpool2d */
    fprintf(stderr, "[block] maxpool2d\n");
    {
        size_t N=1,C=1,H=6,W=6,pool=2;
        rseed(2010);
        double* x = rndarr(N*C*H*W,-1,1);
        double* gp = rndarr(N*C*(H/pool)*(W/pool), -1, 1);
        lancius_graph* g = lancius_graph_create();
        lancius_node* in = lancius_input_4d(g, N, C, H, W);
        bind(in, x);
        lancius_node* o = lancius_maxpool2d(g, in, pool, pool);
        REQUIRE(o != NULL, "maxpool2d built");
        REQUIRE(run(g), "maxpool2d forward executed");
        wd("pool_in.bin", x, N*C*H*W);
        wd("pool_out.bin", o->runtime_data, N*C*(H/pool)*(W/pool));
        wd("pool_grad.bin", gp, N*C*(H/pool)*(W/pool));
        lancius_clear_error();
        {
            lancius_graph* gb = lancius_graph_create();
            lancius_node* ib = lancius_input_4d(gb, N, C, H, W);
            bind(ib, x);
            lancius_node* pb = lancius_maxpool2d(gb, ib, pool, pool);
            REQUIRE(pb != NULL, "maxpool2d (bwd graph) built");
            REQUIRE(run(gb), "maxpool2d bwd graph forward executed");
            lancius_node* gb2 = lancius_input_4d(gb, N, C, H/pool, W/pool);
            bind(gb2, gp);
            lancius_node* d2 = lancius_maxpool2d_bwd(gb, gb2, ib, pool, pool);
            REQUIRE(d2 != NULL, "maxpool2d_bwd built");
            REQUIRE(run(gb), "maxpool2d bwd executed");
            wd("pool_dx.bin", d2->runtime_data, N*C*H*W);
            lancius_graph_destroy(gb);
        }
        lancius_graph_destroy(g); free(x); free(gp);
    }

    /* -------------------------------------------------- fused conv+relu */
    fprintf(stderr, "[block] fused conv+relu\n");
    {
        size_t N=1,Cin=2,Hin=6,Win=6,Cout=3,Kh=3,Kw=3,st=1,pad=1;
        size_t Hout=(Hin+2*pad-Kh)/st+1, Wout=(Win+2*pad-Kw)/st+1;
        rseed(2011);
        double* in = rndarr(N*Cin*Hin*Win,-1,1);
        double* w  = rndarr(Cout*Cin*Kh*Kw,-1,1);
        lancius_graph* gf = lancius_graph_create();
        lancius_node* in_n = lancius_input_4d(gf, N, Cin, Hin, Win);
        lancius_node* w_n  = lancius_input_4d(gf, Cout, Cin, Kh, Kw);
        bind(in_n, in); bind(w_n, w);
        lancius_node* c = lancius_conv2d(gf, in_n, w_n, st, pad);
        lancius_node* r = lancius_relu(gf, c);
        REQUIRE(run(gf), "separate conv->relu executed");
        wd("fused_sep.bin", r->runtime_data, N*Cout*Hout*Wout);
        lancius_optimize_fusion(gf);
        REQUIRE(c->op == LANCIUS_OP_NOP, "conv demoted to NOP by fusion");
        REQUIRE(r->op == LANCIUS_OP_CONV2D_RELU_FUSED, "relu became fused node");
        lancius_clear_error();
        REQUIRE(run(gf), "fused graph executed");
        wd("fused_out.bin", r->runtime_data, N*Cout*Hout*Wout);
        wm("fused.meta", (size_t[]){N,Cin,Hin,Win,Cout,Kh,Kw,st,pad,Hout,Wout}, 11);
        wd("fused_in.bin", in, N*Cin*Hin*Win);
        wd("fused_w.bin", w, Cout*Cin*Kh*Kw);
        lancius_graph_destroy(gf); free(in); free(w);
    }

    /* ------------------------------- autodiff: 2-layer MLP vs torch ---- */
    fprintf(stderr, "[block] autodiff\n");
    {
        size_t Bn=4, In=3, H1=5, Out=2;
        rseed(2012);
        double* X  = rndarr(Bn*In,-1,1);
        double* W1 = rndarr(In*H1,-1,1);
        double* B1 = rndarr(H1,-0.5,0.5);
        double* W2 = rndarr(H1*Out,-1,1);
        double* B2 = rndarr(Out,-0.5,0.5);
        double* T  = rndarr(Bn*Out,-1,1);

        lancius_graph* g = lancius_graph_create();
        lancius_node* xn = lancius_input(g, Bn, In);
        lancius_node* w1 = lancius_input(g, In, H1);
        lancius_node* b1 = lancius_input(g, 1, H1);
        lancius_node* w2 = lancius_input(g, H1, Out);
        lancius_node* b2 = lancius_input(g, 1, Out);
        lancius_node* tn = lancius_input(g, Bn, Out);
        bind(xn, X); bind(w1, W1); bind(b1, B1);
        bind(w2, W2); bind(b2, B2); bind(tn, T);

        lancius_node* a   = lancius_matmul(g, xn, w1);
        lancius_node* ab  = lancius_broadcast(g, a, Bn, H1);
        lancius_node* ab2 = lancius_add(g, ab, b1);
        lancius_node* h   = lancius_tanh(g, ab2);
        lancius_node* p1  = lancius_matmul(g, h, w2);
        lancius_node* pb  = lancius_broadcast(g, p1, Bn, Out);
        lancius_node* pb2 = lancius_add(g, pb, b2);
        lancius_node* loss = lancius_mse(g, pb2, tn);
        REQUIRE(loss != NULL, "mlp loss built");
        REQUIRE(run(g), "mlp forward executed");

        lancius_clear_error();
        lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
        REQUIRE(tg != NULL, "autodiff built");
        if (tg) {
            REQUIRE(run(tg->graph), "backward executed");
            char nm[128];
            #define DUMPGRAD(var, node, cnt) do { \
                lancius_node* gn = ((node)->id < tg->max_id) ? tg->grad_nodes[(node)->id] : NULL; \
                checks++; \
                if (gn && gn->runtime_data) { snprintf(nm,sizeof(nm),"%s.bin",var); wd(nm, gn->runtime_data, cnt); } \
                else { failures++; printf("  PROBE FAIL: missing grad %s\\n", var); } } while (0)
            DUMPGRAD("ad_dx", xn, Bn*In);
            DUMPGRAD("ad_dw1", w1, In*H1);
            DUMPGRAD("ad_db1", b1, H1);
            DUMPGRAD("ad_dw2", w2, H1*Out);
            DUMPGRAD("ad_db2", b2, Out);
            wd("ad_x.bin", X, Bn*In);
            wd("ad_w1.bin", W1, In*H1); wd("ad_b1.bin", B1, H1);
            wd("ad_w2.bin", W2, H1*Out); wd("ad_b2.bin", B2, Out);
            wd("ad_tg.bin", T, Bn*Out);
            { char p[512];
              snprintf(p, sizeof(p), "%s/autodiff.json", D);
              FILE* jf = fopen(p, "w");
              if (jf) {
                  fprintf(jf, "[{\"name\":\"mlp4x3x5x2\",");
                  fprintf(jf, "\"x\":\"ad_x.bin\",\"w1\":\"ad_w1.bin\",\"b1\":\"ad_b1.bin\",");
                  fprintf(jf, "\"w2\":\"ad_w2.bin\",\"b2\":\"ad_b2.bin\",\"tg\":\"ad_tg.bin\",");
                  fprintf(jf, "\"dx\":\"ad_dx.bin\",\"dw1\":\"ad_dw1.bin\",\"db1\":\"ad_db1.bin\",");
                  fprintf(jf, "\"dw2\":\"ad_dw2.bin\",\"db2\":\"ad_db2.bin\"}]\n");
                  fclose(jf);
              } }
            lancius_training_graph_destroy(tg);
        }
        lancius_graph_destroy(g);
        free(X); free(W1); free(B1); free(W2); free(B2); free(T);
    }

    printf("graph_oracle_dump: %d probe checks, %d failed, final err=%d\n",
           checks, failures, (int)lancius_get_error());
    return failures ? 1 : 0;
}