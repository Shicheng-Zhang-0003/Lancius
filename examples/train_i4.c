/* Lancius I4 trainer -- binary next-year fire risk on a 2 km cell-year corpus.
 *
 * Derived from examples/train_verifier.c (same persistent graph, fused CE,
 * checkpoint/resume, export and round-trip verification) but with:
 *   - explicit train/val/test splits from corpus.hdr (no random val split)
 *   - NCLASS=2 and validation average precision as the selection metric
 *   - positive-row oversampling for the rare-fire corpus
 *   - z-score normalization fit on train rows only, clipped to +/-8
 *   - val/test probability dumps for Python metrics and calibration
 *
 * The corpus is built by generation_1/I4/tools/build_corpus.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>

#include <lancius.h>
#include <lancius/lancius_autodiff.h>
#include <lancius/lancius_train.h>

#define CKPT_MAGIC 0x4C564331u /* "LVC1" */
#define PI_2 3.14159265358979323846
#define NCMAX 8

static void die(const char* m){ fprintf(stderr, "FATAL: %s\n", m); exit(2); }
static void die2(const char* m, const char* w){
    fprintf(stderr, "FATAL: %s (%s)\n", m, w); exit(2);
}
static void* xcalloc(size_t n, size_t sz, const char* what){
    void* p = calloc(n ? n : 1, sz);
    if (!p) { fprintf(stderr, "FATAL: oom %s\n", what); exit(2); }
    return p;
}

/* ---- deterministic RNG (xorshift64*), state is checkpointed ---- */
static uint64_t rng_state;
static void rseed(uint64_t s){
    rng_state = s*6364136223846793005ULL + 1442695040888963407ULL;
}
static double u01(void){
    rng_state = rng_state*6364136223846793007ULL + 1442695040888963407ULL;
    uint64_t z = rng_state;
    z ^= z>>12; z ^= z<<25; z ^= z>>27; z *= 0x2545F4914F6CDD1DULL;
    rng_state = z;
    return (double)(z>>11)/9007199254740992.0;
}
static double gauss_(void){
    double u1 = u01(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = u01();
    return sqrt(-2.0*log(u1))*cos(2.0*PI_2*u2);
}

/* ---- configuration (defaults follow the gen-1 verifier lineage) ---- */
static size_t FEAT = 77, NCLASS = 2, H1 = 512, H2 = 256, BATCH = 512;
static size_t POS_REPEAT = 20;
static double LR = 0.01, LR_MIN_FRAC = 0.01, MOM = 0.9, WD = 1e-4;
static double CLIP = 5.0, EVAL_FRAC = 0.5, WARMUP_FRAC = 0.03;
static int    EPOCHS = 30;
static unsigned SEED = 20261022u;
static size_t g_step = 0;

static const double LEVEL[NCMAX] = {0.0, 1.0, 0, 0, 0, 0, 0, 0};

/* ---- corpus ---- */
typedef struct { size_t n; double* X; int* L; int* S; int* A; } corpus;

static long fsize(const char* p){
    struct stat st; return stat(p, &st) == 0 ? (long)st.st_size : -1;
}

/* corpus.hdr: "KEY value" lines (FEAT, NCLASS, NTRAIN, NTEST). */
static void read_hdr(const char* dir, size_t *feat, size_t *nclass,
                     size_t *ntrain, size_t *nval, size_t *ntest){
    char p[512]; snprintf(p, sizeof p, "%s/corpus.hdr", dir);
    FILE* f = fopen(p, "r");
    if (!f) die2("corpus.hdr missing (run generation_1/I4/tools/build_corpus.py)", p);
    char line[128]; int seen = 0;
    while (fgets(line, sizeof line, f)) {
        char k[64]; long v;
        if (sscanf(line, "%63s %ld", k, &v) != 2) continue;
        if      (!strcmp(k, "FEAT"))   { *feat = (size_t)v;   seen |= 1; }
        else if (!strcmp(k, "NCLASS")) { *nclass = (size_t)v; seen |= 2; }
        else if (!strcmp(k, "NTRAIN")) { *ntrain = (size_t)v; seen |= 4; }
        else if (!strcmp(k, "NVAL"))   { *nval = (size_t)v;   seen |= 8; }
        else if (!strcmp(k, "NTEST"))  { *ntest = (size_t)v;  seen |= 16; }
    }
    fclose(f);
    if (seen != 31) die2("corpus.hdr incomplete (need FEAT NCLASS NTRAIN NVAL NTEST)", p);
}

static void* read_bin(const char* p, size_t expect_bytes, const char* what){
    long by = fsize(p);
    if (by < 0) die2("cannot stat", p);
    if ((size_t)by != expect_bytes) {
        fprintf(stderr, "FATAL: %s is %ld bytes, header says %zu\n",
                p, by, expect_bytes);
        exit(2);
    }
    void* buf = xcalloc(expect_bytes ? expect_bytes : 1, 1, what);
    FILE* f = fopen(p, "rb");
    if (!f || fread(buf, 1, expect_bytes, f) != expect_bytes) die2("short read", p);
    fclose(f);
    return buf;
}

static corpus load(const char* dir, const char* split, size_t feat,
                   size_t n_expect, const char* tag){
    corpus c; memset(&c, 0, sizeof c);
    char p[512];
    snprintf(p, sizeof p, "%s/%s.X.bin", dir, split);
    c.n = n_expect;
    size_t xb = c.n*feat*sizeof(double);
    c.X = (double*)read_bin(p, xb, "X");
    /* Poison-row guard: NaN/Inf features kill training silently otherwise. */
    for (size_t i = 0; i < c.n*feat; i++)
        if (!isfinite(c.X[i])) {
            fprintf(stderr, "FATAL: non-finite feature in %s row %zu (idx %zu)\n",
                    p, i/feat, i); exit(2);
        }
    snprintf(p, sizeof p, "%s/%s.L.bin", dir, split);
    c.L = (int*)read_bin(p, c.n*sizeof(int), "L");
    snprintf(p, sizeof p, "%s/%s.S.bin", dir, split);
    c.S = (int*)read_bin(p, c.n*sizeof(int), "S");
    snprintf(p, sizeof p, "%s/%s.A.bin", dir, split);
    c.A = (int*)read_bin(p, c.n*sizeof(int), "A");
    for (size_t i = 0; i < c.n; i++)
        if (c.L[i] < 0 || (size_t)c.L[i] >= NCLASS)
            { fprintf(stderr, "FATAL: label OOB in %s row %zu\n", tag, i); exit(2); }
    return c;
}

/* ---- model ---- */
typedef struct {
    double *W1,*b1,*W2,*b2,*W3,*b3;
    double *mW1,*mb1,*mW2,*mb2,*mW3,*mb3;   /* momentum (incl. biases) */
    double *gW1,*gb1,*gW2,*gb2,*gW3,*gb3;
    double *Xb,*Tb;                          /* bound batch buffers */
} model;

static size_t nparams(void){
    return FEAT*H1 + H1 + H1*H2 + H2 + H2*NCLASS + NCLASS;
}

static void minit(model* m){
    m->W1=(double*)xcalloc(FEAT*H1,8,"W1");  m->b1=(double*)xcalloc(H1,8,"b1");
    m->W2=(double*)xcalloc(H1*H2,8,"W2");    m->b2=(double*)xcalloc(H2,8,"b2");
    m->W3=(double*)xcalloc(H2*NCLASS,8,"W3");m->b3=(double*)xcalloc(NCLASS,8,"b3");
    m->mW1=(double*)xcalloc(FEAT*H1,8,"mW1");m->mb1=(double*)xcalloc(H1,8,"mb1");
    m->mW2=(double*)xcalloc(H1*H2,8,"mW2");  m->mb2=(double*)xcalloc(H2,8,"mb2");
    m->mW3=(double*)xcalloc(H2*NCLASS,8,"mW3");m->mb3=(double*)xcalloc(NCLASS,8,"mb3");
    m->gW1=(double*)xcalloc(FEAT*H1,8,"gW1");m->gb1=(double*)xcalloc(H1,8,"gb1");
    m->gW2=(double*)xcalloc(H1*H2,8,"gW2");  m->gb2=(double*)xcalloc(H2,8,"gb2");
    m->gW3=(double*)xcalloc(H2*NCLASS,8,"gW3");m->gb3=(double*)xcalloc(NCLASS,8,"gb3");
    m->Xb=(double*)xcalloc(BATCH*FEAT,8,"Xb");
    m->Tb=(double*)xcalloc(BATCH*NCLASS,8,"Tb");
    double s1=sqrt(2.0/(double)FEAT), s2=sqrt(2.0/(double)H1),
           s3=sqrt(2.0/(double)H2);
    for(size_t i=0;i<FEAT*H1;i++)     m->W1[i]=gauss_()*s1;
    for(size_t i=0;i<H1*H2;i++)       m->W2[i]=gauss_()*s2;
    for(size_t i=0;i<H2*NCLASS;i++)   m->W3[i]=gauss_()*s3;
}

static void mfree(model* m){
    free(m->W1);free(m->b1);free(m->W2);free(m->b2);free(m->W3);free(m->b3);
    free(m->mW1);free(m->mb1);free(m->mW2);free(m->mb2);free(m->mW3);free(m->mb3);
    free(m->gW1);free(m->gb1);free(m->gW2);free(m->gb2);free(m->gW3);free(m->gb3);
    free(m->Xb);free(m->Tb);
}

/* ---- persistent training graph ---- */
static lancius_arena* ARENA = NULL;
static lancius_graph* TG_FWD = NULL;     /* definition graph (export shape) */
static lancius_training_graph* TG = NULL;
static lancius_schedule* BS = NULL;
static lancius_node *NX,*NW1,*NB1,*NW2,*NB2,*NW3,*NB3,*NT,*NLOSS;

static void tgraph_build(model* m){
    TG_FWD = lancius_graph_create(); if(!TG_FWD) die("graph");
    NX  = lancius_input(TG_FWD, BATCH, FEAT);
    NW1 = lancius_input(TG_FWD, FEAT, H1);
    NB1 = lancius_input(TG_FWD, 1, H1);
    NW2 = lancius_input(TG_FWD, H1, H2);
    NB2 = lancius_input(TG_FWD, 1, H2);
    NW3 = lancius_input(TG_FWD, H2, NCLASS);
    NB3 = lancius_input(TG_FWD, 1, NCLASS);
    NT  = lancius_input(TG_FWD, BATCH, NCLASS);
    if(!NX||!NW1||!NB1||!NW2||!NB2||!NW3||!NB3||!NT) die("builders");
    lancius_node_bind_external(NX, m->Xb);
    lancius_node_bind_external(NW1, m->W1);
    lancius_node_bind_external(NB1, m->b1);
    lancius_node_bind_external(NW2, m->W2);
    lancius_node_bind_external(NB2, m->b2);
    lancius_node_bind_external(NW3, m->W3);
    lancius_node_bind_external(NB3, m->b3);
    lancius_node_bind_external(NT, m->Tb);
    lancius_node* h1=lancius_tanh(TG_FWD,lancius_add(TG_FWD,
        lancius_matmul(TG_FWD,NX,NW1),NB1));
    lancius_node* h2=lancius_tanh(TG_FWD,lancius_add(TG_FWD,
        lancius_matmul(TG_FWD,h1,NW2),NB2));
    lancius_node* z3=lancius_add(TG_FWD,lancius_matmul(TG_FWD,h2,NW3),NB3);
    /* FIX 1: framework's own fused softmax cross-entropy (mean over rows).
     * Exact VJP (softmax-y)/R -- no lr_eff fudge, no MSE-on-logits. */
    NLOSS=lancius_cross_entropy(TG_FWD,z3,NT);
    if(!h1||!h2||!z3||!NLOSS) die("loss");
    TG = lancius_ir_autodiff(TG_FWD, NLOSS); if(!TG) die("autodiff");
    BS = lancius_ir_schedule(TG->graph);     if(!BS) die("bwd sched");
}

static void tgraph_free(void){
    if (BS) lancius_schedule_destroy(BS);
    if (TG) lancius_training_graph_destroy(TG);
    if (TG_FWD) lancius_graph_destroy(TG_FWD);
    BS = NULL; TG = NULL; TG_FWD = NULL;
}

/* grad tensor registry for global-norm clip + guards */
static double* GPTR[6]; static size_t GN[6];
static void grad_registry(model* m){
    GPTR[0]=m->gW1; GN[0]=FEAT*H1;   GPTR[1]=m->gb1; GN[1]=H1;
    GPTR[2]=m->gW2; GN[2]=H1*H2;     GPTR[3]=m->gb2; GN[3]=H2;
    GPTR[4]=m->gW3; GN[4]=H2*NCLASS; GPTR[5]=m->gb3; GN[5]=NCLASS;
}

static double step_model(model* m){
    /* Last batch is PADDED by replaying earlier rows (static graph needs
     * fixed BATCH; padded rows contribute but shuffled order makes that
     * harmless). Buffers were filled by the caller; execute fused fwd+bwd. */
    lancius_arena_reset(ARENA);
    lancius_schedule_execute(BS, ARENA);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        fprintf(stderr, "train step err=%s\n",
                lancius_error_string(lancius_get_error())); exit(2);
    }
    lancius_clear_error();
    /* FIX 4: read the loss every step (old drivers printed "-"). */
    double loss = TG->loss_node->runtime_data[0];
    if (!isfinite(loss)) {
        fprintf(stderr, "FATAL: non-finite loss at step %zu\n", g_step);
        exit(2);
    }
    #define HG(dst,node,cnt) do{ lancius_node* g_=TG->grad_nodes[(node)->id]; \
        if(!g_||!g_->runtime_data){ fprintf(stderr,"FATAL: missing grad\n"); exit(2);} \
        memcpy((dst),g_->runtime_data,(cnt)*sizeof(double)); }while(0)
    HG(m->gW1,NW1,FEAT*H1); HG(m->gb1,NB1,H1);
    HG(m->gW2,NW2,H1*H2);   HG(m->gb2,NB2,H2);
    HG(m->gW3,NW3,H2*NCLASS); HG(m->gb3,NB3,NCLASS);
    #undef HG
    /* FIX 4b: fail loud on non-finite grads, then clip global norm. */
    for (int t = 0; t < 6; t++)
        for (size_t i = 0; i < GN[t]; i++)
            if (!isfinite(GPTR[t][i])) {
                fprintf(stderr, "FATAL: non-finite grad tensor %d at step %zu\n",
                        t, g_step); exit(2);
            }
    lancius_clip_global_norm(GPTR, GN, 6, CLIP);
    return loss;
}

/* SGD with momentum + weight decay (decoupled position, pre-momentum). */
static void apply_step(model* m, double lr){
    #define UPD(W,M,G,N,WDV) do{ for(size_t i=0;i<(size_t)(N);i++){ \
        double gi=(G)[i]+(WDV)*(W)[i]; (M)[i]=MOM*(M)[i]+gi; (W)[i]-=lr*(M)[i]; } }while(0)
    UPD(m->W1,m->mW1,m->gW1,FEAT*H1,WD);
    UPD(m->b1,m->mb1,m->gb1,H1,0.0);
    UPD(m->W2,m->mW2,m->gW2,H1*H2,WD);
    UPD(m->b2,m->mb2,m->gb2,H2,0.0);
    UPD(m->W3,m->mW3,m->gW3,H2*NCLASS,WD);
    UPD(m->b3,m->mb3,m->gb3,NCLASS,0.0);
    #undef UPD
    /* FIX 4c: full weight sanity after the step (cheap vs the matmul). */
    for (size_t i = 0; i < FEAT*H1; i++)
        if (!isfinite(m->W1[i])) { fprintf(stderr,"FATAL: W1[%zu] diverged at step %zu\n",i,g_step); exit(2); }
    for (size_t i = 0; i < H1*H2; i++)
        if (!isfinite(m->W2[i])) { fprintf(stderr,"FATAL: W2[%zu] diverged at step %zu\n",i,g_step); exit(2); }
    for (size_t i = 0; i < H2*NCLASS; i++)
        if (!isfinite(m->W3[i])) { fprintf(stderr,"FATAL: W3[%zu] diverged at step %zu\n",i,g_step); exit(2); }
}

/* ---- inference forward (OpenMP, deterministic per-row order) ---- */
static void fwd_logits(const model* m,const double* X,size_t n,double* z){
    #pragma omp parallel
    {
        double* hid1=(double*)malloc(H1*sizeof(double));
        double* hid2=(double*)malloc(H2*sizeof(double));
        if(!hid1||!hid2) die("fwd tmp");
        #pragma omp for schedule(static)
        for(size_t i=0;i<n;i++){
            for(size_t j=0;j<H1;j++){
                double s=m->b1[j];
                for(size_t k=0;k<FEAT;k++) s+=X[i*FEAT+k]*m->W1[k*H1+j];
                hid1[j]=tanh(s);
            }
            for(size_t j=0;j<H2;j++){
                double s=m->b2[j];
                for(size_t k=0;k<H1;k++) s+=hid1[k]*m->W2[k*H2+j];
                hid2[j]=tanh(s);
            }
            for(size_t c=0;c<NCLASS;c++){
                double s=m->b3[c];
                for(size_t j=0;j<H2;j++) s+=hid2[j]*m->W3[j*NCLASS+c];
                z[i*NCLASS+c]=s;
            }
        }
        free(hid1); free(hid2);
    }
}

/* ---- metrics ---- */
typedef struct {
    size_t n, exact, within1;
    double mae, macro_f1, kappa;
    long cm[NCMAX][NCMAX];
} met;

static void eval_split(met* mo, met* ms, const model* md,
                       const double* X, const int* L, const int* S, size_t n){
    double* z=(double*)malloc(n*NCLASS*sizeof(double));
    if(!z) die("eval z");
    fwd_logits(md,X,n,z);
    memset(mo,0,sizeof *mo); memset(ms,0,sizeof *ms);
    for(size_t i=0;i<n;i++){
        met* m = S[i]>0 ? ms : mo;
        int p=0;
        for(size_t c=1;c<NCLASS;c++) if(z[i*NCLASS+c]>z[i*NCLASS+p]) p=(int)c;
        int t=L[i];
        m->cm[t][p]++;
        if(p==t) m->exact++;
        if(p==t || (p>t? p-t : t-p)<=1) m->within1++;
        m->mae += fabs(LEVEL[p]-LEVEL[t]);
        m->n++;
    }
    for (int which = 0; which < 2; which++){
        met* m = which ? ms : mo;
        if (!m->n) continue;
        m->mae /= (double)m->n;
        double f1=0;
        for(size_t c=0;c<NCLASS;c++){
            double tp=m->cm[c][c], fp=0, fn=0;
            for(size_t r=0;r<NCLASS;r++){ if((int)r!=(int)c){ fp+=m->cm[r][c]; fn+=m->cm[c][r]; } }
            double pr=(tp+fp>0)?tp/(tp+fp):0.0, rc=(tp+fn>0)?tp/(tp+fn):0.0;
            f1 += (pr+rc>0)? 2*pr*rc/(pr+rc):0.0;
        }
        m->macro_f1=f1/(double)NCLASS;
        double po=(double)m->exact/(double)m->n, pe=0;
        for(size_t c=0;c<NCLASS;c++){
            double rt=0, ct=0;
            for(size_t r=0;r<NCLASS;r++){ rt+=m->cm[r][c]; ct+=m->cm[r][c]; }
            pe += (rt>0)? (rt/(double)m->n)*(ct/(double)m->n) : 0.0;
        }
        m->kappa = (pe<1.0)? (po-pe)/(1.0-pe) : 0.0;
    }
    free(z);
}

static void mprint(const char* tag,const met* m){
    printf("%-10s n=%-6zu exact=%.4f within1=%.4f macroF1=%.4f MAE=%.4f kappa=%.4f\n",
           tag,m->n, m->n?(double)m->exact/(double)m->n:0.0,
           m->n?(double)m->within1/(double)m->n:0.0, m->macro_f1, m->mae, m->kappa);
}

/* ---- checkpoint / resume (FIX 7) ---- */
static int save_state(const char* path, const model* m, const double* mu,
                      const double* sd, long step, double best_score){
    FILE* f=fopen(path,"wb"); if(!f) return -1;
    uint32_t magic=CKPT_MAGIC;
    size_t dims[5]={FEAT,NCLASS,H1,H2,BATCH};
    fwrite(&magic,sizeof magic,1,f);
    fwrite(dims,sizeof dims,1,f);
    fwrite(&step,sizeof step,1,f);
    fwrite(&best_score,sizeof best_score,1,f);
    fwrite(&rng_state,sizeof rng_state,1,f);
    fwrite(mu,sizeof(double),FEAT,f);
    fwrite(sd,sizeof(double),FEAT,f);
    fwrite(m->W1,8,FEAT*H1,f);   fwrite(m->b1,8,H1,f);
    fwrite(m->W2,8,H1*H2,f);     fwrite(m->b2,8,H2,f);
    fwrite(m->W3,8,H2*NCLASS,f); fwrite(m->b3,8,NCLASS,f);
    fwrite(m->mW1,8,FEAT*H1,f);  fwrite(m->mb1,8,H1,f);
    fwrite(m->mW2,8,H1*H2,f);    fwrite(m->mb2,8,H2,f);
    fwrite(m->mW3,8,H2*NCLASS,f);fwrite(m->mb3,8,NCLASS,f);
    int ok = ferror(f)==0;
    fclose(f);
    return ok ? 0 : -1;
}

static int load_state(const char* path, model* m, double* mu, double* sd,
                      long* step, double* best_score){
    FILE* f=fopen(path,"rb"); if(!f) return -1;
    uint32_t magic=0; size_t dims[5];
    if (fread(&magic,sizeof magic,1,f)!=1 || magic!=CKPT_MAGIC) { fclose(f); return -1; }
    if (fread(dims,sizeof dims,1,f)!=1) { fclose(f); return -1; }
    if (dims[0]!=FEAT||dims[1]!=NCLASS||dims[2]!=H1||dims[3]!=H2||dims[4]!=BATCH)
        { fprintf(stderr,"FATAL: checkpoint dims mismatch\n"); exit(2); }
    long st; double bs; uint64_t rs;
    if (fread(&st,sizeof st,1,f)!=1 || fread(&bs,sizeof bs,1,f)!=1 ||
        fread(&rs,sizeof rs,1,f)!=1) { fclose(f); return -1; }
    if (fread(mu,8,FEAT,f)!=FEAT) { fclose(f); return -1; }
    if (fread(sd,8,FEAT,f)!=FEAT) { fclose(f); return -1; }
    if (fread(m->W1,8,FEAT*H1,f)!=FEAT*H1) { fclose(f); return -1; }
    if (fread(m->b1,8,H1,f)!=H1) { fclose(f); return -1; }
    if (fread(m->W2,8,H1*H2,f)!=H1*H2) { fclose(f); return -1; }
    if (fread(m->b2,8,H2,f)!=H2) { fclose(f); return -1; }
    if (fread(m->W3,8,H2*NCLASS,f)!=H2*NCLASS) { fclose(f); return -1; }
    if (fread(m->b3,8,NCLASS,f)!=NCLASS) { fclose(f); return -1; }
    if (fread(m->mW1,8,FEAT*H1,f)!=FEAT*H1) { fclose(f); return -1; }
    if (fread(m->mb1,8,H1,f)!=H1) { fclose(f); return -1; }
    if (fread(m->mW2,8,H1*H2,f)!=H1*H2) { fclose(f); return -1; }
    if (fread(m->mb2,8,H2,f)!=H2) { fclose(f); return -1; }
    if (fread(m->mW3,8,H2*NCLASS,f)!=H2*NCLASS) { fclose(f); return -1; }
    if (fread(m->mb3,8,NCLASS,f)!=NCLASS) { fclose(f); return -1; }
    fclose(f);
    g_step=(size_t)st; *step=st; *best_score=bs; rng_state=rs;
    return 0;
}

/* ---- .lancius export (FIX 8: default rows=1) ---- */
static int export_lancius(const model* m, const char* path, size_t rows) {
    if (rows < 1) rows = 1;
    lancius_graph* g = lancius_graph_create();
    if (!g) return -1;
    lancius_node* Xn  = lancius_input(g, rows, FEAT);
    lancius_node* W1n = lancius_input(g, FEAT, H1);
    lancius_node* b1n = lancius_input(g, 1, H1);
    lancius_node* W2n = lancius_input(g, H1, H2);
    lancius_node* b2n = lancius_input(g, 1, H2);
    lancius_node* W3n = lancius_input(g, H2, NCLASS);
    lancius_node* b3n = lancius_input(g, 1, NCLASS);
    if (!Xn||!W1n||!b1n||!W2n||!b2n||!W3n||!b3n) {
        fprintf(stderr, "export: node builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g); return -1;
    }
    lancius_node_bind_external(W1n, m->W1);
    lancius_node_bind_external(b1n, m->b1);
    lancius_node_bind_external(W2n, m->W2);
    lancius_node_bind_external(b2n, m->b2);
    lancius_node_bind_external(W3n, m->W3);
    lancius_node_bind_external(b3n, m->b3);
    lancius_node* h1 = lancius_tanh(g, lancius_add(g, lancius_matmul(g, Xn, W1n), b1n));
    lancius_node* h2 = lancius_tanh(g, lancius_add(g, lancius_matmul(g, h1, W2n), b2n));
    lancius_node* z3 = lancius_add(g, lancius_matmul(g, h2, W3n), b3n);
    if (!h1 || !h2 || !z3) {
        fprintf(stderr, "export: op builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g); return -1;
    }
    int rc = lancius_graph_save(g, path);
    if (rc != 0) fprintf(stderr, "lancius_graph_save failed: %s\n",
                         lancius_error_string(lancius_get_error()));
    lancius_graph_destroy(g);
    return rc;
}

static int verify_lancius(const char* path, const model* m,
                          const double* X, size_t n) {
    lancius_graph* g = lancius_graph_load(path);
    if (!g) { fprintf(stderr, "load %s failed: %s\n", path,
                      lancius_error_string(lancius_get_error())); return -1; }
    lancius_node* Xn = NULL;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* nd = g->nodes[i];
        if (nd && nd->op == LANCIUS_OP_INPUT && nd->ndim == 2 &&
            nd->shape[0] == n && nd->shape[1] == FEAT) {
            Xn = nd; break;
        }
    }
    if (!Xn) { fprintf(stderr, "no [%zu,%zu] INPUT in %s\n", n, FEAT, path);
               lancius_graph_destroy(g); return -1; }
    lancius_arena* ar = lancius_arena_create(256u*1024u*1024u);
    if (!ar) { lancius_graph_destroy(g); return -1; }
    lancius_node_bind_external(Xn, (double*)X);
    lancius_schedule* sch = lancius_ir_schedule(g);
    if (!sch) { lancius_arena_destroy(ar); lancius_graph_destroy(g); return -1; }
    lancius_schedule_execute(sch, ar);
    int err = (int)lancius_get_error();
    if (err != 0) {
        fprintf(stderr, "reloaded execute err=%s\n",
                lancius_error_string(lancius_get_error()));
        lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
        lancius_graph_destroy(g); return -1;
    }
    lancius_node* out = NULL;
    for (uint32_t i = 0; i < g->node_count && !out; i++) {
        lancius_node* nd = g->nodes[i];
        if (!nd || !nd->runtime_data || nd->ndim != 2) continue;
        if (nd->op == LANCIUS_OP_INPUT || nd->op == LANCIUS_OP_CONST) continue;
        if (nd->shape[0] != n || nd->shape[1] != NCLASS) continue;
        int consumed = 0;
        for (uint32_t j = 0; j < g->node_count && !consumed; j++) {
            lancius_node* mm = g->nodes[j];
            if (!mm || !mm->inputs) continue;
            for (uint32_t c = 0; c < mm->input_count; c++)
                if (mm->inputs[c] == nd) { consumed = 1; break; }
        }
        if (!consumed) out = nd;
    }
    if (!out) { fprintf(stderr, "no unconsumed [n,%zu] node\n", NCLASS);
                lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
                lancius_graph_destroy(g); return -1; }
    double worst = 0.0;
    double* ref = (double*)malloc(n*NCLASS*sizeof(double));
    if (!ref) { fprintf(stderr, "oom verify\n"); return -1; }
    fwd_logits(m, X, n, ref);
    for (size_t i = 0; i < n*NCLASS; i++) {
        double d = fabs(ref[i] - out->runtime_data[i]);
        if (d > worst) worst = d;
    }
    free(ref);
    lancius_schedule_destroy(sch); lancius_arena_destroy(ar);
    lancius_graph_destroy(g);
    printf("\n.lancius round-trip: max |diff| = %.3e  %s\n",
           worst, worst < 1e-12 ? "OK" : "MISMATCH");
    return worst < 1e-12 ? 0 : -1;
}

/* ---- binary metrics, probability dumps ---- */
static double sigmoid1(double z0, double z1){
    double d = z0 - z1;
    if (d > 50.0) return 0.0;
    if (d < -50.0) return 1.0;
    return 1.0/(1.0 + exp(d));
}

typedef struct { double s; int y; } scored;
static int cmp_score_desc(const void* a, const void* b){
    const scored* x=(const scored*)a; const scored* y=(const scored*)b;
    if (x->s < y->s) return 1;
    if (x->s > y->s) return -1;
    return 0;
}

static void binary_metrics(const model* md, const double* X, const int* L, size_t n,
                           double* out_ap, double* out_ll, double* out_brier){
    *out_ap=0.0; *out_ll=0.0; *out_brier=0.0;
    if (n==0) return;
    double* z=(double*)xcalloc(n*NCLASS,8,"metric z");
    fwd_logits(md,X,n,z);
    scored* sc=(scored*)xcalloc(n,sizeof(scored),"metric sort");
    long P=0;
    double ll=0.0, br=0.0;
    for (size_t i=0;i<n;i++){
        double p=sigmoid1(z[i*NCLASS+0],z[i*NCLASS+1]);
        if (L[i]==1) P++;
        sc[i].s=p; sc[i].y=L[i];
        double pc = p<1e-12?1e-12:(p>1.0-1e-12?1.0-1e-12:p);
        ll += (L[i]==1) ? -log(pc) : -log(1.0-pc);
        double d=p-(double)L[i]; br += d*d;
    }
    *out_ll=ll/(double)n; *out_brier=br/(double)n;
    qsort(sc,n,sizeof(scored),cmp_score_desc);
    double hit=0.0;
    for (size_t k=0;k<n;k++){
        if (sc[k].y==1){
            hit += 1.0;
            *out_ap += hit/(double)(k+1);
        }
    }
    *out_ap = P>0 ? *out_ap/(double)P : 0.0;
    free(sc); free(z);
}

static int save_probs(const char* path, const model* md, const double* X, size_t n){
    double* z=(double*)xcalloc(n*NCLASS,8,"prob z");
    fwd_logits(md,X,n,z);
    FILE* f=fopen(path,"wb");
    if (!f){ free(z); return -1; }
    for (size_t i=0;i<n;i++){
        double p=sigmoid1(z[i*NCLASS+0],z[i*NCLASS+1]);
        double row[2]={1.0-p,p};
        if (fwrite(row,sizeof(double),2,f)!=2){ fclose(f); free(z); return -1; }
    }
    fclose(f); free(z); return 0;
}

static void print_cm(const char* tag,const met* m){
    printf("\nconfusion (%s) rows=true cols=pred\n        ",tag);
    for(size_t c=0;c<NCLASS;c++) printf("%9.0f",LEVEL[c]);
    printf("\n");
    for(size_t r=0;r<NCLASS;r++){
        printf("%+5.1f  ",LEVEL[r]);
        for(size_t c=0;c<NCLASS;c++) printf("%9ld",m->cm[r][c]);
        printf("\n");
    }
}

/* ---- self-contained gradient check for the CE objective ---- */
static void gradcheck(void){
    size_t n = BATCH > 8 ? 8 : BATCH;
    size_t save_batch = BATCH;
    BATCH = n;
    model m; memset(&m,0,sizeof m);
    m.W1=(double*)xcalloc(FEAT*H1,8,"gW1"); m.b1=(double*)xcalloc(H1,8,"gb1");
    m.W2=(double*)xcalloc(H1*H2,8,"gW2");   m.b2=(double*)xcalloc(H2,8,"gb2");
    m.W3=(double*)xcalloc(H2*NCLASS,8,"gW3");m.b3=(double*)xcalloc(NCLASS,8,"gb3");
    m.Xb=(double*)xcalloc(BATCH*FEAT,8,"gXb");
    m.Tb=(double*)xcalloc(BATCH*NCLASS,8,"gTb");
    double* X = m.Xb;                 /* bindings read these in place */
    int*    L =(int*)xcalloc(n,8,"gL");
    rseed(SEED^0x5EEDu);
    for(size_t i=0;i<FEAT*H1;i++)   m.W1[i]=gauss_()*sqrt(2.0/(double)FEAT);
    for(size_t i=0;i<H1*H2;i++)     m.W2[i]=gauss_()*sqrt(2.0/(double)H1);
    for(size_t i=0;i<H2*NCLASS;i++) m.W3[i]=gauss_()*sqrt(2.0/(double)H2);
    for(size_t i=0;i<n*FEAT;i++) X[i]=gauss_();
    for(size_t i=0;i<n;i++){
        L[i]=(int)((u01()*(double)NCLASS));
        if (L[i]>= (int)NCLASS) L[i]=(int)NCLASS-1;
        m.Tb[i*NCLASS+L[i]]=1.0;
    }

    tgraph_build(&m);   /* binds m's buffers by pointer */
    ARENA = lancius_arena_create(256u*1024u*1024u); if(!ARENA) die("gc arena");
    lancius_arena_reset(ARENA);
    lancius_schedule_execute(BS, ARENA);
    if (lancius_get_error()!=LANCIUS_ERROR_OK) die("gc execute");
    lancius_clear_error();
    double* GW=(double*)xcalloc(H1*H2,8,"gcGW");
    lancius_node* gn = TG->grad_nodes[NW2->id];
    if(!gn||!gn->runtime_data) die("gc grad");
    memcpy(GW, gn->runtime_data, H1*H2*8);
    double loss_gc = TG->loss_node->runtime_data[0];
    tgraph_free();
    lancius_arena_destroy(ARENA); ARENA=NULL;

    /* central differences of mean-CE in plain fp64 */
    double* h1=(double*)xcalloc(H1,8,"gc h1");
    double* h2=(double*)xcalloc(H2,8,"gc h2");
    double ce=0;
    #define FORWARD_CE(M) do{ \
        ce=0; \
        for(size_t i=0;i<n;i++){ \
            for(size_t j=0;j<H1;j++){ double s=(M).b1[j]; \
                for(size_t k=0;k<FEAT;k++) s+=X[i*FEAT+k]*(M).W1[k*H1+j]; \
                h1[j]=tanh(s);} \
            for(size_t j=0;j<H2;j++){ double s=(M).b2[j]; \
                for(size_t k=0;k<H1;k++) s+=h1[k]*(M).W2[k*H2+j]; \
                h2[j]=tanh(s);} \
            double mx=-1e30; double zz[NCMAX]; \
            for(size_t c=0;c<NCLASS;c++){ double s=(M).b3[c]; \
                for(size_t j=0;j<H2;j++) s+=h2[j]*(M).W3[j*NCLASS+c]; \
                zz[c]=s; if(s>mx) mx=s; } \
            double lse=0; for(size_t c=0;c<NCLASS;c++) lse+=exp(zz[c]-mx); \
            lse=log(lse)+mx; \
            ce += -(zz[L[i]]-lse); \
        } \
        ce /= (double)n; \
    }while(0)
    FORWARD_CE(m);
    double rms=0;
    for(size_t i=0;i<H1*H2;i++) rms+=GW[i]*GW[i];
    rms=sqrt(rms/(double)(H1*H2)); if(!(rms>0)) rms=1.0;
    double worst=0; int probes=0; double hh=1e-6;
    for(int t=0;t<40 && probes<20;t++){
        size_t idx=(size_t)((t*7919u)%(H1*H2));
        double o=m.W2[idx];
        m.W2[idx]=o+hh; FORWARD_CE(m); double lp=ce;
        m.W2[idx]=o-hh; FORWARD_CE(m); double lm=ce;
        m.W2[idx]=o;
        double fd=(lp-lm)/(2.0*hh);
        double den=fmax(fmax(fabs(fd),fabs(GW[idx])),rms);
        double e=fabs(fd-GW[idx])/den;
        if(e>worst) worst=e;
        probes++;
    }
    printf("\ngradient check (CE): autodiff vs central differences\n"
           "  %d probes on W2, gc-loss=%.6f, worst scale-relative err = %.3e  %s\n",
           probes, loss_gc, worst, worst<1e-5?"OK":"MISMATCH");
    free(h1); free(h2); free(GW); free(L);
    mfree(&m);
    BATCH = save_batch;
}

int main(int argc, char** argv){
    const char* dir = NULL;
    const char* export_path = NULL;
    const char* resume_path = NULL;
    const char* save_dir = ".";
    int do_verify_rt = 0, feat_cli = 0, nclass_cli = 0;
    int pos = 0;
    size_t lancius_rows = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i+1 < argc) SEED=(unsigned)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) LR=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--lr-min") && i+1 < argc) LR_MIN_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--wd") && i+1 < argc) WD=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--mom") && i+1 < argc) MOM=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--clip") && i+1 < argc) CLIP=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--warmup-frac") && i+1 < argc) WARMUP_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--eval-frac") && i+1 < argc) EVAL_FRAC=strtod(argv[++i],NULL);
        else if (!strcmp(argv[i], "--pos-repeat") && i+1 < argc) POS_REPEAT=(size_t)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--export-lancius") && i+1 < argc) export_path=argv[++i];
        else if (!strcmp(argv[i], "--lancius-rows") && i+1 < argc) lancius_rows=(size_t)strtoul(argv[++i],NULL,10);
        else if (!strcmp(argv[i], "--verify-lancius")) do_verify_rt=1;
        else if (!strcmp(argv[i], "--resume") && i+1 < argc) resume_path=argv[++i];
        else if (!strcmp(argv[i], "--save-dir") && i+1 < argc) save_dir=argv[++i];
        else if (!strcmp(argv[i], "--feat") && i+1 < argc) { FEAT=(size_t)strtoul(argv[++i],NULL,10); feat_cli=1; }
        else if (!strcmp(argv[i], "--nclass") && i+1 < argc) { NCLASS=(size_t)strtoul(argv[++i],NULL,10); nclass_cli=1; }
        else if (!dir) dir = argv[i];
        else switch (pos++) {
            case 0: H1=strtoul(argv[i],NULL,10); break;
            case 1: H2=strtoul(argv[i],NULL,10); break;
            case 2: EPOCHS=atoi(argv[i]); break;
            case 3: BATCH=strtoul(argv[i],NULL,10); break;
            default: fprintf(stderr,"too many positional args\n"); return 2;
        }
    }
    if (!dir) {
        fprintf(stderr,
        "usage: train_i4 <corpus_dir> [H1 H2 epochs batch]\n"
        "  [--lr f] [--lr-min frac] [--wd f] [--mom f] [--clip f]\n"
        "  [--warmup-frac f] [--eval-frac f] [--pos-repeat n] [--seed s]\n"
        "  [--export-lancius p] [--lancius-rows r] [--verify-lancius]\n"
        "  [--resume last.bin] [--save-dir d]\n");
        return 2;
    }
    if (POS_REPEAT < 1) POS_REPEAT = 1;
    mkdir(save_dir, 0755);

    size_t ntrain=0, nval=0, ntest=0;
    size_t hdr_feat=0, hdr_nclass=0;
    read_hdr(dir, &hdr_feat, &hdr_nclass, &ntrain, &nval, &ntest);
    if (!feat_cli)  FEAT = hdr_feat;
    if (!nclass_cli) NCLASS = hdr_nclass;
    if (NCLASS != 2) die("train_i4 expects binary NCLASS=2");
    if (FEAT != 77) fprintf(stderr, "WARN: FEAT=%zu (target spec expects 77)\n", FEAT);

    corpus tr = load(dir, "train", FEAT, ntrain, "train");
    corpus va = load(dir, "val",   FEAT, nval,   "val");
    corpus te = load(dir, "test",  FEAT, ntest,  "test");
    if (va.n == 0 || te.n == 0) die("empty val or test split");

    printf("================================================================\n");
    printf("  Lancius I4 binary fire-risk trainer  %zu -> %zu -> %zu -> %zu\n"
           "  (%.3fM params)  CE objective, persistent graph\n",
           FEAT, H1, H2, NCLASS, (double)nparams()/1e6);
    printf("================================================================\n");

    /* z-score from train rows only; clip to +/-8 */
    double* mu=(double*)xcalloc(FEAT,8,"mu");
    double* sd=(double*)xcalloc(FEAT,8,"sd");
    for(size_t i=0;i<tr.n;i++)
        for(size_t k=0;k<FEAT;k++) mu[k]+=tr.X[i*FEAT+k];
    for(size_t k=0;k<FEAT;k++) mu[k]/=(double)tr.n;
    for(size_t i=0;i<tr.n;i++)
        for(size_t k=0;k<FEAT;k++){ double d=tr.X[i*FEAT+k]-mu[k]; sd[k]+=d*d; }
    for(size_t k=0;k<FEAT;k++){ sd[k]=sqrt(sd[k]/(double)tr.n);
        if(!(sd[k]>1e-12)) sd[k]=1.0; }
    #define NORM(C) do{ for(size_t i=0;i<(C).n;i++) for(size_t k=0;k<FEAT;k++){ \
        double v=((C).X[i*FEAT+k]-mu[k])/sd[k]; \
        if(v>8.0) v=8.0; \
        if(v<-8.0) v=-8.0; \
        (C).X[i*FEAT+k]=v; } }while(0)
    NORM(tr); NORM(va); NORM(te);
    #undef NORM

    size_t npos=0;
    for(size_t i=0;i<tr.n;i++) if(tr.L[i]==1) npos++;
    if(!npos) die("train split has no positive rows");
    size_t nfit = tr.n + npos*(POS_REPEAT-1);
    size_t* idx=(size_t*)xcalloc(nfit,sizeof(size_t),"idx");
    size_t q=0;
    for(size_t i=0;i<tr.n;i++) idx[q++]=i;
    for(size_t r=1;r<POS_REPEAT;r++)
        for(size_t i=0;i<tr.n;i++) if(tr.L[i]==1) idx[q++]=i;
    if(q!=nfit) die("oversampling index size mismatch");

    gradcheck();
    rseed(SEED);
    model m; minit(&m);
    rseed(SEED^0xBEEFu);
    for(size_t i=nfit-1;i>0;i--){
        size_t j=(size_t)(u01()*(double)(i+1));
        size_t t=idx[i]; idx[i]=idx[j]; idx[j]=t;
    }

    char pbest[1024], plast[1024];
    snprintf(pbest,sizeof pbest,"%s/best.bin",save_dir);
    snprintf(plast,sizeof plast,"%s/last.bin",save_dir);
    long step=0;
    double best_ap=-1.0, best_ll=1e30;
    if (resume_path) {
        double bs;
        if (load_state(resume_path,&m,mu,sd,&step,&bs)!=0) die2("resume failed",resume_path);
        best_ap=bs;
        printf("resumed from %s (step %ld, best AP %.4f)\n",resume_path,step,bs);
    }

    grad_registry(&m);
    ARENA=lancius_arena_create(1024u*1024u*1024u); if(!ARENA) die("arena");
    tgraph_build(&m);

    size_t steps_per_epoch=(nfit+BATCH-1)/BATCH;
    size_t eval_every=(size_t)(EVAL_FRAC*(double)steps_per_epoch);
    if(eval_every==0) eval_every=steps_per_epoch;
    size_t total_steps=steps_per_epoch*(size_t)EPOCHS + (size_t)step;
    size_t warmup=(size_t)(WARMUP_FRAC*(double)total_steps);
    double lr_min=LR*LR_MIN_FRAC;

    printf("\ntrain=%zu (pos=%zu, repeat=%zu, fit=%zu) val=%zu test=%zu\n",
           tr.n,npos,POS_REPEAT,nfit,va.n,te.n);
    printf("steps/epoch=%zu total=%zu warmup=%zu eval_every=%zu\n",
           steps_per_epoch,total_steps,warmup,eval_every);
    printf("lr=%.3g lr_min=%.3g mom=%.2f wd=%.1g clip=%.1g\n",
           LR,lr_min,MOM,WD,CLIP);
    printf("\n   step   lr        CE(ema)   val-AP    val-logloss  val-Brier\n");
    printf("---------------------------------------------------------------\n");

    double ema_ce=0; int have_ema=0;
    for(int ep=0; ep<EPOCHS; ep++){
        for(size_t i=nfit-1;i>0;i--){
            size_t j=(size_t)(u01()*(double)(i+1));
            size_t t=idx[i]; idx[i]=idx[j]; idx[j]=t;
        }
        for(size_t s=0; s<nfit; s+=BATCH){
            size_t rows=nfit-s; if(rows>BATCH) rows=BATCH;
            for(size_t r=0;r<BATCH;r++){
                size_t src=idx[s+(r%rows)];
                memcpy(m.Xb+r*FEAT,tr.X+src*FEAT,FEAT*8);
                for(size_t c=0;c<NCLASS;c++) m.Tb[r*NCLASS+c]=0.0;
                m.Tb[r*NCLASS+tr.L[src]]=1.0;
            }
            double lr = lancius_lr_warmup_cosine((int)g_step,(int)warmup,
                                                 (int)total_steps,LR,lr_min);
            double ce = step_model(&m);
            ema_ce = have_ema ? 0.98*ema_ce+0.02*ce : ce;
            have_ema = 1;
            apply_step(&m,lr);
            g_step++; step=(long)g_step;
            if (g_step % eval_every == 0 || g_step == total_steps){
                double ap,ll,br;
                binary_metrics(&m,va.X,va.L,va.n,&ap,&ll,&br);
                printf("%7zu  %.3g  %8.4f  %8.4f  %10.4f  %8.4f%s\n",
                       g_step,lr,ema_ce,ap,ll,br,
                       (ap>best_ap+1e-9 || (fabs(ap-best_ap)<=1e-9 && ll<best_ll))
                       ? "  <- best" : "");
                fflush(stdout);
                if (ap>best_ap+1e-9 || (fabs(ap-best_ap)<=1e-9 && ll<best_ll)){
                    best_ap=ap; best_ll=ll;
                    if (save_state(pbest,&m,mu,sd,step,best_ap)!=0)
                        fprintf(stderr,"WARN: cannot write %s\n",pbest);
                }
            }
        }
        if (save_state(plast,&m,mu,sd,step,best_ap)!=0)
            fprintf(stderr,"WARN: cannot write %s\n",plast);
    }

    {
        double bs; long st;
        if (load_state(pbest,&m,mu,sd,&st,&bs)!=0) die("best snapshot unreadable");
        printf("\nselected best validation AP %.4f (step %ld)\n",bs,st);
    }

    double vap,vll,vbr,tap,tll,tbr;
    binary_metrics(&m,va.X,va.L,va.n,&vap,&vll,&vbr);
    binary_metrics(&m,te.X,te.L,te.n,&tap,&tll,&tbr);
    printf("\nvalidation AP=%.4f logloss=%.4f Brier=%.4f\n",vap,vll,vbr);
    printf("\n================ HELD-OUT TEST ================\n");
    size_t test_pos=0;
    for(size_t i=0;i<te.n;i++) if(te.L[i]==1) test_pos++;
    printf("test AP=%.4f logloss=%.4f Brier=%.4f prevalence=%.4f\n",
           tap,tll,tbr,(double)test_pos/(double)te.n);

    met mo, ms;
    eval_split(&mo,&ms,&m,te.X,te.L,te.S,te.n);
    mprint("test",&mo);
    print_cm("test",&mo);

    char pp[1100];
    snprintf(pp,sizeof pp,"%s/test.prob.bin",save_dir);
    if (save_probs(pp,&m,te.X,te.n)==0) printf("\nwrote %s\n",pp);
    else fprintf(stderr,"WARN: cannot write %s\n",pp);
    snprintf(pp,sizeof pp,"%s/val.prob.bin",save_dir);
    if (save_probs(pp,&m,va.X,va.n)==0) printf("wrote %s\n",pp);
    else fprintf(stderr,"WARN: cannot write %s\n",pp);

    if (export_path) {
        if (lancius_rows > va.n) lancius_rows = 1;
        if (export_lancius(&m, export_path, lancius_rows) == 0) {
            struct stat st_;
            size_t fsz = stat(export_path,&st_)==0 ? (size_t)st_.st_size : 0;
            printf("\nwrote %s (%zu bytes, batch=%zu rows, %.3fM params)\n",
                   export_path,fsz,lancius_rows,(double)nparams()/1e6);
        } else { fprintf(stderr,"export FAILED\n"); return 2; }
        if (do_verify_rt) {
            if (verify_lancius(export_path,&m,va.X,lancius_rows)!=0) {
                fprintf(stderr,"round-trip FAILED\n"); return 2;
            }
        }
    }

    {
        char p[600]; snprintf(p,sizeof p,"%s/model_i4.txt",save_dir);
        FILE* f=fopen(p,"w");
        if(!f) fprintf(stderr,"WARN: cannot write %s\n",p);
        else{
            fprintf(f,"H1 %zu\nH2 %zu\nFEAT %zu\nNCLASS %zu\n",H1,H2,FEAT,NCLASS);
            for(size_t k=0;k<FEAT;k++) fprintf(f,"MU %.17g\n",mu[k]);
            for(size_t k=0;k<FEAT;k++) fprintf(f,"SD %.17g\n",sd[k]);
            for(size_t i=0;i<FEAT*H1;i++) fprintf(f,"W1 %.17g\n",m.W1[i]);
            for(size_t i=0;i<H1;i++) fprintf(f,"B1 %.17g\n",m.b1[i]);
            for(size_t i=0;i<H1*H2;i++) fprintf(f,"W2 %.17g\n",m.W2[i]);
            for(size_t i=0;i<H2;i++) fprintf(f,"B2 %.17g\n",m.b2[i]);
            for(size_t i=0;i<H2*NCLASS;i++) fprintf(f,"W3 %.17g\n",m.W3[i]);
            for(size_t i=0;i<NCLASS;i++) fprintf(f,"B3 %.17g\n",m.b3[i]);
            fclose(f);
            printf("\nwrote %s\n",p);
        }
    }

    tgraph_free();
    lancius_arena_destroy(ARENA);
    free(idx); free(mu); free(sd);
    mfree(&m);
    free(tr.X);free(tr.L);free(tr.S);free(tr.A);
    free(va.X);free(va.L);free(va.S);free(va.A);
    free(te.X);free(te.L);free(te.S);free(te.A);
    return 0;
}
