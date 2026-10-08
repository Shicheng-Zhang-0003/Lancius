/* Lancius 5-level veracity verifier -- trained from scratch.
 *
 * Extended boolean scale, ordinal -1 < -0.5 < 0 < +0.5 < +1:
 *   +1.0  plain true                      every checked property correct
 *   +0.5  mostly true, something wrong    no fatal defect, >=1 minor defect
 *    0.0  50/50                           undecidable by construction
 *   -0.5  probably false, something right fatal defect, >=1 property correct
 *   -1.0  plain false                     fatal defect, nothing correct
 *
 * Inputs are MEASUREMENTS from the symbolic checker in tools/prm/verifier5.py
 * (per-step exactness, per-step relative error, sign agreement, structural
 * match, coefficient error, final-result agreement, decidability). Labels are
 * GROUND TRUTH from the injected defect type, decided at generation time and
 * never re-derived by the checker.
 *
 * Two row kinds are trained and scored separately:
 *   S == 0   the whole derivation  -> one overall veracity
 *   S  > 0   one canonical step    -> per-step veracity
 *
 * Objective: mean((logits - onehot)^2) over 5 logits. As established in
 * examples/train_prm_verifier.c, MSE_BWD emits (2/pe)*(z-T) and softmax CE
 * gives (1/rows)*(z-T), so the graph's gradient is exactly (2/NCLASS) times the
 * softmax-CE gradient -- a constant, folded into the learning rate below.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>

#define PI_ 3.14159265358979323846

#include <lancius.h>
#include <lancius/lancius_autodiff.h>

static int    FEAT   = 65;
static int    NCLASS = 5;
static size_t H      = 64;
static double LR     = 0.05, MOM = 0.9, WD = 1e-4;
static size_t BATCH  = 128;
static int    EPOCHS = 60;
static double VALFRAC = 0.15;
static unsigned SEED = 20261007u;

static const double LEVEL[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};

static void die(const char* m){ fprintf(stderr, "FATAL: %s\n", m); exit(2); }
static void* xcalloc(size_t n, size_t sz, const char* what){
    void* p = calloc(n, sz);
    if (!p) { fprintf(stderr, "FATAL: oom %s\n", what); exit(2); }
    return p;
}

/* deterministic RNG (xorshift64*) */
static uint64_t rng_state;
static void rseed(uint64_t s){ rng_state = s*6364136223846793005ULL + 1442695040888963407ULL; }
static double u01(void){
    rng_state = rng_state*6364136223846793007ULL + 1442695040888963407ULL;
    uint64_t z = rng_state;
    z ^= z>>12; z ^= z<<25; z ^= z>>27; z *= 0x2545F4914F6CDD1DULL;
    rng_state = z;
    return (double)(z>>11)/9007199254740992.0;
}
/* Box-Muller: a genuine Gaussian. The previous PRM trainer used a uniform
 * variate times sigma, which has standard deviation sigma/sqrt(3). */
static double gauss_(void){
    double u1 = u01(); if (u1 < 1e-300) u1 = 1e-300;
    double u2 = u01();
    return sqrt(-2.0*log(u1))*cos(2.0*PI_*u2);
}

typedef struct { size_t n; double* X; int* L; int* S; int* A; } corpus;

static corpus load(const char* dir, const char* split){
    corpus c; memset(&c,0,sizeof c);
    char p[512];
    snprintf(p,sizeof p,"%s/%s.X.bin",dir,split);
    FILE* f=fopen(p,"rb"); if(!f){ fprintf(stderr,"cannot open %s\n",p); die("X"); }
    fseek(f,0,SEEK_END); long by=ftell(f); fseek(f,0,SEEK_SET);
    c.n = (size_t)by/(FEAT*(long)sizeof(double));
    c.X = (double*)malloc(c.n*FEAT*sizeof(double));
    if (!c.X) die("oom X");
    if (fread(c.X,sizeof(double),c.n*FEAT,f)!=c.n*FEAT) die("short X");
    fclose(f);
    snprintf(p,sizeof p,"%s/%s.L.bin",dir,split);
    f=fopen(p,"rb"); if(!f) die("L");
    c.L=(int*)malloc(c.n*sizeof(int));
    if (!c.L || fread(c.L,sizeof(int),c.n,f)!=c.n) die("short L");
    fclose(f);
    snprintf(p,sizeof p,"%s/%s.S.bin",dir,split);
    f=fopen(p,"rb"); if(!f) die("S");
    c.S=(int*)malloc(c.n*sizeof(int));
    if (!c.S || fread(c.S,sizeof(int),c.n,f)!=c.n) die("short S");
    fclose(f);
    snprintf(p,sizeof p,"%s/%s.A.bin",dir,split);
    f=fopen(p,"rb"); if(!f) die("A");
    c.A=(int*)malloc(c.n*sizeof(int));
    if (!c.A || fread(c.A,sizeof(int),c.n,f)!=c.n) die("short A");
    fclose(f);
    return c;
}

typedef struct {
    double *W1,*b1,*W2,*b2;
    double *mW1,*v1,*mW2,*v2;
    double *gW1,*gb1,*gW2,*gb2;
    double *Xb,*Tb;
} model;

static void minit(model* m){
    m->W1=(double*)xcalloc(FEAT*H,sizeof(double),"W1");
    m->b1=(double*)xcalloc(H,sizeof(double),"b1");
    m->W2=(double*)xcalloc(H*NCLASS,sizeof(double),"W2");
    m->b2=(double*)xcalloc(NCLASS,sizeof(double),"b2");
    m->mW1=(double*)xcalloc(FEAT*H,sizeof(double),"mW1");
    m->v1 =(double*)xcalloc(H,sizeof(double),"v1");
    m->mW2=(double*)xcalloc(H*NCLASS,sizeof(double),"mW2");
    m->v2 =(double*)xcalloc(NCLASS,sizeof(double),"v2");
    m->gW1=(double*)xcalloc(FEAT*H,sizeof(double),"gW1");
    m->gb1=(double*)xcalloc(H,sizeof(double),"gb1");
    m->gW2=(double*)xcalloc(H*NCLASS,sizeof(double),"gW2");
    m->gb2=(double*)xcalloc(NCLASS,sizeof(double),"gb2");
    m->Xb=(double*)xcalloc(BATCH*FEAT,sizeof(double),"Xb");
    m->Tb=(double*)xcalloc(BATCH*NCLASS,sizeof(double),"Tb");
    double s1=sqrt(2.0/FEAT), s2=sqrt(2.0/(double)H);
    for(size_t i=0;i<FEAT*H;i++) m->W1[i]=gauss_()*s1;
    for(size_t i=0;i<H*NCLASS;i++) m->W2[i]=gauss_()*s2;
}
static void mfree(model* m){
    free(m->W1);free(m->b1);free(m->W2);free(m->b2);
    free(m->mW1);free(m->v1);free(m->mW2);free(m->v2);
    free(m->gW1);free(m->gb1);free(m->gW2);free(m->gb2);
    free(m->Xb);free(m->Tb);
}

static lancius_arena* ARENA = NULL;

static void step_model(model* m, size_t rows){
    lancius_graph* g=lancius_graph_create(); if(!g) die("graph");
    lancius_node* Xn=lancius_input(g,rows,FEAT);
    lancius_node* W1n=lancius_input(g,FEAT,H);
    lancius_node* b1n=lancius_input(g,1,H);
    lancius_node* W2n=lancius_input(g,H,NCLASS);
    lancius_node* b2n=lancius_input(g,1,NCLASS);
    lancius_node* Tn=lancius_input(g,rows,NCLASS);
    if(!Xn||!W1n||!b1n||!W2n||!b2n||!Tn) die("builders");
    lancius_node_bind_external(Xn,m->Xb);
    lancius_node_bind_external(W1n,m->W1);
    lancius_node_bind_external(b1n,m->b1);
    lancius_node_bind_external(W2n,m->W2);
    lancius_node_bind_external(b2n,m->b2);
    lancius_node_bind_external(Tn,m->Tb);
    lancius_node* z1=lancius_add(g,lancius_matmul(g,Xn,W1n),b1n);
    lancius_node* z2=lancius_add(g,lancius_matmul(g,lancius_tanh(g,z1),W2n),b2n);
    lancius_node* loss=lancius_mse(g,z2,Tn);
    if(!loss) die("loss");

    lancius_schedule* fs=lancius_ir_schedule(g); if(!fs) die("fwd sched");
    lancius_schedule_execute(fs,ARENA);
    if(lancius_get_error()!=LANCIUS_ERROR_OK){
        fprintf(stderr,"fwd err=%s\n",lancius_error_string(lancius_get_error())); exit(2);
    }
    lancius_clear_error();
    lancius_training_graph* tg=lancius_ir_autodiff(g,loss); if(!tg) die("autodiff");
    lancius_schedule* bs=lancius_ir_schedule(tg->graph); if(!bs) die("bwd sched");
    lancius_arena_reset(ARENA);
    lancius_schedule_execute(bs,ARENA);
    if(lancius_get_error()!=LANCIUS_ERROR_OK){
        fprintf(stderr,"bwd err=%s\n",lancius_error_string(lancius_get_error())); exit(2);
    }
#define H2(dst,node,cnt) do{ lancius_node* g_=tg->grad_nodes[(node)->id]; \
    if(!g_||!g_->runtime_data){ fprintf(stderr,"FATAL: missing grad\n"); exit(2);} \
    memcpy((dst),g_->runtime_data,(cnt)*sizeof(double)); }while(0)
    H2(m->gW1,W1n,FEAT*H); H2(m->gb1,b1n,H);
    H2(m->gW2,W2n,H*NCLASS); H2(m->gb2,b2n,NCLASS);
#undef H2
    lancius_schedule_destroy(bs);
    lancius_training_graph_destroy(tg);
    lancius_schedule_destroy(fs);
    lancius_graph_destroy(g);

    double lr_eff = LR * 2.0/((double)rows*NCLASS);
#define UPD(W,M,G,N,WDV) do{ for(size_t i=0;i<(size_t)(N);i++){ \
    double gi=(G)[i]+(WDV)*(W)[i]; (M)[i]=MOM*(M)[i]+gi; (W)[i]-=lr_eff*(M)[i]; } }while(0)
    UPD(m->W1,m->mW1,m->gW1,FEAT*H,WD);
    UPD(m->b1,m->v1 ,m->gb1,H,0.0);
    UPD(m->W2,m->mW2,m->gW2,H*NCLASS,WD);
    UPD(m->b2,m->v2 ,m->gb2,NCLASS,0.0);
#undef UPD
}

static void fwd_logits(const model* m,const double* X,size_t n,double* z){
    double hid[512];
    if (H > 512) die("H exceeds fwd_logits stack buffer");
    for(size_t i=0;i<n;i++){
        for(size_t j=0;j<H;j++){
            double s=m->b1[j];
            for(int k=0;k<FEAT;k++) s+=X[i*FEAT+k]*m->W1[k*H+j];
            hid[j]=tanh(s);
        }
        for(int c=0;c<NCLASS;c++){
            double s=m->b2[c];
            for(size_t j=0;j<H;j++) s+=hid[j]*m->W2[j*NCLASS+c];
            z[i*NCLASS+c]=s;
        }
    }
}

static double mse_obj(const double* z,const int* L,size_t n){
    double s=0;
    for(size_t i=0;i<(size_t)n*NCLASS;i++){
        int row=(int)(i/NCLASS), c=(int)(i%NCLASS);
        double y=(L[row]==c)?1.0:0.0;
        double d=z[i]-y; s+=d*d;
    }
    return s/((double)n*NCLASS);
}

/* ---- metrics ---- */
typedef struct {
    size_t n, exact, within1;
    double mae, macro_f1, kappa;
    long cm[5][5];
} met;

static void mreset(met* m){ memset(m,0,sizeof *m); }
static void macc(met* m,const model* md,const double* X,const int* L,const int* S,
                 size_t n,int want_step){
    double* z=(double*)malloc(n*NCLASS*sizeof(double));
    fwd_logits(md,X,n,z);
    long correct=0, w1=0; double absd=0;
    memset(m->cm,0,sizeof m->cm);
    for(size_t i=0;i<n;i++){
        int is_step = (S && S[i]>0);
        if (is_step != want_step) continue;
        int p=0;
        for(int c=1;c<NCLASS;c++) if(z[i*NCLASS+c]>z[i*NCLASS+p]) p=c;
        int t=L[i];
        m->cm[t][p]++;
        if(p==t){ correct++; w1++; }
        else if(abs(p-t)<=1) w1++;
        absd += fabs(LEVEL[p]-LEVEL[t]);
        m->n++;
    }
    m->exact=correct; m->within1=w1;
    m->mae = m->n? absd/(double)m->n : 0.0;
    double f1=0;
    for(int c=0;c<NCLASS;c++){
        double tp=m->cm[c][c], fp=0, fn=0;
        for(int r=0;r<NCLASS;r++){ if(r!=c) fp+=m->cm[r][c]; fn+=m->cm[c][r]; }
        double pr=(tp+fp>0)?tp/(tp+fp):0.0, rc=(tp+fn>0)?tp/(tp+fn):0.0;
        f1 += (pr+rc>0)? 2*pr*rc/(pr+rc):0.0;
    }
    m->macro_f1=f1/NCLASS;
    double po=m->n?(double)correct/(double)m->n:0.0, pe=0;
    for(int c=0;c<NCLASS;c++){
        double rt=0,ct=0;
        for(int r=0;r<NCLASS;r++){ rt+=m->cm[r][c]; ct+=m->cm[r][c]; }
        pe += (m->n&&rt>0)? (rt/m->n)*(ct/m->n) : 0.0;
    }
    m->kappa = (pe<1.0)? (po-pe)/(1.0-pe) : 0.0;
    free(z);
}
static void mprint(const char* tag,const met* m){
    printf("%-10s n=%-6zu exact=%.4f within1=%.4f macroF1=%.4f MAE(level)=%.4f kappa=%.4f\n",
           tag,m->n, m->n?(double)m->exact/(double)m->n:0.0,
           m->n?(double)m->within1/(double)m->n:0.0, m->macro_f1, m->mae, m->kappa);
}
static void confusion(const char* tag,const met* m){
    printf("\nconfusion (%s) rows = true {-1,-.5,0,+.5,+1}, cols = predicted\n",tag);
    printf("        ");
    for(int c=0;c<NCLASS;c++) printf("%9.1f",LEVEL[c]);
    printf("\n");
    for(int r=0;r<NCLASS;r++){
        printf("%+5.1f  ",LEVEL[r]);
        for(int c=0;c<NCLASS;c++) printf("%9ld",m->cm[r][c]);
        printf("\n");
    }
}


/* ---- .lancius export ----------------------------------------------------
 *
 * The v2 format serialises node structure for every node, and node VALUES only
 * for LANCIUS_OP_INPUT nodes that have runtime_data bound (lancius_serialize.c
 * sets has_weights on exactly that condition). A LANCIUS_OP_CONST node carries
 * only a scalar attr_val, so weights cannot ride along as constants. A frozen
 * inference model therefore has to be built with its weights as bound INPUT
 * nodes and its features as an unbound INPUT.
 *
 * The stable C API cannot express this graph -- lancius_graph_handle is a
 * wrapper struct, not a lancius_graph*, and its builders expose only
 * input/matmul/relu, with no tanh and no bias add. lancius_graph_save() is what
 * lancius_graph_save_stable() delegates to anyway, so calling it directly is
 * the same code path without the ABI detour.
 */
static int export_lancius(const model* m, const char* path, size_t rows) {
    lancius_graph* g = lancius_graph_create();
    if (!g) return -1;
    /* The batch dimension is FIXED in the saved file: node shapes are written
     * out and the loader restores them, so patching Xn->shape[0] after load
     * does not propagate to the downstream nodes. A frozen model therefore
     * declares its batch size at export time and is verified at that size. */
    if (rows < 1) rows = 1;
    lancius_node* Xn  = lancius_input(g, rows, FEAT);
    lancius_node* W1n = lancius_input(g, FEAT, H);
    lancius_node* b1n = lancius_input(g, 1, H);
    lancius_node* W2n = lancius_input(g, H, NCLASS);
    lancius_node* b2n = lancius_input(g, 1, NCLASS);
    if (!Xn||!W1n||!b1n||!W2n||!b2n) {
        fprintf(stderr, "export: node builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g); return -1;
    }
    /* only the weights carry data; X stays unbound so its values are NOT saved */
    lancius_node_bind_external(W1n, (double*)m->W1);
    lancius_node_bind_external(b1n, (double*)m->b1);
    lancius_node_bind_external(W2n, (double*)m->W2);
    lancius_node_bind_external(b2n, (double*)m->b2);
    lancius_node* z1 = lancius_add(g, lancius_matmul(g, Xn, W1n), b1n);
    lancius_node* z2 = lancius_add(g, lancius_matmul(g, lancius_tanh(g, z1), W2n), b2n);
    if (!z1 || !z2) {
        fprintf(stderr, "export: op builder failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g); return -1;
    }
    if (lancius_graph_save(g, path) != 0) {
        fprintf(stderr, "lancius_graph_save failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g);
        return -1;
    }
    lancius_graph_destroy(g);
    return 0;
}

/* Reload a .lancius model and confirm it reproduces the in-memory logits.
 * A save path that is never read back is an untested claim. */
static int verify_lancius(const char* path, const model* m,
                          const double* X, size_t n) {
    lancius_graph* g = lancius_graph_load(path);
    if (!g) {
        fprintf(stderr, "load %s failed: %s\n", path,
                lancius_error_string(lancius_get_error()));
        return -1;
    }
    /* find the feature INPUT: the INPUT whose second dim is FEAT and whose
     * runtime_data is NULL (weights were saved, features were not) */
    lancius_node* Xn = NULL;
    for (uint32_t i = 0; i < g->node_count; i++) {
        lancius_node* nd = g->nodes[i];
        if (nd && nd->op == LANCIUS_OP_INPUT && nd->ndim == 2 &&
            nd->shape[1] == (size_t)FEAT && nd->runtime_data == NULL) {
            if (n != nd->shape[0]) {
                fprintf(stderr, "verify: saved batch is %zu, asked for %zu; "
                        "re-export with --lancius-rows %zu\n",
                        nd->shape[0], n, n);
                lancius_graph_destroy(g); return -1;
            }
            Xn = nd; break;
        }
    }
    if (!Xn) { fprintf(stderr, "could not locate feature INPUT in %s\n", path); lancius_graph_destroy(g); return -1; }

    lancius_arena* ar = lancius_arena_create(256u*1024u*1024u);
    if (!ar) { lancius_graph_destroy(g); return -1; }
    lancius_node_bind_external(Xn, (double*)X);
    lancius_schedule* sch = lancius_ir_schedule(g);
    if (!sch) { lancius_arena_destroy(ar); lancius_graph_destroy(g); return -1; }
    lancius_schedule_execute(sch, ar);
    int err = (int)lancius_get_error();
    if (err != 0) {
        fprintf(stderr, "reloaded graph execute err=%s\n", lancius_error_string(lancius_get_error()));
        lancius_schedule_destroy(sch); lancius_arena_destroy(ar); lancius_graph_destroy(g);
        return -1;
    }
    /* the graph's last node is the logits; find the sink by walking to a node
     * with no consumers would need a reverse map, so locate it as the INPUT-free
     * ADD/MATMUL output: instead, read every node and take the widest 2-D result */
    /* Locate the SINK: a computed [n, NCLASS] node that no other node
     * consumes. Matching on shape alone is not sufficient and cost two wrong
     * answers in a row here -- W2 is [H, NCLASS] and so is the intermediate
     * matmul, so "first match" compared logits against the weights (max |diff|
     * 4.04) and "first computed match" compared them against the matmul output
     * (max |diff| 0.217). Both were artifacts of the check, not of the export;
     * the sink itself reproduces to 2.2e-16. */
    lancius_node* out = NULL;
    for (uint32_t i = 0; i < g->node_count && !out; i++) {
        lancius_node* nd = g->nodes[i];
        if (!nd || !nd->runtime_data || nd->ndim != 2) continue;
        if (nd->op == LANCIUS_OP_INPUT || nd->op == LANCIUS_OP_CONST) continue;
        if (nd->shape[0] != n || nd->shape[1] != (size_t)NCLASS) continue;
        int consumed = 0;
        for (uint32_t j = 0; j < g->node_count && !consumed; j++) {
            lancius_node* m = g->nodes[j];
            if (!m || !m->inputs) continue;
            for (int c = 0; c < (int)m->input_count; c++)
                if (m->inputs[c] == nd) { consumed = 1; break; }
        }
        if (!consumed) out = nd;
    }
    if (!out) {
        fprintf(stderr, "verify: no unconsumed [n,%d] node found in %s\n", NCLASS, path);
        lancius_schedule_destroy(sch); lancius_arena_destroy(ar); lancius_graph_destroy(g);
        return -1;
    }

    double worst = 0.0;
    if (out) {
        double* ref = (double*)malloc(n*NCLASS*sizeof(double));
        fwd_logits(m, X, n, ref);
        for (size_t i = 0; i < n*NCLASS; i++) {
            double d = fabs(ref[i] - out->runtime_data[i]);
            if (d > worst) worst = d;
        }
        free(ref);
    }
    lancius_schedule_destroy(sch);
    lancius_arena_destroy(ar);
    lancius_graph_destroy(g);
    printf("\n.lancius round-trip: %s\n", path);
    printf("  reloaded graph reproduces in-memory logits, max |diff| = %.3e  %s\n",
           worst, worst < 1e-12 ? "OK" : "MISMATCH");
    return worst < 1e-12 ? 0 : -1;
}

/* ---- self-contained gradient check (graph's own MSE objective) ---- */
static void gradcheck(void){
    size_t n=BATCH;
    double* W1=(double*)xcalloc(FEAT*H,sizeof(double),"gW1");
    double* b1=(double*)xcalloc(H,sizeof(double),"gb1");
    double* W2=(double*)xcalloc(H*NCLASS,sizeof(double),"gW2");
    double* b2=(double*)xcalloc(NCLASS,sizeof(double),"gb2");
    double* X =(double*)xcalloc(n*FEAT,sizeof(double),"gX");
    double* T =(double*)xcalloc(n*NCLASS,sizeof(double),"gT");
    int*   L =(int*)xcalloc(n,sizeof(int),"gL");
    rseed(SEED^0x5EEDu);
    for(size_t i=0;i<FEAT*H;i++) W1[i]=gauss_()*sqrt(2.0/FEAT);
    for(size_t i=0;i<H*NCLASS;i++) W2[i]=gauss_()*sqrt(2.0/(double)H);
    for(size_t i=0;i<n*FEAT;i++) X[i]=gauss_();
    for(size_t i=0;i<n;i++){ L[i]=(int)(u01()*NCLASS)%NCLASS;
        for(int c=0;c<NCLASS;c++) T[i*NCLASS+c]=0.0;
        T[i*NCLASS+L[i]]=1.0; }
    lancius_arena* ar=lancius_arena_create(128u*1024u*1024u); if(!ar) die("gc arena");
    lancius_graph* g=lancius_graph_create(); if(!g) die("gc graph");
    lancius_node* Xn=lancius_input(g,n,FEAT);
    lancius_node* W1n=lancius_input(g,FEAT,H);
    lancius_node* b1n=lancius_input(g,1,H);
    lancius_node* W2n=lancius_input(g,H,NCLASS);
    lancius_node* b2n=lancius_input(g,1,NCLASS);
    lancius_node* Tn=lancius_input(g,n,NCLASS);
    lancius_node_bind_external(Xn,X); lancius_node_bind_external(W1n,W1);
    lancius_node_bind_external(b1n,b1); lancius_node_bind_external(W2n,W2);
    lancius_node_bind_external(b2n,b2); lancius_node_bind_external(Tn,T);
    lancius_node* z1=lancius_add(g,lancius_matmul(g,Xn,W1n),b1n);
    lancius_node* z2=lancius_add(g,lancius_matmul(g,lancius_tanh(g,z1),W2n),b2n);
    lancius_node* loss=lancius_mse(g,z2,Tn);
    lancius_schedule* fs=lancius_ir_schedule(g);
    lancius_schedule_execute(fs,ar);
    lancius_clear_error();
    lancius_training_graph* tg=lancius_ir_autodiff(g,loss); if(!tg) die("gc autodiff");
    lancius_schedule* bs=lancius_ir_schedule(tg->graph);
    lancius_arena_reset(ar);
    lancius_schedule_execute(bs,ar);
    lancius_node* gW1=tg->grad_nodes[W1n->id];
    double* GW=(double*)xcalloc(FEAT*H,sizeof(double),"gcGW");
    memcpy(GW,gW1->runtime_data,FEAT*H*sizeof(double));
    lancius_schedule_destroy(bs); lancius_training_graph_destroy(tg);
    lancius_schedule_destroy(fs); lancius_graph_destroy(g); lancius_arena_destroy(ar);

    model rm; memset(&rm,0,sizeof rm); rm.W1=W1; rm.b1=b1; rm.W2=W2; rm.b2=b2;
    double* Z=(double*)malloc(n*NCLASS*sizeof(double));
    double ss=0; for(size_t i=0;i<FEAT*H;i++) ss+=GW[i]*GW[i];
    double rms=sqrt(ss/(double)(FEAT*H)); if(!(rms>0)) rms=1.0;
    double worst=0; int probes=0; double hh=1e-6;
    for(int t=0;t<40 && probes<20;t++){
        size_t idx=(size_t)((t*7919u)%(FEAT*H));
        double o=W1[idx];
        W1[idx]=o+hh; fwd_logits(&rm,X,n,Z); double lp=mse_obj(Z,L,n);
        W1[idx]=o-hh; fwd_logits(&rm,X,n,Z); double lm=mse_obj(Z,L,n);
        W1[idx]=o;
        double fd=(lp-lm)/(2.0*hh);
        double den=fmax(fmax(fabs(fd),fabs(GW[idx])),rms);
        double e=fabs(fd-GW[idx])/den;
        if(e>worst) worst=e;
        probes++;
    }
    printf("\ngradient check: autodiff vs central differences of mean-MSE (self-contained)\n"
           "  %d probes on W1, worst scale-relative err = %.3e  %s\n",
           probes,worst,worst<1e-5?"OK":"MISMATCH");
    free(Z); free(GW); free(W1); free(b1); free(W2); free(b2); free(X); free(T); free(L);
}

int main(int argc,char** argv){
    const char* dir = argc>1?argv[1]:"data_vec/v5";
    if(argc>2) H=strtoul(argv[2],NULL,10);
    if(argc>3) EPOCHS=atoi(argv[3]);
    if(argc>4) BATCH=strtoul(argv[4],NULL,10);
    const char* export_path = NULL; int do_verify_rt = 0; size_t lancius_rows = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) SEED = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--export-lancius") && i + 1 < argc) export_path = argv[++i];
        else if (!strcmp(argv[i], "--verify-lancius")) do_verify_rt = 1;
        else if (!strcmp(argv[i], "--lancius-rows") && i + 1 < argc)
            lancius_rows = (size_t)strtoul(argv[++i], NULL, 10);
    }

    corpus tr=load(dir,"train"), te=load(dir,"test");

    /* shuffle train, then carve a VALIDATION split out of it. Model selection
     * uses validation only; test is touched once, at the end. */
    size_t* idx=(size_t*)malloc(tr.n*sizeof(size_t));
    for(size_t i=0;i<tr.n;i++) idx[i]=i;
    rseed(SEED);
    for(size_t i=tr.n-1;i>0;i--){
        size_t j=(size_t)(u01()*(double)(i+1));
        size_t t=idx[i]; idx[i]=idx[j]; idx[j]=t;
    }
    size_t nval=(size_t)(VALFRAC*(double)tr.n), nfit=tr.n-nval;

    /* standardise on the FIT portion only */
    double* mu=(double*)xcalloc(FEAT,sizeof(double),"mu");
    double* sd=(double*)xcalloc(FEAT,sizeof(double),"sd");
    for(size_t q=0;q<nfit;q++){ size_t i=idx[q];
        for(int k=0;k<FEAT;k++) mu[k]+=tr.X[i*FEAT+k]; }
    for(int k=0;k<FEAT;k++) mu[k]/=(double)nfit;
    for(size_t q=0;q<nfit;q++){ size_t i=idx[q];
        for(int k=0;k<FEAT;k++){ double d=tr.X[i*FEAT+k]-mu[k]; sd[k]+=d*d; } }
    for(int k=0;k<FEAT;k++){ sd[k]=sqrt(sd[k]/(double)nfit); if(!(sd[k]>1e-12)) sd[k]=1.0; }
    for(size_t q=0;q<nfit;q++){ size_t i=idx[q];
        for(int k=0;k<FEAT;k++) tr.X[i*FEAT+k]=(tr.X[i*FEAT+k]-mu[k])/sd[k]; }
    for(size_t q=0;q<nval;q++){ size_t i=idx[nfit+q];
        for(int k=0;k<FEAT;k++) tr.X[i*FEAT+k]=(tr.X[i*FEAT+k]-mu[k])/sd[k]; }
    for(size_t i=0;i<te.n;i++)
        for(int k=0;k<FEAT;k++) te.X[i*FEAT+k]=(te.X[i*FEAT+k]-mu[k])/sd[k];

    printf("================================================================\n");
    printf("  Lancius 5-LEVEL VERACITY VERIFIER  (from scratch)\n");
    printf("  scale {-1, -0.5, 0, +0.5, +1}   %d feats -> %zu hidden -> %d\n",FEAT,H,NCLASS);
    printf("================================================================\n");
    printf("fit=%zu  validation=%zu  test=%zu   epochs=%d batch=%zu lr=%.3g\n",
           nfit,nval,te.n,EPOCHS,BATCH,LR);

    gradcheck();

    ARENA=lancius_arena_create(512u*1024u*1024u); if(!ARENA) die("arena");
    model m; minit(&m);
    rseed(SEED);

    double best_val=-1.0;
    double* bW1=(double*)malloc(FEAT*H*sizeof(double));
    double* bW2=(double*)malloc(H*NCLASS*sizeof(double));
    double* bb1=(double*)malloc(H*sizeof(double));
    double* bb2=(double*)malloc(NCLASS*sizeof(double));
    int best_ep=-1;

    /* The train split was shuffled, so its validation rows are NOT contiguous
     * in memory. An earlier version passed tr.X + idx[nfit]*FEAT straight to
     * the forward pass, which read the wrong (and partly out-of-bounds) rows:
     * validation reported 0.28 while the same model scored 0.70 on test, and
     * every epoch-selection decision was made on that noise. Gather once. */
    double* valX=(double*)malloc(nval*FEAT*sizeof(double));
    int*    valL=(int*)malloc(nval*sizeof(int));
    int*    valS=(int*)malloc(nval*sizeof(int));
    for(size_t q=0;q<nval;q++){
        size_t i=idx[nfit+q];
        memcpy(valX+q*FEAT, tr.X+i*FEAT, FEAT*sizeof(double));
        valL[q]=tr.L[i]; valS[q]=tr.S[i];
    }

    printf("\nepoch  fit-CE   val-exact  val-MAE   (model selection = validation)\n");
    printf("--------------------------------------------------------------\n");
    for(int ep=0;ep<EPOCHS;ep++){
        for(size_t i=nfit;i>0;i--){ /* reshuffle the fit portion */
            size_t j=(size_t)(u01()*(double)i);
            size_t t=idx[i-1]; idx[i-1]=idx[nfit-1-j]; idx[nfit-1-j]=t;
        }
        for(size_t s=0;s<nfit;s+=BATCH){
            size_t rows=nfit-s; if(rows>BATCH) rows=BATCH;
            for(size_t r=0;r<rows;r++){
                size_t src=idx[s+r];
                memcpy(m.Xb+r*FEAT,tr.X+src*FEAT,FEAT*sizeof(double));
                for(int c=0;c<NCLASS;c++) m.Tb[r*NCLASS+c]=0.0;
                m.Tb[r*NCLASS+tr.L[src]]=1.0;
            }
            step_model(&m,rows);
        }
        /* validation on overall rows only */
        double* vz=(double*)malloc(nval*NCLASS*sizeof(double));
        fwd_logits(&m,valX,nval,vz);
        /* score overall rows only, so this is comparable with the test report */
        long ok=0, tot=0; double absd=0;
        for(size_t q=0;q<nval;q++){
            if(valS[q]>0) continue;
            int p=0; for(int c=1;c<NCLASS;c++) if(vz[q*NCLASS+c]>vz[q*NCLASS+p]) p=c;
            if(p==valL[q]) ok++;
            absd+=fabs(LEVEL[p]-LEVEL[valL[q]]);
            tot++;
        }
        double vexact = tot? (double)ok/(double)tot : 0.0;
        double vmae   = tot? absd/(double)tot : 0.0;
        free(vz);
        if(vexact>best_val){
            best_val=vexact; best_ep=ep;
            memcpy(bW1,m.W1,FEAT*H*sizeof(double));
            memcpy(bW2,m.W2,H*NCLASS*sizeof(double));
            memcpy(bb1,m.b1,H*sizeof(double));
            memcpy(bb2,m.b2,NCLASS*sizeof(double));
        }
        if((ep+1)%5==0||ep==0||ep==EPOCHS-1)
            printf("%5d  %8s  %8.4f   %7.4f\n",ep+1,"-",vexact,vmae);
    }
    /* restore the epoch selected on VALIDATION (not on test) */
    memcpy(m.W1,bW1,FEAT*H*sizeof(double));
    memcpy(m.W2,bW2,H*NCLASS*sizeof(double));
    memcpy(m.b1,bb1,H*sizeof(double));
    memcpy(m.b2,bb2,NCLASS*sizeof(double));

    printf("\nselected epoch %d on validation (val exact=%.4f)\n",best_ep+1,best_val);

    printf("\n================ HELD-OUT TEST ================\n");
    met mo, ms; mreset(&mo); mreset(&ms);
    macc(&mo,&m,te.X,te.L,te.S,te.n,0);
    macc(&ms,&m,te.X,te.L,te.S,te.n,1);
    mprint("overall",&mo);
    mprint("per-step",&ms);
    confusion("overall",&mo);
    confusion("per-step",&ms);

    /* .lancius export + round-trip proof, on the validation rows so the test
     * set is not touched by a correctness check */
    if (export_path) {
        if (export_lancius(&m, export_path, lancius_rows) == 0) {
            size_t fsz = 0;
            { struct stat st_; if (stat(export_path, &st_) == 0) fsz = (size_t)st_.st_size; }
            printf("\nwrote %s (%zu bytes, fixed batch = %zu rows)\n",
                   export_path, fsz, lancius_rows);
        } else {
            fprintf(stderr, "export FAILED\n"); return 2;
        }
        if (do_verify_rt) {
            size_t rn = nval < lancius_rows ? nval : lancius_rows;
            if (verify_lancius(export_path, &m, valX, rn) != 0) {
                fprintf(stderr, "round-trip FAILED\n"); return 2;
            }
        }
    }

    /* per-algorithm held-out accuracy. The generator writes a plain
     * algos.txt (one name per line) alongside the bins; parsing JSON here was
     * a bad idea and silently reported zero algorithms. */
    {
        char mp[600]; snprintf(mp,sizeof mp,"%s/algos.txt",dir);
        FILE* mf=fopen(mp,"r");
        if (mf) {
            char names[64][64]; int nalgo=0;
            while (nalgo<64 && fgets(names[nalgo],sizeof names[nalgo],mf)) {
                char* nl=strchr(names[nalgo],'\n'); if(nl) *nl=0;
                if(names[nalgo][0]) nalgo++;
            }
            fclose(mf);
            int nrows[64]={0}, nok[64]={0};
            double* z=(double*)malloc(te.n*NCLASS*sizeof(double));
            if (z) {
                fwd_logits(&m,te.X,te.n,z);
                for(size_t i=0;i<te.n;i++){
                    if (te.S[i]!=0) continue;
                    int ai=te.A[i];
                    if (ai<0||ai>=nalgo) continue;
                    int p=0; for(int c=1;c<NCLASS;c++) if(z[i*NCLASS+c]>z[i*NCLASS+p]) p=c;
                    nrows[ai]++; if(p==te.L[i]) nok[ai]++;
                }
            }
            free(z);
            printf("\nper-algorithm held-out accuracy (overall rows)\n");
            printf("  %-3s %-26s %8s\n","#","algorithm","acc");
            for(int a2=0;a2<nalgo;a2++)
                printf("  %-3d %-26s %8.4f  (%d/%d)\n",a2,names[a2],
                       nrows[a2]? (double)nok[a2]/nrows[a2] : 0.0, nok[a2], nrows[a2]);
        } else fprintf(stderr,"WARN: no algos.txt, skipping per-algorithm report\n");
    }

    /* export */
    {
        char p[600]; snprintf(p,sizeof p,"%s/model5.txt",dir);
        FILE* f=fopen(p,"w");
        if(!f) fprintf(stderr,"WARN: cannot write %s\n",p);
        else{
            fprintf(f,"H %zu\nFEAT %d\nNCLASS %d\n",H,FEAT,NCLASS);
            for(int k=0;k<FEAT;k++) fprintf(f,"MU %.17g\n",mu[k]);
            for(int k=0;k<FEAT;k++) fprintf(f,"SD %.17g\n",sd[k]);
            for(size_t i=0;i<FEAT*H;i++) fprintf(f,"W1 %.17g\n",m.W1[i]);
            for(size_t i=0;i<H;i++) fprintf(f,"B1 %.17g\n",m.b1[i]);
            for(size_t i=0;i<H*NCLASS;i++) fprintf(f,"W2 %.17g\n",m.W2[i]);
            for(int i=0;i<NCLASS;i++) fprintf(f,"B2 %.17g\n",m.b2[i]);
            fclose(f);
            printf("\nwrote %s\n",p);
        }
    }
    free(valX); free(valL); free(valS);
    free(idx); free(mu); free(sd); free(bW1); free(bW2); free(bb1); free(bb2);
    mfree(&m);
    free(tr.X); free(tr.L); free(tr.S); free(te.X); free(te.L); free(te.S);
    return 0;
}
