/* Pure-C reference trainer for the PRM800K verifier.
 *
 * Purpose: decide whether the Lancius-run training gradients are correct, by
 * training the IDENTICAL model on the IDENTICAL features with hand-derived
 * softmax-cross-entropy gradients and no framework at all. If this lands near
 * the framework's test accuracy, the framework's training gradients are sound
 * and only the trainer's gradient-check harness is mis-plumbed.
 *
 * Everything is mirrored from examples/train_prm_verifier.c:
 *   H=32  LR=0.05  MOM=0.9  WD=1e-4  BATCH=64  EPOCHS=60  SEED=20261007
 *   He init  w = rnd()*sqrt(2/fan_in),  b = 0
 *   Fisher-Yates shuffle with the same xorshift64* RNG
 *   step = LR * (2/pe) * (2/NCLASS) * dCE/dw  +  SGDM with L2 into the gradient
 *
 * The (2/NCLASS) factor reproduces the framework's convention: its graph builds
 * MSE(logits, onehot), whose VJP is (2/pe)*(z-T) = (2/NCLASS)*dCE/dz, and it then
 * steps with LR*(2/pe). So the two paths take the same step iff the gradients
 * agree.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define FD 16
#define NC 3

static size_t H = 32;
static double LR = 0.05, MOM = 0.9, WD = 1e-4;
static size_t BATCH = 64;
static int EPOCHS = 60;
static unsigned SEED = 20261007u;

static uint64_t rng_state;
static void rseed(uint64_t s){ rng_state = s * 6364136223846793005ULL + 1442695040888963407ULL; }
static double rnd(void){
    rng_state = rng_state * 6364136223846793007ULL + 1442695040888963407ULL;
    uint64_t z = rng_state;
    z ^= z >> 12; z ^= z << 25; z ^= z >> 27; z *= 0x2545F4914F6CDD1DULL;
    rng_state = z;
    return (double)(int64_t)(z >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}
static size_t shuf_next(void){
    uint64_t z = rng_state;
    z ^= z >> 12; z ^= z << 25; z ^= z >> 27; z *= 0x2545F4914F6CDD1DULL;
    rng_state = z;
    return (size_t)(z >> 11);
}

typedef struct { size_t n; double* X; int* T; } corpus;

static corpus load(const char* dir, const char* split){
    corpus c; char p[512]; snprintf(p,sizeof p,"%s/%s.X.bin",dir,split);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);}
    fseek(f,0,SEEK_END); long by=ftell(f); fseek(f,0,SEEK_SET);
    c.n=(size_t)by/(FD*sizeof(double));
    c.X=malloc(c.n*FD*sizeof(double));
    if(fread(c.X,sizeof(double),c.n*FD,f)!=c.n*FD){fprintf(stderr,"short X\n");exit(2);}
    fclose(f);
    snprintf(p,sizeof p,"%s/%s.T.bin",dir,split);
    f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(2);}
    fseek(f,0,SEEK_END); long tb=ftell(f); fseek(f,0,SEEK_SET);
    size_t tn=(size_t)tb/sizeof(double);
    if(tn!=c.n){fprintf(stderr,"len mismatch\n");exit(2);}
    double* raw=malloc(tn*sizeof(double));
    if(fread(raw,sizeof(double),tn,f)!=tn){fprintf(stderr,"short T\n");exit(2);}
    fclose(f);
    c.T=malloc(tn*sizeof(int));
    for(size_t i=0;i<tn;i++) c.T[i]=(int)raw[i]+1;   /* -1/0/+1 -> 0/1/2 */
    free(raw);
    return c;
}

static void fwd(const double* X, size_t n, const double* W1, const double* b1,
                const double* W2, const double* b2, double* h, double* z){
    for(size_t i=0;i<n;i++){
        for(size_t j=0;j<H;j++){
            double s=b1[j];
            for(int k=0;k<FD;k++) s+=X[i*FD+k]*W1[k*H+j];
            h[i*H+j]=tanh(s);
        }
        for(int c=0;c<NC;c++){
            double s=b2[c];
            for(size_t j=0;j<H;j++) s+=h[i*H+j]*W2[j*NC+c];
            z[i*NC+c]=s;
        }
    }
}

static double ce_and_grad(const double* X, const int* T, size_t n,
                          const double* W1, const double* b1,
                          const double* W2, const double* b2,
                          double* gW1, double* gb1, double* gW2, double* gb2){
    double* h = malloc(n*H*sizeof(double));
    double* z = malloc(n*NC*sizeof(double));
    fwd(X,n,W1,b1,W2,b2,h,z);
    memset(gW1,0,FD*H*sizeof(double)); memset(gb1,0,H*sizeof(double));
    memset(gW2,0,H*NC*sizeof(double)); memset(gb2,0,NC*sizeof(double));
    double* d2 = malloc(n*NC*sizeof(double));
    double loss=0.0;
    for(size_t i=0;i<n;i++){
        double mx=z[i*NC]; for(int c=1;c<NC;c++) if(z[i*NC+c]>mx) mx=z[i*NC+c];
        double sm=0; for(int c=0;c<NC;c++) sm+=exp(z[i*NC+c]-mx);
        loss += -((z[i*NC+T[i]]-mx)-log(sm));
        for(int c=0;c<NC;c++) d2[i*NC+c]=exp(z[i*NC+c]-mx)/sm;
        d2[i*NC+T[i]] -= 1.0;
        for(int c=0;c<NC;c++) d2[i*NC+c] /= (double)n;     /* dCE/dz */
    }
    for(size_t i=0;i<n;i++){
        for(size_t j=0;j<H;j++){
            double a=0; for(int c=0;c<NC;c++) a+=d2[i*NC+c]*W2[j*NC+c];
            double g = a*(1.0 - h[i*H+j]*h[i*H+j]);           /* dCE/dz1 */
            gb1[j] += g;
            for(int k=0;k<FD;k++) gW1[k*H+j] += X[i*FD+k]*g;
        }
        for(int c=0;c<NC;c++){
            double a=d2[i*NC+c];
            gb2[c] += a;
            for(size_t j=0;j<H;j++) gW2[j*NC+c] += h[i*H+j]*a;
        }
    }
    free(h); free(z); free(d2);
    return loss/(double)n;
}

static void metrics(const double* X, const int* T, size_t n,
                    const double* W1, const double* b1, const double* W2,
                    const double* b2, double* acc, double* f1){
    double* h=malloc(n*H*sizeof(double)), *z=malloc(n*NC*sizeof(double));
    fwd(X,n,W1,b1,W2,b2,h,z);
    long cm[NC][NC]; memset(cm,0,sizeof cm);
    for(size_t i=0;i<n;i++){
        int p=0; for(int c=1;c<NC;c++) if(z[i*NC+c]>z[i*NC+p]) p=c;
        cm[T[i]][p]++;
    }
    long tot=0; for(int a=0;a<NC;a++) for(int b=0;b<NC;b++) tot+=cm[a][b];
    *acc=(double)cm[0][0]+cm[1][1]+cm[2][2]/(double)tot;
    long acc_ok=cm[0][0]+cm[1][1]+cm[2][2];
    *acc=(double)acc_ok/(double)tot;
    double s=0;
    for(int c=0;c<NC;c++){
        double tp=(double)cm[c][c];
        double fp=0, fn=0;
        for(int r=0;r<NC;r++){ if(r!=c) fp+=cm[r][c]; fn+=cm[c][r]; }
        double pr = (tp+fp>0)? tp/(tp+fp) : 0.0;
        double rc = (tp+fn>0)? tp/(tp+fn) : 0.0;
        s += (pr+rc>0)? 2*pr*rc/(pr+rc) : 0.0;
    }
    *f1=s/NC;
    free(h); free(z);
}

int main(int argc, char** argv){
    const char* dir = (argc>1)? argv[1] : "data_vec";
    if(argc>2) H=strtoul(argv[2],NULL,10);
    if(argc>3) EPOCHS=atoi(argv[3]);
    if(argc>4) BATCH=strtoul(argv[4],NULL,10);

    corpus tr=load(dir,"train"), te=load(dir,"test");
    double mu[FD], sd[FD];
    for(int j=0;j<FD;j++){ mu[j]=0; sd[j]=0; }
    for(size_t i=0;i<tr.n;i++) for(int j=0;j<FD;j++) mu[j]+=tr.X[i*FD+j];
    for(int j=0;j<FD;j++) mu[j]/=(double)tr.n;
    for(size_t i=0;i<tr.n;i++) for(int j=0;j<FD;j++){ double d=tr.X[i*FD+j]-mu[j]; sd[j]+=d*d; }
    for(int j=0;j<FD;j++){ sd[j]=sqrt(sd[j]/(double)tr.n); if(!(sd[j]>1e-12)) sd[j]=1.0; }
    for(size_t i=0;i<tr.n;i++) for(int j=0;j<FD;j++) tr.X[i*FD+j]=(tr.X[i*FD+j]-mu[j])/sd[j];
    for(size_t i=0;i<te.n;i++) for(int j=0;j<FD;j++) te.X[i*FD+j]=(te.X[i*FD+j]-mu[j])/sd[j];

    double* W1=malloc(FD*H*sizeof(double)), *b1=calloc(H,sizeof(double));
    double* W2=malloc(H*NC*sizeof(double)), *b2=calloc(NC,sizeof(double));
    double* mW1=calloc(FD*H,sizeof(double)), *vW1=calloc(H,sizeof(double));
    double* mW2=calloc(H*NC,sizeof(double)), *vW2=calloc(NC,sizeof(double));
    double* gW1=malloc(FD*H*sizeof(double)), *gb1=malloc(H*sizeof(double));
    double* gW2=malloc(H*NC*sizeof(double)), *gb2=malloc(NC*sizeof(double));

    rseed(SEED);
    for(size_t i=0;i<FD*H;i++) W1[i]=rnd()*sqrt(2.0/(double)FD);
    for(size_t i=0;i<H*NC;i++) W2[i]=rnd()*sqrt(2.0/(double)H);

    size_t* idx=malloc(tr.n*sizeof(size_t));
    double* xb=malloc(BATCH*FD*sizeof(double));
    int*    tb=malloc(BATCH*sizeof(int));

    double a0,f0; metrics(te.X,te.T,te.n,W1,b1,W2,b2,&a0,&f0);
    printf("corpus: train n=%zu  test n=%zu\n",tr.n,te.n);
    printf("uniform-init test acc=%.4f macroF1=%.4f\n\n",a0,f0);
    printf("epoch   loss      train-acc   test-acc    test-macroF1\n");
    printf("--------------------------------------------------------------\n");

    for(int ep=0;ep<EPOCHS;ep++){
        for(size_t i=0;i<tr.n;i++) idx[i]=i;
        for(size_t i=tr.n-1;i>0;i--){
            size_t j=shuf_next()%(i+1);
            size_t t=idx[i]; idx[i]=idx[j]; idx[j]=t;
        }
        for(size_t start=0; start<tr.n; start+=BATCH){
            size_t rows=tr.n-start; if(rows>BATCH) rows=BATCH;
            for(size_t r=0;r<rows;r++){
                size_t s=idx[start+r];
                memcpy(xb+r*FD, tr.X+s*FD, FD*sizeof(double));
                tb[r]=tr.T[s];
            }
            ce_and_grad(xb,tb,rows,W1,b1,W2,b2,gW1,gb1,gW2,gb2);
            double lr_eff = LR * 2.0/((double)rows*NC) * (2.0/NC);
            #define STEP(W,M,G,N,WDV) do{ \
                for(size_t i=0;i<(N);i++){ \
                    double gi=(G)[i]+(WDV)*(W)[i]; \
                    (M)[i]=MOM*(M)[i]+gi; \
                    (W)[i]-=lr_eff*(M)[i]; } }while(0)
            STEP(W1,mW1,gW1,FD*H,WD);
            STEP(b1,vW1,gb1,H,0.0);
            STEP(W2,mW2,gW2,H*NC,WD);
            STEP(b2,vW2,gb2,NC,0.0);
            #undef STEP
        }
        double full=ce_and_grad(tr.X,tr.T,tr.n,W1,b1,W2,b2,gW1,gb1,gW2,gb2);
        double ta,tf; metrics(tr.X,tr.T,tr.n,W1,b1,W2,b2,&ta,&tf);
        double ea,ef; metrics(te.X,te.T,te.n,W1,b1,W2,b2,&ea,&ef);
        if(ep==0||(ep+1)%5==0||ep==EPOCHS-1)
            printf("%4d  %8.5f   %.4f      %.4f      %.4f\n",ep+1,full,ta,ea,ef);
    }
    return 0;
}
