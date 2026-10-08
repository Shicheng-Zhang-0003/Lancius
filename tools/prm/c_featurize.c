#define _POSIX_C_SOURCE 200809L
/*
 * Featurizer equivalence gate.
 *
 * prm_extract.py reimplements featurize() from examples/distill_prm800k.c in
 * Python so that the real (all-completions) corpus can be built. That is only
 * legitimate if the two agree exactly, so this program reads a JSONL corpus,
 * runs the SHIPPED C featurize() over the same texts, and writes the result;
 * verify_featurizer.py then requires bit-identical float64 across all 16 lanes
 * for every completion.
 *
 * Without this the Python reimplementation would be an unverified reimplementation,
 * which is the exact failure mode this project exists to avoid.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define FEAT_DIM 16
#define MAX_JSON_DEPTH 256

/* ---- minimal JSON: enough for PRM800K rows, hostile-input bounded ---- */
typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype_t;
typedef struct jval {
    jtype_t type; double num; int num_is_int; int boolean;
    char* str; size_t slen;
    struct jval** items; size_t nitems;
    char** keys; struct jval** vals; size_t nfields;
} jval_t;

typedef struct { const char* p, *end; int err, depth; } pstate_t;

static void skip_ws(pstate_t* s) {
    while (s->p < s->end) {
        char c = *s->p;
        if (c==' '||c=='\t'||c=='\n'||c=='\r') s->p++; else break;
    }
}
static jval_t* parse_value(pstate_t* s);

static size_t utf8_emit_len(unsigned cp, char* tmp, size_t cap) {
    if (cp < 0x80)          { if (cap < 1) return 0; tmp[0]=(char)cp; return 1; }
    if (cp < 0x800)         { if (cap < 2) return 0; tmp[0]=(char)(0xC0|(cp>>6)); tmp[1]=(char)(0x80|(cp&0x3F)); return 2; }
    if (cp < 0x10000)       { if (cap < 3) return 0; tmp[0]=(char)(0xE0|(cp>>12)); tmp[1]=(char)(0x80|((cp>>6)&0x3F)); tmp[2]=(char)(0x80|(cp&0x3F)); return 3; }
    if (cp <= 0x10FFFF)     { if (cap < 4) return 0; tmp[0]=(char)(0xF0|(cp>>18)); tmp[1]=(char)(0x80|((cp>>12)&0x3F)); tmp[2]=(char)(0x80|((cp>>6)&0x3F)); tmp[3]=(char)(0x80|(cp&0x3F)); return 4; }
    return 0;
}

/* Mirrors parse_string() in examples/distill_prm800k.c, including the detail
 * that matters: buf[len] = 0 is the terminator and v->slen = len EXCLUDES it.
 * Folding the NUL into slen shifts every downstream length by one. */
static char* parse_string_raw(pstate_t* s, size_t* outlen) {
    s->p++; /* opening quote */
    size_t cap = 64, len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) { s->err = 1; return NULL; }
    for (;;) {
        if (s->p >= s->end) { free(buf); s->err = 1; return NULL; }
        char c = *s->p++;
        if (c == '"') break;
        if (c == '\\') {
            if (s->p >= s->end) { free(buf); s->err = 1; return NULL; }
            char e = *s->p++;
            if (e=='"'||e=='\\'||e=='/') c = e;
            else if (e=='b') c = '\b';
            else if (e=='f') c = '\f';
            else if (e=='n') c = '\n';
            else if (e=='r') c = '\r';
            else if (e=='t') c = '\t';
            else if (e=='u') {
                unsigned cp = 0; int ok = 0;
                for (int i = 0; i < 4; i++) {
                    if (s->p >= s->end) { free(buf); s->err = 1; return NULL; }
                    char h = *s->p++;
                    cp <<= 4;
                    if (h>='0'&&h<='9') cp |= (unsigned)(h-'0');
                    else if (h>='a'&&h<='f') cp |= (unsigned)(h-'a'+10);
                    else if (h>='A'&&h<='F') cp |= (unsigned)(h-'A'+10);
                    else { free(buf); s->err = 1; return NULL; }
                    ok = 1;
                }
                (void)ok;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (s->end - s->p >= 6 && s->p[0]=='\\' && s->p[1]=='u') {
                        const char* q = s->p + 2;
                        unsigned lo = 0; int ok2 = 0;
                        for (int i = 0; i < 4; i++) {
                            if (q >= s->end) { ok2 = 0; break; }
                            char h = *q++; lo <<= 4;
                            if (h>='0'&&h<='9') lo |= (unsigned)(h-'0');
                            else if (h>='a'&&h<='f') lo |= (unsigned)(h-'a'+10);
                            else if (h>='A'&&h<='F') lo |= (unsigned)(h-'A'+10);
                            else { ok2 = 0; break; }
                            ok2 = 1;
                        }
                        if (ok2 && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                            s->p = q;
                        } else { free(buf); s->err = 1; return NULL; }
                    } else { free(buf); s->err = 1; return NULL; }
                }
                char tmp[4]; size_t nb = utf8_emit_len(cp, tmp, sizeof tmp);
                if (nb == 0) { free(buf); s->err = 1; return NULL; }
                while (len + nb + 1 > cap) {
                    size_t ncap = cap * 2;
                    char* nbuf = (char*)realloc(buf, ncap);
                    if (!nbuf) { free(buf); s->err = 1; return NULL; }
                    buf = nbuf; cap = ncap;
                }
                memcpy(buf + len, tmp, nb);
                len += nb;
                continue;
            } else { free(buf); s->err = 1; return NULL; }
        }
        if (len + 2 > cap) {
            size_t ncap = cap * 2;
            char* nbuf = (char*)realloc(buf, ncap);
            if (!nbuf) { free(buf); s->err = 1; return NULL; }
            buf = nbuf; cap = ncap;
        }
        buf[len++] = c;
    }
    buf[len] = '\0';
    if (outlen) *outlen = len;      /* EXCLUDES the terminator */
    return buf;
}
static jval_t* jnew(jtype_t t){ jval_t* v=(jval_t*)calloc(1,sizeof(jval_t)); if(v) v->type=t; return v; }
static void jfree(jval_t* v){
    if(!v) return;
    free(v->str);
    for(size_t i=0;i<v->nitems;i++) jfree(v->items[i]);
    free(v->items);
    for(size_t i=0;i<v->nfields;i++){ free(v->keys[i]); jfree(v->vals[i]); }
    free(v->keys); free(v->vals);
    free(v);
}
static void obj_add(jval_t* o, char* k, jval_t* v){
    o->keys=(char**)realloc(o->keys,(o->nfields+1)*sizeof(char*));
    o->vals=(jval_t**)realloc(o->vals,(o->nfields+1)*sizeof(jval_t*));
    o->keys[o->nfields]=k; o->vals[o->nfields]=v; o->nfields++;
}
static jval_t* parse_value(pstate_t* s){
    if (++s->depth > MAX_JSON_DEPTH) { s->err=1; s->depth--; return NULL; }
    skip_ws(s);
    if (s->p >= s->end) { s->err=1; s->depth--; return NULL; }
    jval_t* v = NULL;
    char c = *s->p;
    if (c=='"'){ size_t l; char* str=parse_string_raw(s,&l); v=jnew(J_STR); if(v){v->str=str;v->slen=l;} }
    else if (c=='{'){
        v=jnew(J_OBJ); s->p++;
        skip_ws(s);
        if (s->p<s->end && *s->p=='}') s->p++;
        else for(;;){
            skip_ws(s);
            if (s->p>=s->end || *s->p!='"'){ s->err=1; break; }
            size_t kl; char* key=parse_string_raw(s,&kl);
            skip_ws(s);
            if (s->p>=s->end || *s->p!=':'){ s->err=1; free(key); break; }
            s->p++;
            jval_t* val=parse_value(s);
            if (s->err){ free(key); jfree(val); break; }
            obj_add(v,key,val);
            skip_ws(s);
            if (s->p<s->end && *s->p==','){ s->p++; continue; }
            if (s->p<s->end && *s->p=='}'){ s->p++; break; }
            s->err=1; break;
        }
    }
    else if (c=='['){
        v=jnew(J_ARR); s->p++;
        skip_ws(s);
        if (s->p<s->end && *s->p==']') s->p++;
        else for(;;){
            jval_t* it=parse_value(s);
            if (s->err){ jfree(it); break; }
            v->items=(jval_t**)realloc(v->items,(v->nitems+1)*sizeof(jval_t*));
            v->items[v->nitems++]=it;
            skip_ws(s);
            if (s->p<s->end && *s->p==','){ s->p++; continue; }
            if (s->p<s->end && *s->p==']'){ s->p++; break; }
            s->err=1; break;
        }
    }
    else if (c=='t' && (size_t)(s->end-s->p)>=4 && !memcmp(s->p,"true",4)){ v=jnew(J_BOOL); v->boolean=1; s->p+=4; }
    else if (c=='f' && (size_t)(s->end-s->p)>=5 && !memcmp(s->p,"false",5)){ v=jnew(J_BOOL); v->boolean=0; s->p+=5; }
    else if (c=='n' && (size_t)(s->end-s->p)>=4 && !memcmp(s->p,"null",4)){ v=jnew(J_NULL); s->p+=4; }
    else {
        const char* start=s->p; int isint=1, seen=0;
        if (s->p<s->end && (*s->p=='-'||*s->p=='+')) s->p++;
        while (s->p<s->end && ((*s->p>='0'&&*s->p<='9')||*s->p=='.'||*s->p=='e'||*s->p=='E'||*s->p=='-'||*s->p=='+')){
            if (*s->p=='.'||*s->p=='e'||*s->p=='E') isint=0;
            if (*s->p>='0'&&*s->p<='9') seen=1;
            s->p++;
        }
        if (!seen) { s->err=1; s->depth--; return NULL; }
        char tmp[64]; size_t n=(size_t)(s->p-start); if(n>=sizeof tmp)n=sizeof tmp-1;
        memcpy(tmp,start,n); tmp[n]=0;
        v=jnew(J_NUM); v->num=strtod(tmp,NULL); v->num_is_int=isint;
    }
    s->depth--;
    if (s->err){ jfree(v); return NULL; }
    return v;
}
static jval_t* obj_get(jval_t* o, const char* k){
    if(!o||o->type!=J_OBJ) return NULL;
    for(size_t i=0;i<o->nfields;i++) if(!strcmp(o->keys[i],k)) return o->vals[i];
    return NULL;
}

/* ---- the SHIPPED featurizer, copied verbatim from distill_prm800k.c ---- */
static int is_opc(unsigned long cp) {
    return cp == '+' || cp == '-' || cp == '*' || cp == '/' || cp == '^' || cp == '=';
}
static int is_wsc(unsigned long cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
static int utf8_next(const char** pp, const char* end, unsigned long* out) {
    const char* p = *pp;
    if (p >= end) return 0;
    unsigned char c0 = (unsigned char)*p++;
    if (c0 < 0x80) { *out = c0; *pp = p; return 1; }
    int need; unsigned long cp;
    if (c0 < 0xC0) return 0;
    else if (c0 < 0xE0) { need = 1; cp = c0 & 0x1F; }
    else if (c0 < 0xF0) { need = 2; cp = c0 & 0x0F; }
    else if (c0 < 0xF8) { need = 3; cp = c0 & 0x07; }
    else return 0;
    if ((size_t)(end - p) < (size_t)need) return 0;
    for (int i = 0; i < need; i++) {
        unsigned char ci = (unsigned char)p[i];
        if ((ci & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (ci & 0x3F);
    }
    *pp = p + need; *out = cp; return 1;
}
static void featurize(const char* s, size_t nbytes, long step_idx, long n_steps, double* f) {
    const char* p = s;
    const char* end = s + nbytes;
    unsigned long cp;
    long ncp = 0, nwords = 0, ndigits = 0, nops = 0, nlatex = 0, nnums = 0;
    long max_depth = 0, depth = 0, nupper = 0, nspace = 0;
    long wlen = 0, wsum = 0;
    int in_word = 0, in_num = 0, has_q = 0;
    while (p < end) {
        if (!utf8_next(&p, end, &cp)) break;
        ncp++;
        if (is_wsc(cp)) {
            if (in_word) { nwords++; wsum += wlen; wlen = 0; in_word = 0; }
            nspace++;
            in_num = 0;
            continue;
        }
        if (!in_word) { in_word = 1; wlen = 0; }
        wlen++;
        if (cp >= '0' && cp <= '9') ndigits++;
        if (is_opc(cp)) nops++;
        if (cp == '(') { depth++; if (depth > max_depth) max_depth = depth; }
        else if (cp == ')') { if (depth > 0) depth--; }
        if (cp == '\\') nlatex++;
        if (cp == '?') has_q = 1;
        if (cp >= 'A' && cp <= 'Z') nupper++;
        if ((cp >= '0' && cp <= '9') || cp == '.') {
            if (!in_num) { nnums++; in_num = 1; }
        } else in_num = 0;
    }
    if (in_word) { nwords++; wsum += wlen; }
    {
        const char* t = end;
        while (t > s && (t[-1]==' '||t[-1]=='\t'||t[-1]=='\n'||t[-1]=='\r'||t[-1]=='\f'||t[-1]=='\v')) t--;
        {
            double avg = nwords ? (double)wsum / (double)nwords : 0.0;
            double frac = (double)(step_idx + 1) / (double)(n_steps > 0 ? n_steps : 1);
            char last = (t > s) ? t[-1] : '\0';
            f[0] = log1p((double)ncp);
            f[1] = log1p((double)nwords);
            f[2] = log1p((double)ndigits);
            f[3] = log1p((double)nops);
            f[4] = (double)max_depth;
            f[5] = (double)step_idx;
            f[6] = (double)n_steps;
            f[7] = frac;
            f[8] = log1p((double)nlatex);
            f[9] = log1p((double)nnums);
            f[10] = avg;
            f[11] = (last == '.' || last == '?' || last == '!') ? 1.0 : 0.0;
            f[12] = has_q ? 1.0 : 0.0;
            f[13] = log1p((double)nupper);
            f[14] = log1p((double)nspace);
            f[15] = (n_steps > 0 && step_idx == n_steps - 1) ? 1.0 : 0.0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <train.jsonl> <out.bin>\n", argv[0]); return 2; }
    FILE* in = fopen(argv[1], "rb");
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    FILE* out = fopen(argv[2], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[2]); fclose(in); return 2; }
    char* line = NULL; size_t cap = 0; ssize_t n;
    double f[FEAT_DIM];
    long nrow = 0;
    while ((n = getline(&line, &cap, in)) > 0) {
        while (n > 0 && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n] = 0;
        if (n == 0) continue;
        pstate_t ps; ps.p = line; ps.end = line + n; ps.err = 0; ps.depth = 0;
        jval_t* root = parse_value(&ps);
        if (ps.err || !root) { jfree(root); continue; }
        jval_t* label = obj_get(root, "label");
        jval_t* steps = label ? obj_get(label, "steps") : NULL;
        if (!steps || steps->type != J_ARR) { jfree(root); continue; }
        for (size_t i = 0; i < steps->nitems; i++) {
            jval_t* st = steps->items[i];
            jval_t* comps = obj_get(st, "completions");
            if (!comps || comps->type != J_ARR) continue;
            for (size_t j = 0; j < comps->nitems; j++) {
                jval_t* c = comps->items[j];
                jval_t* rating = obj_get(c, "rating");
                jval_t* text = obj_get(c, "text");
                if (!rating || rating->type != J_NUM) continue;
                if (!(rating->num == -1.0 || rating->num == 0.0 || rating->num == 1.0)) continue;
                if (!text || text->type != J_STR) continue;
                featurize(text->str, text->slen, (long)i, (long)steps->nitems, f);
                fwrite(f, sizeof(double), FEAT_DIM, out);
                nrow++;
            }
        }
        jfree(root);
    }
    free(line); fclose(in); fclose(out);
    fprintf(stderr, "c_featurize_rows=%ld -> %s\n", nrow, argv[2]);
    return 0;
}