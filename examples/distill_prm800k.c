#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>

/*
 * LANCIUS distill_prm800k (C): PRM800k step rows -> fixed numeric vectors.
 *
 * Replaces distill_prm800k.py (retired): the Python version is stdlib-only
 * (JSON + math), so a C port costs no third-party dependency. Schemas that
 * need torch/onnx/network access stay in Python — porting those would cost
 * a bloody arm and a leg for zero gain.
 *
 * Equivalence contract with the retired script (verified byte-identical
 * .bin output on the vendored corpus):
 *  - JSON strings decoded with full escapes incl. \uXXXX + surrogates.
 *  - Lengths counted in Unicode code points (UTF-8 decoded), matching
 *    Python len(str). Property checks (digit/upper/space/whitespace-split)
 *    are ASCII-scoped: EXACT wherever no non-ASCII digit/upper/space
 *    character occurs (true of the entire vendored corpus — the only
 *    non-ASCII chosen-text chars are U+200B, U+2019, U+00D7, all
 *    property-negative); documented boundary otherwise.
 *  - Ratings accepted iff numeric value in {-1,0,1} (covers 1.0, like
 *    Python's `in` test). Choices must be integer-typed (Python crashes
 *    on float indices, so no valid-data divergence).
 *  - Non-string text, null/out-of-range choice, bad rating, broken JSON:
 *    row skipped and counted, never fatal.
 *
 * Outputs (float64 LE, row-major): <out>/<split>.X.bin [n,16],
 * <out>/<split>.T.bin [n], <out>/<split>.meta.json. Exit 0 on success
 * (single-class splits reported, not hidden), 1 on I/O failure.
 * `--selftest` runs embedded parser/featurizer proofs, no data needed.
 */

#define FEAT_DIM 16
#define MAX_FILE_BYTES ((size_t)512 * 1024 * 1024)

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype_t;

typedef struct jval {
    jtype_t type;
    double num;          /* J_NUM */
    int num_is_int;      /* J_NUM: integer syntax (no ./e) */
    int boolean;         /* J_BOOL */
    char* str;           /* J_STR: decoded UTF-8, NUL-terminated */
    size_t slen;         /* J_STR: byte length excl. NUL */
    struct jval** items; /* J_ARR */
    size_t nitems;
    char** keys;         /* J_OBJ */
    struct jval** vals;
    size_t nfields;
} jval_t;

/* Hostile-file guard: nesting deeper than this fails the value (skip the
 * line), instead of exhausting the C stack on adversarial input. */
#define MAX_JSON_DEPTH 256

typedef struct { const char* p; const char* end; int err; int depth; } parser_t;

static void jval_free(jval_t* v) {
    size_t i;
    if (!v) return;
    free(v->str);
    for (i = 0; i < v->nitems; i++) jval_free(v->items[i]);
    free(v->items);
    for (i = 0; i < v->nfields; i++) { free(v->keys[i]); jval_free(v->vals[i]); }
    free(v->keys);
    free(v->vals);
    free(v);
}

static void skip_ws(parser_t* ps) {
    while (ps->p < ps->end && (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r')) ps->p++;
}

/* Append UTF-8 encoding of cp to buf (cap-checked). Returns bytes, or 0. */
static size_t utf8_emit(unsigned long cp, char* buf, size_t cap) {
    if (cp < 0x80) { if (cap < 1) return 0; buf[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        if (cap < 2) return 0;
        buf[0] = (char)(0xC0 | (cp >> 6)); buf[1] = (char)(0x80 | (cp & 0x3F)); return 2;
    }
    if (cp < 0x10000) {
        if (cap < 3) return 0;
        buf[0] = (char)(0xE0 | (cp >> 12)); buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    if (cp <= 0x10FFFF) {
        if (cap < 4) return 0;
        buf[0] = (char)(0xF0 | (cp >> 18)); buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[3] = (char)(0x80 | (cp & 0x3F)); return 4;
    }
    return 0;
}

static unsigned hexval(char c) {
    if (c >= '0' && c <= '9') return (unsigned)(c - '0');
    if (c >= 'a' && c <= 'f') return (unsigned)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (unsigned)(c - 'A' + 10);
    return 99;
}

/* Parse \uXXXX at *pp (after backslash-u consumed). Advances past digits. */
static unsigned long parse_hex4(parser_t* ps, const char** pp, int* ok) {
    unsigned long v = 0;
    int i;
    for (i = 0; i < 4; i++) {
        if (*pp >= ps->end) { *ok = 0; return 0; }
        unsigned h = hexval(**pp);
        if (h > 15) { *ok = 0; return 0; }
        v = (v << 4) | h;
        (*pp)++;
    }
    *ok = 1;
    return v;
}

static jval_t* jval_new(jtype_t t) {
    jval_t* v = (jval_t*)calloc(1, sizeof(jval_t));
    if (v) v->type = t;
    return v;
}

static jval_t* parse_value(parser_t* ps);

static jval_t* parse_string(parser_t* ps) {
    /* ps->p at opening quote */
    size_t cap = 64, len = 0;
    char* buf;
    jval_t* v;
    ps->p++; /* consume " */
    buf = (char*)malloc(cap);
    if (!buf) { ps->err = 1; return NULL; }
    for (;;) {
        char c;
        if (ps->p >= ps->end) { free(buf); ps->err = 1; return NULL; }
        c = *ps->p++;
        if (c == '"') break;
        if (c == '\\') {
            char e;
            if (ps->p >= ps->end) { free(buf); ps->err = 1; return NULL; }
            e = *ps->p++;
            if (e == '"' || e == '\\' || e == '/') c = e;
            else if (e == 'b') c = '\b';
            else if (e == 'f') c = '\f';
            else if (e == 'n') c = '\n';
            else if (e == 'r') c = '\r';
            else if (e == 't') c = '\t';
            else if (e == 'u') {
                int ok = 0;
                unsigned long cp = parse_hex4(ps, &ps->p, &ok);
                char tmp[4];
                size_t nb;
                if (!ok) { free(buf); ps->err = 1; return NULL; }
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    /* high surrogate: expect \uDC00-\uDFFF */
                    if (ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
                        const char* q = ps->p + 2;
                        unsigned long lo = parse_hex4(ps, &q, &ok);
                        if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            ps->p = q;
                        } else { free(buf); ps->err = 1; return NULL; }
                    } else { free(buf); ps->err = 1; return NULL; }
                }
                nb = utf8_emit(cp, tmp, sizeof(tmp));
                if (nb == 0) { free(buf); ps->err = 1; return NULL; }
                while (len + nb + 1 > cap) {
                    cap *= 2;
                    buf = (char*)realloc(buf, cap);
                    if (!buf) { ps->err = 1; return NULL; }
                }
                memcpy(buf + len, tmp, nb);
                len += nb;
                continue;
            } else { free(buf); ps->err = 1; return NULL; }
        }
        if (len + 2 > cap) {
            cap *= 2;
            buf = (char*)realloc(buf, cap);
            if (!buf) { ps->err = 1; return NULL; }
        }
        buf[len++] = c;
    }
    buf[len] = '\0';
    v = jval_new(J_STR);
    if (!v) { free(buf); ps->err = 1; return NULL; }
    v->str = buf;
    v->slen = len;
    return v;
}

static jval_t* parse_number(parser_t* ps) {
    const char* start = ps->p;
    char* e = NULL;
    double d;
    int is_int = 1;
    jval_t* v;
    if (ps->p < ps->end && (*ps->p == '-')) ps->p++;
    while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9') ps->p++;
    if (ps->p < ps->end && (*ps->p == '.' || *ps->p == 'e' || *ps->p == 'E')) is_int = 0;
    while (ps->p < ps->end && ((*ps->p >= '0' && *ps->p <= '9') || *ps->p == '.' ||
           *ps->p == 'e' || *ps->p == 'E' || *ps->p == '+' || *ps->p == '-')) ps->p++;
    d = strtod(start, &e);
    if (e != ps->p) { ps->err = 1; return NULL; }
    v = jval_new(J_NUM);
    if (!v) { ps->err = 1; return NULL; }
    v->num = d;
    v->num_is_int = is_int;
    return v;
}

static int parse_literal(parser_t* ps, const char* lit, jval_t** out, jtype_t t, int b) {
    size_t n = strlen(lit);
    if ((size_t)(ps->end - ps->p) < n || memcmp(ps->p, lit, n) != 0) { ps->err = 1; return 0; }
    ps->p += n;
    *out = jval_new(t);
    if (!*out) { ps->err = 1; return 0; }
    if (t == J_BOOL) (*out)->boolean = b;
    return 1;
}

static jval_t* parse_array(parser_t* ps) {
    jval_t* v = jval_new(J_ARR);
    size_t cap = 0;
    if (!v) { ps->err = 1; return NULL; }
    ps->p++; /* [ */
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') { ps->p++; return v; }
    for (;;) {
        jval_t* it;
        skip_ws(ps);
        it = parse_value(ps);
        if (ps->err || !it) { jval_free(v); return NULL; }
        if (v->nitems + 1 > cap) {
            size_t ncap = cap ? cap * 2 : 8;
            jval_t** ni = (jval_t**)realloc(v->items, ncap * sizeof(jval_t*));
            if (!ni) { jval_free(it); jval_free(v); ps->err = 1; return NULL; }
            v->items = ni;
            cap = ncap;
        }
        v->items[v->nitems++] = it;
        skip_ws(ps);
        if (ps->p >= ps->end) { jval_free(v); ps->err = 1; return NULL; }
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == ']') { ps->p++; return v; }
        jval_free(v);
        ps->err = 1;
        return NULL;
    }
}

static jval_t* parse_object(parser_t* ps) {
    jval_t* v = jval_new(J_OBJ);
    size_t cap = 0;
    if (!v) { ps->err = 1; return NULL; }
    ps->p++; /* { */
    skip_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') { ps->p++; return v; }
    for (;;) {
        jval_t* ks;
        char* key;
        jval_t* val;
        skip_ws(ps);
        if (ps->p >= ps->end || *ps->p != '"') { jval_free(v); ps->err = 1; return NULL; }
        ks = parse_string(ps);
        if (ps->err || !ks) { jval_free(v); return NULL; }
        key = ks->str;
        ks->str = NULL;
        jval_free(ks);
        skip_ws(ps);
        if (ps->p >= ps->end || *ps->p != ':') { free(key); jval_free(v); ps->err = 1; return NULL; }
        ps->p++;
        skip_ws(ps);
        val = parse_value(ps);
        if (ps->err || !val) { free(key); jval_free(v); return NULL; }
        if (v->nfields + 1 > cap) {
            size_t ncap = cap ? cap * 2 : 8;
            char** nk = (char**)realloc(v->keys, ncap * sizeof(char*));
            jval_t** nv;
            if (!nk) { free(key); jval_free(val); jval_free(v); ps->err = 1; return NULL; }
            v->keys = nk;
            nv = (jval_t**)realloc(v->vals, ncap * sizeof(jval_t*));
            if (!nv) { free(key); jval_free(val); jval_free(v); ps->err = 1; return NULL; }
            v->vals = nv;
            cap = ncap;
        }
        v->keys[v->nfields] = key;
        v->vals[v->nfields] = val;
        v->nfields++;
        skip_ws(ps);
        if (ps->p >= ps->end) { jval_free(v); ps->err = 1; return NULL; }
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; return v; }
        jval_free(v);
        ps->err = 1;
        return NULL;
    }
}

static jval_t* parse_value(parser_t* ps) {
    jval_t* out = NULL;
    jval_t* v = NULL;
    char c;
    skip_ws(ps);
    if (ps->p >= ps->end) { ps->err = 1; return NULL; }
    c = *ps->p;
    /* Depth counts nesting only (siblings must not accumulate). */
    if (c == '{' || c == '[') {
        if (ps->depth >= MAX_JSON_DEPTH) { ps->err = 1; return NULL; }
        ps->depth++;
        v = (c == '{') ? parse_object(ps) : parse_array(ps);
        ps->depth--;
        return v;
    }
    if (c == '"') return parse_string(ps);
    if (c == 't') { if (parse_literal(ps, "true", &out, J_BOOL, 1)) return out; return NULL; }
    if (c == 'f') { if (parse_literal(ps, "false", &out, J_BOOL, 0)) return out; return NULL; }
    if (c == 'n') { if (parse_literal(ps, "null", &out, J_NULL, 0)) return out; return NULL; }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(ps);
    ps->err = 1;
    return NULL;
}

static jval_t* obj_get(const jval_t* o, const char* key) {
    size_t i;
    if (!o || o->type != J_OBJ) return NULL;
    for (i = 0; i < o->nfields; i++)
        if (strcmp(o->keys[i], key) == 0) return o->vals[i];
    return NULL;
}

/* UTF-8 decode one code point; advances *pp. Returns 1 on valid. */
static int utf8_next(const char** pp, const char* end, unsigned long* cp) {
    const unsigned char* p = (const unsigned char*)*pp;
    if (p >= (const unsigned char*)end) return 0;
    if (*p < 0x80) { *cp = *p; *pp += 1; return 1; }
    if ((*p & 0xE0) == 0xC0 && p + 1 < (const unsigned char*)end) {
        *cp = ((unsigned long)(*p & 0x1F) << 6) | (p[1] & 0x3F);
        *pp += 2; return 1;
    }
    if ((*p & 0xF0) == 0xE0 && p + 2 < (const unsigned char*)end) {
        *cp = ((unsigned long)(*p & 0x0F) << 12) | ((unsigned long)(p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        *pp += 3; return 1;
    }
    if ((*p & 0xF8) == 0xF0 && p + 3 < (const unsigned char*)end) {
        *cp = ((unsigned long)(*p & 0x07) << 18) | ((unsigned long)(p[1] & 0x3F) << 12) |
              ((unsigned long)(p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        *pp += 4; return 1;
    }
    *cp = 0xFFFD;
    *pp += 1;
    return 1;
}

static int is_opc(unsigned long cp) {
    return cp == '+' || cp == '-' || cp == '*' || cp == '/' || cp == '^' || cp == '=';
}

static int is_wsc(unsigned long cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}

/* Mirrors distill_prm800k.py featurize_step (placeholder byte-level v1). */
static void featurize(const char* s, size_t nbytes, long step_idx, long n_steps, double* f) {
    const char* p = s;
    const char* end = s + nbytes;
    unsigned long cp;
    long ncp = 0, nwords = 0, ndigits = 0, nops = 0, nlatex = 0, nnums = 0;
    long max_depth = 0, depth = 0, nupper = 0, nspace = 0;
    long wlen = 0, wsum = 0;
    int in_word = 0, in_num = 0, has_q = 0;
    while (p < end) {
        const char* q = p;
        unsigned long c;
        (void)q;
        if (!utf8_next(&p, end, &c)) break;
        cp = c;
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
        /* rstrip ASCII whitespace for the terminator test (none of the
         * non-ASCII chars in scope are whitespace, so this matches). */
        const char* t = end;
        while (t > s && (t[-1] == ' ' || t[-1] == '\t' || t[-1] == '\n' || t[-1] == '\r' || t[-1] == '\f' || t[-1] == '\v')) t--;
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

typedef struct {
    double* X; size_t nx, cx;
    double* T; size_t nt, ct;
    long hist_neg, hist_zero, hist_pos;
    long skipped;
} corpus_t;

static void corpus_push(corpus_t* c, const double* f, double t) {
    if (c->nx + FEAT_DIM > c->cx) {
        size_t ncap = c->cx ? c->cx * 2 : 4096;
        double* nx = (double*)realloc(c->X, ncap * sizeof(double));
        if (!nx) { fprintf(stderr, "OOM\n"); exit(1); }
        c->X = nx;
        c->cx = ncap;
    }
    if (c->nt + 1 > c->ct) {
        size_t ncap = c->ct ? c->ct * 2 : 4096;
        double* nt = (double*)realloc(c->T, ncap * sizeof(double));
        if (!nt) { fprintf(stderr, "OOM\n"); exit(1); }
        c->T = nt;
        c->ct = ncap;
    }
    memcpy(c->X + c->nx, f, FEAT_DIM * sizeof(double));
    c->nx += FEAT_DIM;
    c->T[c->nt++] = t;
}

/* Process one JSONL line. Returns 1 if a step was consumed... actually
 * consumes all usable steps of the row; counts skips. */
static void process_line(const char* line, size_t len, corpus_t* c) {
    parser_t ps;
    jval_t* root;
    jval_t* label;
    jval_t* steps;
    size_t i;
    ps.p = line;
    ps.end = line + len;
    ps.err = 0;
    ps.depth = 0;
    root = parse_value(&ps);
    if (ps.err || !root) { jval_free(root); c->skipped++; return; }
    skip_ws(&ps);
    if (ps.p != ps.end) { jval_free(root); c->skipped++; return; }
    if (root->type != J_OBJ) { jval_free(root); c->skipped++; return; }
    label = obj_get(root, "label");
    steps = label ? obj_get(label, "steps") : NULL;
    if (!steps || steps->type != J_ARR) { jval_free(root); c->skipped++; return; }
    for (i = 0; i < steps->nitems; i++) {
        jval_t* st = steps->items[i];
        jval_t* comps;
        jval_t* chc;
        jval_t* comp;
        jval_t* rating;
        jval_t* text;
        double f[FEAT_DIM];
        double t;
        if (!st || st->type != J_OBJ) { c->skipped++; continue; }
        comps = obj_get(st, "completions");
        chc = obj_get(st, "chosen_completion");
        if (!comps || comps->type != J_ARR) { c->skipped++; continue; }
        if (!chc || chc->type != J_NUM || !chc->num_is_int) { c->skipped++; continue; }
        {
            long ci = (long)chc->num;
            if (ci < 0 || (size_t)ci >= comps->nitems) { c->skipped++; continue; }
            comp = comps->items[(size_t)ci];
        }
        if (!comp || comp->type != J_OBJ) { c->skipped++; continue; }
        rating = obj_get(comp, "rating");
        text = obj_get(comp, "text");
        if (!rating || rating->type != J_NUM) { c->skipped++; continue; }
        if (rating->num == -1.0) t = -1.0;
        else if (rating->num == 0.0) t = 0.0;
        else if (rating->num == 1.0) t = 1.0;
        else { c->skipped++; continue; }
        if (!text || text->type != J_STR) { c->skipped++; continue; }
        featurize(text->str, text->slen, (long)i, (long)steps->nitems, f);
        corpus_push(c, f, t);
        if (t < 0.0) c->hist_neg++;
        else if (t > 0.0) c->hist_pos++;
        else c->hist_zero++;
    }
    jval_free(root);
}

/* ---- sha256 (public domain style, self-contained) ---- */
typedef struct { uint32_t h[8]; uint64_t len; unsigned char buf[64]; size_t n; } sha256_t;
static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
static void sha256_block(sha256_t* s, const unsigned char* p) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64], a, b, cc, d, e, f, g, h;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4*i] << 24) | ((uint32_t)p[4*i+1] << 16) | ((uint32_t)p[4*i+2] << 8) | p[4*i+3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a=s->h[0]; b=s->h[1]; cc=s->h[2]; d=s->h[3]; e=s->h[4]; f=s->h[5]; g=s->h[6]; h=s->h[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25);
        uint32_t ch = (e&f)^(~e&g);
        uint32_t t1 = h+S1+ch+K[i]+w[i];
        uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22);
        uint32_t mj = (a&b)^(a&cc)^(b&cc);
        uint32_t t2 = S0+mj;
        h=g; g=f; f=e; e=d+t1; d=cc; cc=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=cc; s->h[3]+=d; s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}
static void sha256_init(sha256_t* s) {
    s->h[0]=0x6a09e667; s->h[1]=0xbb67ae85; s->h[2]=0x3c6ef372; s->h[3]=0xa54ff53a;
    s->h[4]=0x510e527f; s->h[5]=0x9b05688c; s->h[6]=0x1f83d9ab; s->h[7]=0x5be0cd19;
    s->len = 0; s->n = 0;
}
static void sha256_update(sha256_t* s, const void* data, size_t n) {
    const unsigned char* p = (const unsigned char*)data;
    s->len += n;
    while (n) {
        size_t take = 64 - s->n;
        if (take > n) take = n;
        memcpy(s->buf + s->n, p, take);
        s->n += take; p += take; n -= take;
        if (s->n == 64) { sha256_block(s, s->buf); s->n = 0; }
    }
}
static void sha256_final(sha256_t* s, char out_hex[65]) {
    uint64_t bits = s->len * 8;
    unsigned char pad = 0x80, zero = 0;
    int i;
    sha256_update(s, &pad, 1);
    while (s->n != 56) sha256_update(s, &zero, 1);
    {
        unsigned char lb[8];
        for (i = 0; i < 8; i++) lb[i] = (unsigned char)(bits >> (56 - 8 * i));
        sha256_update(s, lb, 8);
    }
    for (i = 0; i < 8; i++) sprintf(out_hex + 8 * i, "%08x", s->h[i]);
    out_hex[64] = '\0';
}

/* ---- selftest: embedded parser/featurizer proofs, no data needed ---- */
static int selftest_fail = 0;
#define ST_CHECK(cond, msg) do { \
    if (!(cond)) { printf("  ❌ SELFTEST FAIL: %s\n", msg); selftest_fail = 1; } \
} while (0)
/* Transcendentals: GCC constant-folds log1p(k) at compile time (MPFR,
 * correctly rounded) while runtime libm may round the last ulp
 * differently — same 1-ulp phenomenon as Python-vs-C constant comparison.
 * Production features always call log1p on variables (runtime libm, exactly
 * like CPython's math.log1p), so .bin bytes are unaffected; only the
 * selftest's constant expectations need a 1-ulp-tolerant compare. */
static int close_ulp(double a, double b) { return fabs(a - b) <= 1e-15; }

static jval_t* parse_doc(const char* s) {
    parser_t ps;
    jval_t* v;
    ps.p = s;
    ps.end = s + strlen(s);
    ps.err = 0;
    ps.depth = 0;
    v = parse_value(&ps);
    if (ps.err) { jval_free(v); return NULL; }
    skip_ws(&ps);
    if (ps.p != ps.end) { jval_free(v); return NULL; }
    return v;
}

/* code-point length of a UTF-8 buffer */
static long cp_len(const char* s, size_t n) {
    const char* p = s;
    const char* end = s + n;
    unsigned long cp;
    long c = 0;
    (void)cp;
    while (p < end) { unsigned long d; if (!utf8_next(&p, end, &d)) break; c++; }
    return c;
}

static int run_selftest(void) {
    printf("distill_prm800k --selftest: parser/featurizer proofs\n");
    /* 1. escapes incl \u00e9: a " b \ newline é = 7 code points, 8 UTF-8 bytes */
    {
        jval_t* v = parse_doc("\"a\\\"b\\\\c\\n\\u00e9\"");
        ST_CHECK(v && v->type == J_STR, "escape doc parses");
        if (v && v->type == J_STR) {
            ST_CHECK(v->slen == 8, "escaped UTF-8 byte length == 8");
            ST_CHECK(cp_len(v->str, v->slen) == 7, "escaped code-point length == 7");
        }
        jval_free(v);
    }
    /* 2. surrogate pair -> single astral code point */
    {
        jval_t* v = parse_doc("\"\\uD83D\\uDE00\"");
        ST_CHECK(v && v->type == J_STR, "surrogate doc parses");
        if (v && v->type == J_STR) {
            ST_CHECK(v->slen == 4, "astral UTF-8 byte length == 4");
            ST_CHECK(cp_len(v->str, v->slen) == 1, "astral code-point length == 1");
        }
        jval_free(v);
    }
    /* 3. rating forms: 1/1.0/-1/0 accepted; 2/1.5/"1"/true/null rejected */
    {
        const char* good[] = {"1", "1.0", "-1", "-1.0", "0", "0.0"};
        const char* bad[] = {"2", "1.5", "\"1\"", "true", "null", "1e1"};
        size_t i;
        for (i = 0; i < sizeof(good) / sizeof(good[0]); i++) {
            jval_t* v = parse_doc(good[i]);
            int ok = v && v->type == J_NUM && (v->num == -1.0 || v->num == 0.0 || v->num == 1.0);
            char msg[64];
            sprintf(msg, "rating %s accepted", good[i]);
            ST_CHECK(ok, msg);
            jval_free(v);
        }
        for (i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            jval_t* v = parse_doc(bad[i]);
            int ok = v && v->type == J_NUM && (v->num == -1.0 || v->num == 0.0 || v->num == 1.0);
            char msg[64];
            sprintf(msg, "rating %s rejected", bad[i]);
            ST_CHECK(!ok, msg);
            jval_free(v);
        }
    }
    /* 4. navigation skips: null choice, oob choice, missing text, bad rating */
    {
        corpus_t c;
        memset(&c, 0, sizeof(c));
        /* NOTE: lengths via strlen (a past revision hardcoded them wrong and
         * the selftest caught it — the gate works). */
        #define TLINE(s) s, strlen(s)
        process_line(TLINE("{\"label\":{\"steps\":[{\"completions\":[{\"text\":\"x\",\"rating\":1}],\"chosen_completion\":null}]}}"), &c);
        ST_CHECK(c.nt == 0 && c.skipped == 1, "null choice skipped");
        process_line(TLINE("{\"label\":{\"steps\":[{\"completions\":[{\"text\":\"x\",\"rating\":1}],\"chosen_completion\":7}]}}"), &c);
        ST_CHECK(c.nt == 0 && c.skipped == 2, "oob choice skipped");
        process_line(TLINE("{\"label\":{\"steps\":[{\"completions\":[{\"rating\":1}],\"chosen_completion\":0}]}}"), &c);
        ST_CHECK(c.nt == 0 && c.skipped == 3, "missing text skipped");
        process_line(TLINE("{\"label\":{\"steps\":[{\"completions\":[{\"text\":\"x\",\"rating\":5}],\"chosen_completion\":0}]}}"), &c);
        ST_CHECK(c.nt == 0 && c.skipped == 4, "bad rating skipped");
        process_line(TLINE("{\"label\":{\"steps\":[{\"completions\":[{\"text\":\"ok 1+1=2\",\"rating\":-1}],\"chosen_completion\":0}]}}"), &c);
        ST_CHECK(c.nt == 1 && c.hist_neg == 1, "good row consumed");
        process_line(TLINE("{\"label\": {\"steps\": ["), &c);
        ST_CHECK(c.nt == 1 && c.skipped == 5, "truncated line skipped, no crash");
        process_line(TLINE("not json at all"), &c);
        ST_CHECK(c.nt == 1 && c.skipped == 6, "garbage line skipped, no crash");
        #undef TLINE
        free(c.X); free(c.T);
    }
    /* 5. feature assertion: "Solve 2+2=4 (easy)?", step 0 of 1 */
    {
        double f[FEAT_DIM];
        const char* s = "Solve 2+2=4 (easy)?";
        featurize(s, strlen(s), 0, 1, f);
        ST_CHECK(close_ulp(f[0], log1p(19.0)), "feat len");
        ST_CHECK(close_ulp(f[1], log1p(3.0)), "feat words");
        ST_CHECK(close_ulp(f[2], log1p(3.0)), "feat digits");
        ST_CHECK(close_ulp(f[3], log1p(2.0)), "feat ops");
        ST_CHECK(f[4] == 1.0, "feat depth");
        ST_CHECK(f[5] == 0.0 && f[6] == 1.0 && f[7] == 1.0, "feat position");
        ST_CHECK(close_ulp(f[8], log1p(0.0)), "feat latex");
        ST_CHECK(close_ulp(f[9], log1p(3.0)), "feat numbers");
        ST_CHECK(fabs(f[10] - 17.0 / 3.0) < 1e-12, "feat avgwlen");
        ST_CHECK(f[11] == 1.0 && f[12] == 1.0, "feat terminator/question");
        ST_CHECK(close_ulp(f[13], log1p(1.0)), "feat upper");
        ST_CHECK(close_ulp(f[14], log1p(2.0)), "feat space");
        ST_CHECK(f[15] == 1.0, "feat last");
    }
    if (selftest_fail) { printf("--selftest: FAILURES\n"); return 1; }
    printf("--selftest: ALL PROOFS HOLD\n");
    return 0;
}

/* ---- driver ---- */
static const char* data_dir(void) {
    const char* e = getenv("LANCIUS_DATA_DIR");
    return e ? e : "data_text";
}

static int distill_split(const char* split, const char* out_dir) {
    char src[1024];
    FILE* fh;
    char* data;
    long fsize;
    size_t len;
    char* line;
    corpus_t c;
    char xp[1024], tp[1024], mp[1024];
    FILE* fo;
    sha256_t sh;
    char hx[65], ht[65];
    size_t hist_tot;
    snprintf(src, sizeof(src), "%s/prm800k_phase1_%s.jsonl", data_dir(), split);
    fh = fopen(src, "rb");
    if (!fh) { printf("  ❌ cannot open %s\n", src); return 1; }
    fseek(fh, 0, SEEK_END);
    fsize = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    if (fsize < 0 || (size_t)fsize > MAX_FILE_BYTES) { fclose(fh); printf("  ❌ bad size %s\n", src); return 1; }
    data = (char*)malloc((size_t)fsize + 1);
    if (!data) { fclose(fh); return 1; }
    len = fread(data, 1, (size_t)fsize, fh);
    fclose(fh);
    memset(&c, 0, sizeof(c));
    line = data;
    {
        char* cur = data;
        char* end = data + len;
        while (cur < end) {
            char* nl = memchr(cur, '\n', (size_t)(end - cur));
            size_t llen = nl ? (size_t)(nl - cur) : (size_t)(end - cur);
            while (llen && (cur[llen - 1] == '\r' || cur[llen - 1] == ' ' || cur[llen - 1] == '\t')) llen--;
            while (llen && (*cur == ' ' || *cur == '\t')) { cur++; llen--; }
            if (llen) process_line(cur, llen, &c);
            cur = nl ? nl + 1 : end;
        }
    }
    free(data);
    (void)line;
    snprintf(xp, sizeof(xp), "%s/%s.X.bin", out_dir, split);
    snprintf(tp, sizeof(tp), "%s/%s.T.bin", out_dir, split);
    snprintf(mp, sizeof(mp), "%s/%s.meta.json", out_dir, split);
    {
        char cmd[1152];
        snprintf(cmd, sizeof(cmd), "mkdir -p %s", out_dir);
        if (system(cmd) != 0) { printf("  ❌ cannot create %s\n", out_dir); return 1; }
    }
    fo = fopen(xp, "wb");
    if (!fo) { printf("  ❌ cannot write %s\n", xp); return 1; }
    if (c.nx) fwrite(c.X, sizeof(double), c.nx, fo);
    fclose(fo);
    fo = fopen(tp, "wb");
    if (!fo) { printf("  ❌ cannot write %s\n", tp); return 1; }
    if (c.nt) fwrite(c.T, sizeof(double), c.nt, fo);
    fclose(fo);
    sha256_init(&sh);
    if (c.nx) sha256_update(&sh, c.X, c.nx * sizeof(double));
    {
        sha256_t s2;
        char* rawx;
        /* recompute per-file hashes over exact byte streams */
        FILE* fx = fopen(xp, "rb");
        fseek(fx, 0, SEEK_END);
        long xs = ftell(fx);
        fseek(fx, 0, SEEK_SET);
        rawx = (char*)malloc(xs > 0 ? (size_t)xs : 1);
        if (xs > 0) {
            if (fread(rawx, 1, (size_t)xs, fx) != (size_t)xs) { fclose(fx); free(rawx); printf("  ❌ reread\n"); return 1; }
        }
        fclose(fx);
        sha256_init(&s2);
        if (xs > 0) sha256_update(&s2, rawx, (size_t)xs);
        sha256_final(&s2, hx);
        free(rawx);
    }
    {
        sha256_t s3;
        FILE* ft = fopen(tp, "rb");
        fseek(ft, 0, SEEK_END);
        long ts = ftell(ft);
        fseek(ft, 0, SEEK_SET);
        char* rawt = (char*)malloc(ts > 0 ? (size_t)ts : 1);
        if (ts > 0) {
            if (fread(rawt, 1, (size_t)ts, ft) != (size_t)ts) { fclose(ft); free(rawt); printf("  ❌ reread\n"); return 1; }
        }
        fclose(ft);
        sha256_init(&s3);
        if (ts > 0) sha256_update(&s3, rawt, (size_t)ts);
        sha256_final(&s3, ht);
        free(rawt);
    }
    {
        FILE* fm = fopen(mp, "w");
        if (!fm) { printf("  ❌ cannot write %s\n", mp); return 1; }
        fprintf(fm, "{\n  \"split\": \"%s\",\n  \"n_steps\": %lu,\n  \"feat_dim\": %d,\n",
                split, (unsigned long)c.nt, FEAT_DIM);
        fprintf(fm, "  \"featurization\": \"placeholder-byte-level-v1 (see R2-2)\",\n");
        fprintf(fm, "  \"rating_histogram\": {\"-1\": %ld, \"0\": %ld, \"1\": %ld},\n",
                c.hist_neg, c.hist_zero, c.hist_pos);
        fprintf(fm, "  \"skipped\": %ld,\n  \"sha256_X\": \"%s\",\n  \"sha256_T\": \"%s\"\n}\n",
                c.skipped, hx, ht);
        fclose(fm);
    }
    printf("  [%s] steps=%lu skipped=%ld\n", split, (unsigned long)c.nt, c.skipped);
    printf("    ratings(-1/0/+1): {%ld, %ld, %ld}\n", c.hist_neg, c.hist_zero, c.hist_pos);
    hist_tot = (size_t)((c.hist_neg > 0) + (c.hist_zero > 0) + (c.hist_pos > 0));
    if (hist_tot < 2)
        printf("    ⚠️  single-class split: fine for pipeline proof, "
               "NOT for training a real verifier (needs -1/0 labels).\n");
    printf("    wrote %s + %s + %s\n", xp, tp, mp);
    free(c.X);
    free(c.T);
    return 0;
}

int main(int argc, char** argv) {
    const char* splits[2] = {"train", "test"};
    int nsplits = 2;
    const char* out_dir = "data_vec";
    int i, rc = 0;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--selftest") == 0) return run_selftest();
        if (strncmp(argv[i], "--out=", 6) == 0) { out_dir = argv[i] + 6; continue; }
        if (strcmp(argv[i], "train") == 0) { splits[0] = "train"; nsplits = 1; continue; }
        if (strcmp(argv[i], "test") == 0) { splits[0] = "test"; nsplits = 1; continue; }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("usage: distill_prm800k [--out=DIR] [train|test] [--selftest]\n");
            printf("C port of distill_prm800k.py (retired): PRM800k rows -> numeric vectors.\n");
            return 0;
        }
        printf("unknown arg: %s\n", argv[i]);
        return 1;
    }
    printf("PRM800k step distillation, C (placeholder byte-level encoding)\n");
    for (i = 0; i < nsplits; i++)
        if (distill_split(splits[i], out_dir) != 0) rc = 1;
    return rc;
}
