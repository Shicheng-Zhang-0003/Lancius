#include "lancius/lancius_stable_api.h"
#include "lancius/lancius_checked.h"
#include "lancius/lancius_ir.h" /* for LANCIUS_MAX_TENSOR_ELEMS */
#include "lancius/lancius_scheduler.h"
#include "lancius/lancius_arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

// Thread-Local Error State (Production Standard)
static _Thread_local lancius_status g_last_error = LANCIUS_OK;

static void set_error(lancius_status err) {
    g_last_error = err;
}

static lancius_status map_internal_error(lancius_error err) {
    switch (err) {
        case LANCIUS_ERROR_OK:               return LANCIUS_OK;
        case LANCIUS_ERROR_OOM:              return LANCIUS_ERR_OOM;
        case LANCIUS_ERROR_NULL_PTR:         return LANCIUS_ERR_NULL_PTR;
        case LANCIUS_ERROR_SHAPE_MISMATCH:   return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_UNSUPPORTED_OP:   return LANCIUS_ERR_UNSUPPORTED_OP;
        /* v12R1 fix: dtype failures were collapsed into UNSUPPORTED_OP,
         * making misconfigured quantizers indistinguishable from bad graphs. */
        case LANCIUS_ERROR_UNSUPPORTED_DTYPE:return LANCIUS_ERR_UNSUPPORTED_DTYPE;
        case LANCIUS_ERROR_INVALID_DTYPE:    return LANCIUS_ERR_UNSUPPORTED_DTYPE;
        case LANCIUS_ERROR_INVALID_MODEL:    return LANCIUS_ERR_IO;
        case LANCIUS_ERROR_VERSION_MISMATCH: return LANCIUS_ERR_IO;
        case LANCIUS_ERROR_IO:               return LANCIUS_ERR_IO;
        case LANCIUS_ERROR_GRAPH_CYCLE:      return LANCIUS_ERR_GRAPH_CYCLE;
        case LANCIUS_ERROR_GRAPH_INVALID:    return LANCIUS_ERR_UNSUPPORTED_OP;
        case LANCIUS_ERROR_OVERFLOW:         return LANCIUS_ERR_OVERFLOW;
        case LANCIUS_ERROR_LIMIT:            return LANCIUS_ERR_OVERFLOW;
        case LANCIUS_ERROR_INTERNAL:         return LANCIUS_ERR_UNSUPPORTED_OP;
        case LANCIUS_ERROR_INVALID_HANDLE:   return LANCIUS_ERR_INVALID_HANDLE;
        case LANCIUS_ERROR_LIFETIME:         return LANCIUS_ERR_INVALID_HANDLE;
        case LANCIUS_ERROR_INVALID_RANK:     return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_INVALID_SHAPE:    return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_INVALID_STRIDE:   return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_INVALID_PERMUTATION: return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_RESHAPE_MISMATCH: return LANCIUS_ERR_SHAPE_MISMATCH;
        case LANCIUS_ERROR_NUMERICAL:        return LANCIUS_ERR_NUMERICAL;
        default:                             return LANCIUS_ERR_UNSUPPORTED_OP;
    }
}

static void sync_internal_error(void) {
    lancius_error err = lancius_get_error();
    if (err != LANCIUS_ERROR_OK) {
        set_error(map_internal_error(err));
        lancius_clear_error();
    }
}


LANCIUS_EXPORT lancius_status lancius_get_last_error(void) {
    return g_last_error;
}

LANCIUS_EXPORT const char* lancius_get_error_string(lancius_status err) {
    switch(err) {
        case LANCIUS_OK: return "Success";
        case LANCIUS_ERR_OOM: return "Out of Memory";
        case LANCIUS_ERR_SHAPE_MISMATCH: return "Shape Mismatch";
        case LANCIUS_ERR_NULL_PTR: return "Null Pointer";
        case LANCIUS_ERR_UNSUPPORTED_OP: return "Unsupported Operation";
        case LANCIUS_ERR_BUFFER_TOO_SMALL: return "Buffer Too Small";
        case LANCIUS_ERR_IO: return "I/O Error";
        case LANCIUS_ERR_GRAPH_CYCLE: return "Graph Cycle";
        case LANCIUS_ERR_OVERFLOW: return "Overflow/Limit";
        case LANCIUS_ERR_NUMERICAL: return "Numerical Error";
        case LANCIUS_ERR_INVALID_HANDLE: return "Invalid Handle/Lifetime";
        case LANCIUS_ERR_UNSUPPORTED_DTYPE: return "Unsupported/Invalid Dtype";
default: return "Unknown Error";
    }
}


/* ============================================================================
 * Generation-counted handle table (fixes stale-handle use-after-free).
 *
 * Instead of returning direct pointers, we return encoded 64-bit handles
 * that index into a global table. Each table entry has a generation counter
 * that increments on destroy; stale handles fail because the generation
 * no longer matches. This eliminates use-after-free without ASan.
 * ============================================================================ */

#define HANDLE_TABLE_INIT_CAP 256
#define HANDLE_TYPE_CTX   0x01
#define HANDLE_TYPE_GRAPH 0x02
#define HANDLE_TYPE_TENSOR 0x03

/* 64-bit handle layout:
 *   bits 63..40  index       (24 bits -> at most 16,777,215 live/reused rows)
 *   bits 39..8   generation  (32 bits -> 4,294,967,295 registrations)
 *   bits 7..0    type        (8 bits)
 * The generation field is changed on every registration of a row, so a
 * handle from a destroyed object can never match a later object that reuses
 * the same row.  The index field limits the table to 16M rows, far beyond
 * any real graph; registration fails closed (returns 0) if that is reached. */
#define HANDLE_IDX_MASK  0xFFFFFFu
#define HANDLE_GEN_MASK  0xFFFFFFFFu
#define HANDLE_TYPE_MASK 0xFFu
#define HANDLE_FREE_NONE UINT32_MAX

typedef struct {
    void* ptr;
    uint32_t gen;
    uint32_t next_free;
    uint8_t type;   /* HANDLE_TYPE_* */
} handle_entry_t;

typedef struct {
    handle_entry_t* entries;
    size_t cap;
    size_t count;      /* high-water mark of rows ever used */
    uint32_t free_head;/* index of first reusable row, or HANDLE_FREE_NONE */
    uint32_t next_gen;
} handle_table_t;

static handle_table_t g_handle_table = {0};
static pthread_mutex_t g_handle_lock = PTHREAD_MUTEX_INITIALIZER;

/* Caller must hold g_handle_lock. */
static int handle_table_ensure_locked(void) {
    if (g_handle_table.entries != NULL) return 1;
    g_handle_table.cap = HANDLE_TABLE_INIT_CAP;
    g_handle_table.entries = calloc(g_handle_table.cap, sizeof(handle_entry_t));
    if (!g_handle_table.entries) { g_handle_table.cap = 0; return 0; }
    g_handle_table.count = 0;
    g_handle_table.free_head = HANDLE_FREE_NONE;
    g_handle_table.next_gen = 1;
    return 1;
}

/* Caller must hold g_handle_lock. */
static int handle_table_grow_locked(void) {
    if (g_handle_table.cap >= (size_t)HANDLE_IDX_MASK + 1u) return 0;
    size_t new_cap = g_handle_table.cap ? g_handle_table.cap * 2 : HANDLE_TABLE_INIT_CAP;
    if (new_cap > (size_t)HANDLE_IDX_MASK + 1u) new_cap = (size_t)HANDLE_IDX_MASK + 1u;
    handle_entry_t* new_entries = realloc(g_handle_table.entries, new_cap * sizeof(handle_entry_t));
    if (!new_entries) return 0;
    memset(new_entries + g_handle_table.cap, 0, (new_cap - g_handle_table.cap) * sizeof(handle_entry_t));
    g_handle_table.entries = new_entries;
    g_handle_table.cap = new_cap;
    return 1;
}

static uint64_t handle_encode(size_t idx, uint32_t gen, uint8_t type) {
    return ((uint64_t)(idx & HANDLE_IDX_MASK) << 40) |
           ((uint64_t)(gen & HANDLE_GEN_MASK) << 8) |
           (uint64_t)(type & HANDLE_TYPE_MASK);
}

static int handle_decode(uint64_t h, size_t* out_idx, uint32_t* out_gen, uint8_t* out_type) {
    if (h == 0) return 0;
    *out_idx = (size_t)((h >> 40) & HANDLE_IDX_MASK);
    *out_gen = (uint32_t)((h >> 8) & HANDLE_GEN_MASK);
    *out_type = (uint8_t)(h & HANDLE_TYPE_MASK);
    return 1;
}

static int handle_validate(uint64_t h, uint8_t expected_type, void** out_ptr) {
    if (h == 0) return 0;
    size_t idx; uint32_t gen; uint8_t type;
    if (!handle_decode(h, &idx, &gen, &type)) return 0;
    if (type != expected_type) return 0;
    int ok = 0;
    pthread_mutex_lock(&g_handle_lock);
    if (g_handle_table.entries != NULL && idx < g_handle_table.count) {
        handle_entry_t* e = &g_handle_table.entries[idx];
        if (e->ptr != NULL && e->gen == gen && e->type == expected_type) {
            if (out_ptr) *out_ptr = e->ptr;
            ok = 1;
        }
    }
    pthread_mutex_unlock(&g_handle_lock);
    return ok;
}

static uint64_t handle_register(void* ptr, uint8_t type) {
    if (ptr == NULL) return 0;
    uint64_t h = 0;
    pthread_mutex_lock(&g_handle_lock);
    if (!handle_table_ensure_locked()) goto out;
    size_t idx;
    if (g_handle_table.free_head != HANDLE_FREE_NONE) {
        idx = g_handle_table.free_head;
        g_handle_table.free_head = g_handle_table.entries[idx].next_free;
    } else {
        if (g_handle_table.count >= (size_t)HANDLE_IDX_MASK + 1u) goto out;
        if (g_handle_table.count >= g_handle_table.cap && !handle_table_grow_locked()) goto out;
        idx = g_handle_table.count++;
    }
    uint32_t gen = g_handle_table.next_gen++;
    if (g_handle_table.next_gen == 0) g_handle_table.next_gen = 1;
    handle_entry_t* e = &g_handle_table.entries[idx];
    e->ptr = ptr;
    e->gen = gen;
    e->next_free = HANDLE_FREE_NONE;
    e->type = type;
    h = handle_encode(idx, gen, type);
out:
    pthread_mutex_unlock(&g_handle_lock);
    return h;
}

static void handle_unregister(uint64_t h, uint8_t expected_type) {
    size_t idx; uint32_t gen; uint8_t type;
    if (!handle_decode(h, &idx, &gen, &type)) return;
    if (type != expected_type) return;
    pthread_mutex_lock(&g_handle_lock);
    if (g_handle_table.entries != NULL && idx < g_handle_table.count) {
        handle_entry_t* e = &g_handle_table.entries[idx];
        if (e->ptr != NULL && e->gen == gen && e->type == expected_type) {
            e->ptr = NULL;
            e->gen = 0;
            e->type = 0;
            e->next_free = g_handle_table.free_head;
            g_handle_table.free_head = (uint32_t)idx;
        }
    }
    pthread_mutex_unlock(&g_handle_lock);
}

/* Diagnostic used only by the lifetime regression audit.  It reports the
 * number of rows currently holding a live object (not the high-water mark),
 * so a create/destroy loop can prove that rows are recycled. */
size_t lancius_debug_live_handle_count(void) {
    size_t live = 0;
    pthread_mutex_lock(&g_handle_lock);
    for (size_t i = 0; i < g_handle_table.count; i++)
        if (g_handle_table.entries[i].ptr != NULL) live++;
    pthread_mutex_unlock(&g_handle_lock);
    return live;
}

/* ============================================================================
 * Internal types
 * ============================================================================ */

typedef struct {
    uint32_t magic;
    uint64_t cookie;
    lancius_arena* arena;
} lancius_context_internal;

typedef struct lancius_tensor_internal lancius_tensor_internal;
typedef struct {
    uint32_t magic;
    uint64_t cookie;
    lancius_graph* g;
    lancius_arena* scratch;
    lancius_schedule* sched;
    lancius_tensor_internal** tensors;
    size_t ntensors;
    size_t ctensors;
} lancius_graph_internal;

struct lancius_tensor_internal {
    uint32_t magic;
    uint64_t cookie;
    lancius_node* node;
    lancius_graph_internal* owner;
    lancius_tensor_handle handle;
};

/* Per-process secret for handle cookies.  pthread_once makes the lazy
 * initialisation safe when independent graphs are created concurrently. */
static uint64_t lancius_handle_secret = 0;
static pthread_once_t lancius_handle_secret_once = PTHREAD_ONCE_INIT;

static void lancius_handle_secret_init_once(void) {
    uint64_t t = (uint64_t)time(NULL);
    uint64_t p = (uint64_t)getpid();
    uint64_t a = (uint64_t)(uintptr_t)&lancius_handle_secret;
    uint64_t x = t ^ (p * 0x9E3779B97F4A7C15ULL) ^ (a >> 3);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    if (x == 0) x = 0xA5A5A5A5A5A5A5A5ULL;
    lancius_handle_secret = x;
}

static uint64_t lancius_handle_secret_init(void) {
    (void)pthread_once(&lancius_handle_secret_once, lancius_handle_secret_init_once);
    return lancius_handle_secret;
}

/* Internal validation functions */
static int ctx_valid_internal(const lancius_context_internal* c) {
    return c && c->magic == 0xC7A9C7A9u && c->cookie == lancius_handle_secret_init();
}

static int graph_valid_internal(const lancius_graph_internal* w) {
    return w && w->magic == 0x6A9A6A9Au && w->cookie == lancius_handle_secret_init();
}

static int tensor_valid_internal(const lancius_tensor_internal* w) {
    return w && w->magic == 0x7E5507E5u && w->cookie == lancius_handle_secret_init();
}

/* Public validation functions (use handle table + internal checks) */
static int ctx_valid(lancius_context ctx) {
    void* ptr; return handle_validate(ctx, HANDLE_TYPE_CTX, &ptr) && ctx_valid_internal(ptr);
}

static int graph_valid(lancius_graph_handle g) {
    void* ptr; return handle_validate(g, HANDLE_TYPE_GRAPH, &ptr) && graph_valid_internal(ptr);
}

static int tensor_valid(lancius_tensor_handle t) {
    void* ptr; return handle_validate(t, HANDLE_TYPE_TENSOR, &ptr) && tensor_valid_internal(ptr);
}

static uint64_t __attribute__((used)) handle_register_tensor(lancius_tensor_internal* t) {
    return handle_register(t, HANDLE_TYPE_TENSOR);
}

static void __attribute__((used)) handle_unregister_tensor(lancius_tensor_handle t) {
    handle_unregister(t, HANDLE_TYPE_TENSOR);
}

/* External audit V8: tensor handles were raw node* (any forged pointer
 * derefed). Wrap with magic; wrappers are borrowed (die with the graph).
 * NOTE: the graph must outlive its tensors (documented contract); wrapper
 * magic catches forged/cross-type handles, graph-destroy invalidates all. */
static uint64_t wrap_tensor(lancius_graph_internal* w, lancius_node* n) {
    if (!w || !n) return 0;
    if (w->ntensors >= w->ctensors) {
        size_t nc = w->ctensors ? w->ctensors * 2 : 16;
        lancius_tensor_internal** nl = (lancius_tensor_internal**)realloc(w->tensors, nc * sizeof(*nl));
        if (!nl) { set_error(LANCIUS_ERR_OOM); return 0; }
        w->tensors = nl;
        w->ctensors = nc;
    }
    lancius_tensor_internal* t = (lancius_tensor_internal*)malloc(sizeof(*t));
    if (!t) { set_error(LANCIUS_ERR_OOM); return 0; }
    t->magic = 0x7E5507E5u;
    t->cookie = lancius_handle_secret_init();
    t->node = n;
    t->owner = w;
    t->handle = handle_register_tensor(t);
    if (t->handle == 0) {
        free(t);
        set_error(LANCIUS_ERR_OOM);
        return 0;
    }
    w->tensors[w->ntensors++] = t;
    return t->handle;
}

static lancius_node* unwrap_tensor(lancius_tensor_handle t) {
    void* ptr;
    if (!handle_validate(t, HANDLE_TYPE_TENSOR, &ptr)) return 0;
    return ((lancius_tensor_internal*)ptr)->node;
}

/* Despot V9: cross-graph tensor reuse previously unchecked (wrapper held no
 * owner). A tensor from graph A passed into graph B derefed a foreign node.
 * Owner is now recorded at wrap time; callers must pass it for validation. */
static int tensor_owned_by(lancius_tensor_handle t, lancius_graph_internal* w) {
    void* ptr;
    if (!handle_validate(t, HANDLE_TYPE_TENSOR, &ptr)) return 0;
    return ((lancius_tensor_internal*)ptr)->owner == w;
}

LANCIUS_EXPORT lancius_context lancius_create_context(void) {
    set_error(LANCIUS_OK);
    lancius_context_internal* ctx = (lancius_context_internal*)malloc(sizeof(lancius_context_internal));
    if (!ctx) { set_error(LANCIUS_ERR_OOM); return 0; }
    ctx->magic = 0xC7A9C7A9u;
    ctx->cookie = lancius_handle_secret_init();
    ctx->arena = lancius_arena_create(64 * 1024 * 1024);
    if (!ctx->arena) { free(ctx); set_error(LANCIUS_ERR_OOM); return 0; }
    lancius_context h = handle_register(ctx, HANDLE_TYPE_CTX);
    if (h == 0) {
        lancius_arena_destroy(ctx->arena);
        free(ctx);
        set_error(LANCIUS_ERR_OOM);
        return 0;
    }
    set_error(LANCIUS_OK);
    return h;
}

LANCIUS_EXPORT void lancius_destroy_context(lancius_context ctx) {
    set_error(LANCIUS_OK);
    if (!ctx) return;
    void* ptr;
    if (!handle_validate(ctx, HANDLE_TYPE_CTX, &ptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return; }
    lancius_context_internal* internal = (lancius_context_internal*)ptr;
    if (!ctx_valid_internal(internal)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return; }
    internal->magic = 0;
    internal->cookie = 0;
    if (internal->arena) lancius_arena_destroy(internal->arena);
    handle_unregister(ctx, HANDLE_TYPE_CTX);
    free(internal);
}

LANCIUS_EXPORT lancius_graph_handle lancius_graph_create_stable(lancius_context ctx) {
    set_error(LANCIUS_OK);
    if (!ctx_valid(ctx)) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }

    lancius_graph_internal* wrapper = (lancius_graph_internal*)malloc(sizeof(lancius_graph_internal));
    if (!wrapper) { set_error(LANCIUS_ERR_OOM); return 0; }

    wrapper->g = lancius_graph_create();
    if (!wrapper->g) { free(wrapper); sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_OOM); return 0; }

    wrapper->scratch = lancius_arena_create(16 * 1024 * 1024);
    if (!wrapper->scratch) { lancius_graph_destroy(wrapper->g); free(wrapper); set_error(LANCIUS_ERR_OOM); return 0; }
    wrapper->sched = NULL;
    wrapper->magic = 0x6A9A6A9Au;
    wrapper->cookie = lancius_handle_secret_init();
    wrapper->tensors = NULL;
    wrapper->ntensors = 0;
    wrapper->ctensors = 0;

    lancius_graph_handle h = handle_register(wrapper, HANDLE_TYPE_GRAPH);
    if (h == 0) {
        lancius_schedule_destroy(wrapper->sched);
        lancius_arena_destroy(wrapper->scratch);
        lancius_graph_destroy(wrapper->g);
        free(wrapper);
        set_error(LANCIUS_ERR_OOM);
        return 0;
    }
    set_error(LANCIUS_OK);
    return h;
}

LANCIUS_EXPORT void lancius_graph_destroy_stable(lancius_graph_handle g) {
    set_error(LANCIUS_OK);
    void* ptr;
    if (!handle_validate(g, HANDLE_TYPE_GRAPH, &ptr)) { if (!g) return; set_error(LANCIUS_ERR_INVALID_HANDLE); return; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)ptr;
    if (!graph_valid_internal(wrapper)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return; }
    wrapper->magic = 0;
    wrapper->cookie = 0;
    for (size_t i = 0; i < wrapper->ntensors; i++) {
        lancius_tensor_internal* t = wrapper->tensors[i];
        if (t) {
            t->magic = 0;
            t->cookie = 0;
            handle_unregister_tensor(t->handle);
            free(t);
        }
    }
    free(wrapper->tensors);
    if (wrapper->sched) lancius_schedule_destroy(wrapper->sched);
    if (wrapper->scratch) lancius_arena_destroy(wrapper->scratch);
    if (wrapper->g) lancius_graph_destroy(wrapper->g);
    handle_unregister(g, HANDLE_TYPE_GRAPH);
    free(wrapper);
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_input(lancius_graph_handle g, size_t rows, size_t cols) {
    set_error(LANCIUS_OK);
    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }
    if (!graph_valid(g)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    void* wptr;
    if (!handle_validate(g, HANDLE_TYPE_GRAPH, &wptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)wptr;
    /* v12R1-205: prevent abort() via stable API on oversized tensors */
    size_t elems = 0;
    if (!lancius_checked_mul_size(rows, cols, &elems) || elems > 65535) {
        set_error(LANCIUS_ERR_SHAPE_MISMATCH);
        return 0;
    }
    lancius_node* n = lancius_input(wrapper->g, rows, cols);
    if (!n) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_OOM); return 0; }
    return wrap_tensor(wrapper, n);
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_matmul(lancius_graph_handle g, lancius_tensor_handle a, lancius_tensor_handle b) {
    set_error(LANCIUS_OK);
    if (!g || !a || !b) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }
    if (!graph_valid(g)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    void* wptr;
    if (!handle_validate(g, HANDLE_TYPE_GRAPH, &wptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)wptr;
    if (!tensor_valid(a) || !tensor_valid(b)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    if (!tensor_owned_by(a, wrapper) || !tensor_owned_by(b, wrapper)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    void* apr; void* bpr;
    if (!handle_validate(a, HANDLE_TYPE_TENSOR, &apr) || !handle_validate(b, HANDLE_TYPE_TENSOR, &bpr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_node* na = unwrap_tensor(a);
    lancius_node* nb = unwrap_tensor(b);
    if (!na || !nb) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_node* n = lancius_matmul(wrapper->g, na, nb);
    if (!n) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_SHAPE_MISMATCH); return 0; }
    return wrap_tensor(wrapper, n);
}

LANCIUS_EXPORT lancius_tensor_handle lancius_add_relu(lancius_graph_handle g, lancius_tensor_handle a) {
    set_error(LANCIUS_OK);
    if (!g || !a) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }
    if (!graph_valid(g)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    void* wptr;
    if (!handle_validate(g, HANDLE_TYPE_GRAPH, &wptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)wptr;
    if (!tensor_valid(a)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    if (!tensor_owned_by(a, wrapper)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    void* apr;
    if (!handle_validate(a, HANDLE_TYPE_TENSOR, &apr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_node* na = unwrap_tensor(a);
    if (!na) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_node* n = lancius_relu(wrapper->g, na);
    if (!n) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_SHAPE_MISMATCH); return 0; }
    return wrap_tensor(wrapper, n);
}

LANCIUS_EXPORT lancius_status lancius_bind_data(lancius_tensor_handle t, double* data_ptr) {
    set_error(LANCIUS_OK);
    if (!t || !data_ptr) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    if (!tensor_valid(t)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    lancius_node* n = unwrap_tensor(t);
    if (!n) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    /* A2: external data binding is explicitly non-owned */
    lancius_node_bind_external(n, data_ptr);
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

LANCIUS_EXPORT lancius_status lancius_compile_and_run(lancius_graph_handle g) {
    set_error(LANCIUS_OK);
    if (!g) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    if (!graph_valid(g)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    void* wptr;
    if (!handle_validate(g, HANDLE_TYPE_GRAPH, &wptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)wptr;

    if (wrapper->sched) { lancius_schedule_destroy(wrapper->sched); wrapper->sched = NULL; }

    lancius_clear_error();
    wrapper->sched = lancius_ir_schedule(wrapper->g);
    if (!wrapper->sched) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_OOM); return g_last_error; }

    /* v11A3 fix: auto-size scratch arena from liveness analysis */
    {
        size_t peak = lancius_schedule_peak_memory(wrapper->sched);
        if (peak > SIZE_MAX - (1024 * 1024)) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }
        size_t needed = peak + (1024 * 1024); /* 1MB headroom */
        if (needed > 16 * 1024 * 1024) {
            /* Despot V6 truth: allocate first, destroy only on success (was leak+NULL). */
            lancius_arena* na = lancius_arena_create(needed);
            if (!na) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }
            lancius_arena_destroy(wrapper->scratch);
            wrapper->scratch = na;
        }
    }
    lancius_arena_reset(wrapper->scratch);
    lancius_clear_error();
    lancius_schedule_execute(wrapper->sched, wrapper->scratch);
    if (lancius_get_error() != LANCIUS_ERROR_OK) {
        sync_internal_error();
        return g_last_error;
    }
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

LANCIUS_EXPORT lancius_status lancius_read_output(lancius_tensor_handle t, double* out_buffer, size_t buffer_size) {
    set_error(LANCIUS_OK);
    if (!t || !out_buffer) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    if (!tensor_valid(t)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    lancius_node* n = unwrap_tensor(t);
    if (!n) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    if (n->dtype != LANCIUS_DTYPE_FP64) { set_error(LANCIUS_ERR_UNSUPPORTED_OP); return LANCIUS_ERR_UNSUPPORTED_OP; }
    if (!n->runtime_data) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }

    if (buffer_size % sizeof(double) != 0) {
        set_error(LANCIUS_ERR_SHAPE_MISMATCH);
        return LANCIUS_ERR_SHAPE_MISMATCH;
    }
    size_t elems = 0;
    if (!lancius_node_elements_checked(n, &elems)) { set_error(LANCIUS_ERR_SHAPE_MISMATCH); return LANCIUS_ERR_SHAPE_MISMATCH; }
    size_t buffer_elems = buffer_size / sizeof(double);
    /* v11A3 fix: reject silent truncation */
    if (buffer_elems < elems) {
        set_error(LANCIUS_ERR_BUFFER_TOO_SMALL);
        return LANCIUS_ERR_BUFFER_TOO_SMALL;
    }
    if (elems > SIZE_MAX / sizeof(double)) { set_error(LANCIUS_ERR_OOM); return LANCIUS_ERR_OOM; }
    memcpy(out_buffer, n->runtime_data, elems * sizeof(double));
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}


/* v11A3 stable API expansion: model I/O */
LANCIUS_EXPORT lancius_graph_handle lancius_graph_load_stable(lancius_context ctx, const char* path) {
    set_error(LANCIUS_OK);
    if (!ctx || !path) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }
    if (!ctx_valid(ctx)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    void* cptr;
    if (!handle_validate(ctx, HANDLE_TYPE_CTX, &cptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_graph* g = lancius_graph_load(path);
    if (!g) { sync_internal_error(); if (g_last_error == LANCIUS_OK) set_error(LANCIUS_ERR_IO); return 0; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)malloc(sizeof(lancius_graph_internal));
    if (!wrapper) { lancius_graph_destroy(g); set_error(LANCIUS_ERR_OOM); return 0; }
    wrapper->g = g;
    wrapper->scratch = lancius_arena_create(64 * 1024 * 1024);
    wrapper->sched = NULL;
    wrapper->magic = 0x6A9A6A9Au;
    wrapper->cookie = lancius_handle_secret_init();
    wrapper->tensors = NULL;
    wrapper->ntensors = 0;
    wrapper->ctensors = 0;
    if (!wrapper->scratch) { lancius_graph_destroy(g); free(wrapper); set_error(LANCIUS_ERR_OOM); return 0; }
    lancius_graph_handle h = handle_register(wrapper, HANDLE_TYPE_GRAPH);
    if (h == 0) {
        lancius_arena_destroy(wrapper->scratch);
        lancius_graph_destroy(g);
        free(wrapper);
        set_error(LANCIUS_ERR_OOM);
        return 0;
    }
    set_error(LANCIUS_OK);
    return h;
}

LANCIUS_EXPORT lancius_status lancius_graph_save_stable(lancius_graph_handle g, const char* path) {
    set_error(LANCIUS_OK);
    if (!g || !path) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
    if (!graph_valid(g)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    void* wptr;
    if (!handle_validate(g, HANDLE_TYPE_GRAPH, &wptr)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return LANCIUS_ERR_INVALID_HANDLE; }
    lancius_graph_internal* wrapper = (lancius_graph_internal*)wptr;
    /* v11S H1 fix: propagate save failure to FFI consumers */
    lancius_clear_error();
    if (lancius_graph_save(wrapper->g, path) != 0) {
        lancius_error ierr = lancius_get_error();
        lancius_clear_error();
        if (ierr == LANCIUS_ERROR_NULL_PTR) { set_error(LANCIUS_ERR_NULL_PTR); return LANCIUS_ERR_NULL_PTR; }
        if (ierr == LANCIUS_ERROR_OVERFLOW || ierr == LANCIUS_ERROR_LIMIT) { set_error(LANCIUS_ERR_OVERFLOW); return LANCIUS_ERR_OVERFLOW; }
        set_error(LANCIUS_ERR_IO);
        return LANCIUS_ERR_IO;
    }
    set_error(LANCIUS_OK);
    return LANCIUS_OK;
}

/* v11A3 stable API expansion: tensor introspection */
LANCIUS_EXPORT size_t lancius_tensor_element_count(lancius_tensor_handle t) {
    set_error(LANCIUS_OK);
    if (!t) { set_error(LANCIUS_ERR_NULL_PTR); return 0; }
    if (!tensor_valid(t)) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    lancius_node* n = unwrap_tensor(t);
    if (!n) { set_error(LANCIUS_ERR_INVALID_HANDLE); return 0; }
    size_t elems = 0;
    if (!lancius_node_elements_checked(n, &elems)) { set_error(LANCIUS_ERR_SHAPE_MISMATCH); return 0; }
    set_error(LANCIUS_OK);
    return elems;
}

/* A3: dtype query */
LANCIUS_EXPORT int lancius_tensor_get_dtype(lancius_tensor_handle t) {
    set_error(LANCIUS_OK);
    if (!t) {
        set_error(LANCIUS_ERR_NULL_PTR);
        return -1;
    }

    if (!tensor_valid(t)) {
        set_error(LANCIUS_ERR_INVALID_HANDLE);
        return -1;
    }
    lancius_node* n = unwrap_tensor(t);
    if (!n) {
        set_error(LANCIUS_ERR_INVALID_HANDLE);
        return -1;
    }
    set_error(LANCIUS_OK);
    return (int)n->dtype;
}