#include "basis/v8_arena.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

typedef struct v8_block {
    struct v8_block* next;
    size_t size;
    size_t used;
    uint8_t* memory;
} v8_block;

struct v8_arena {
    v8_block* first;
    v8_block* current;
    size_t default_block_size;
};

static v8_block* block_create(size_t size) {
    v8_block* b = (v8_block*)malloc(sizeof(v8_block));
    if (!b) return NULL;
    b->memory = (uint8_t*)malloc(size);
    if (!b->memory) { free(b); return NULL; }
    b->size = size; b->used = 0; b->next = NULL;
    return b;
}

v8_arena* v8_arena_create(size_t block_size) {
    v8_arena* a = (v8_arena*)calloc(1, sizeof(v8_arena));
    if (!a) return NULL;
    a->default_block_size = block_size > 0 ? block_size : (16 * 1024 * 1024); // 16MB default
    a->first = block_create(a->default_block_size);
    a->current = a->first;
    return a;
}

void* v8_arena_alloc(v8_arena* a, size_t size, size_t alignment) {
    if (!a || size == 0) return NULL;
    if (alignment == 0) alignment = 8;
    if (alignment & (alignment - 1)) return NULL; // Must be power of 2

    v8_block* b = a->current;
    uintptr_t ptr = (uintptr_t)(b->memory + b->used);
    uintptr_t aligned = ALIGN_UP(ptr, alignment);
    size_t pad = aligned - ptr;

    if (b->used + pad + size > b->size) {
        size_t new_size = size + alignment > a->default_block_size ? size + alignment : a->default_block_size;
        v8_block* nb = block_create(new_size);
        if (!nb) { fprintf(stderr, "[ARENA FATAL] Failed to allocate block of size %zu!\n", new_size); return NULL; }
        b->next = nb;
        a->current = nb;
        b = nb;
        ptr = (uintptr_t)(b->memory + b->used);
        aligned = ALIGN_UP(ptr, alignment);
        pad = aligned - ptr;
    }
    b->used += pad + size;
    return (void*)aligned;
}

void v8_arena_reset(v8_arena* a) {
    if (!a) return;
    v8_block* b = a->first->next;
    while (b) {
        v8_block* next = b->next;
        free(b->memory); free(b);
        b = next;
    }
    a->first->next = NULL;
    a->first->used = 0;
    a->current = a->first;
}

void v8_arena_destroy(v8_arena* a) {
    if (!a) return;
    v8_block* b = a->first;
    while (b) {
        v8_block* next = b->next;
        free(b->memory); free(b);
        b = next;
    }
    free(a);
}
