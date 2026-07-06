#include "lancius/lancius_arena.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define ALIGN_UP(x, align) (((x) + ((align) - 1)) & ~((align) - 1))

typedef struct lancius_block {
    struct lancius_block* next;
    size_t size;
    size_t used;
    uint8_t* memory;
} lancius_block;

struct lancius_arena {
    lancius_block* first;
    lancius_block* current;
    size_t default_block_size;
};

static lancius_block* block_create(size_t size) {
    lancius_block* b = (lancius_block*)malloc(sizeof(lancius_block));
    if (!b) return NULL;
    b->memory = (uint8_t*)malloc(size);
    if (!b->memory) { free(b); return NULL; }
    b->size = size; b->used = 0; b->next = NULL;
    return b;
}

lancius_arena* lancius_arena_create(size_t block_size) {
    lancius_arena* a = (lancius_arena*)calloc(1, sizeof(lancius_arena));
    if (!a) return NULL;
    a->default_block_size = block_size > 0 ? block_size : (16 * 1024 * 1024); // 16MB default
    a->first = block_create(a->default_block_size);
    a->current = a->first;
    return a;
}

void* lancius_arena_alloc(lancius_arena* a, size_t size, size_t alignment) {
    if (!a || size == 0) return NULL;
    if (alignment == 0) alignment = 8;
    if (alignment & (alignment - 1)) return NULL; // Must be power of 2

    lancius_block* b = a->current;
    uintptr_t ptr = (uintptr_t)(b->memory + b->used);
    uintptr_t aligned = ALIGN_UP(ptr, alignment);
    size_t pad = aligned - ptr;

    if (b->used + pad + size > b->size) {
        size_t new_size = size + alignment > a->default_block_size ? size + alignment : a->default_block_size;
        lancius_block* nb = block_create(new_size);
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

void lancius_arena_reset(lancius_arena* a) {
    if (!a) return;
    lancius_block* b = a->first->next;
    while (b) {
        lancius_block* next = b->next;
        free(b->memory); free(b);
        b = next;
    }
    a->first->next = NULL;
    a->first->used = 0;
    a->current = a->first;
}

void lancius_arena_destroy(lancius_arena* a) {
    if (!a) return;
    lancius_block* b = a->first;
    while (b) {
        lancius_block* next = b->next;
        free(b->memory); free(b);
        b = next;
    }
    free(a);
}
