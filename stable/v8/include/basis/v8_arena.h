#ifndef BASIS_V8_ARENA_H
#define BASIS_V8_ARENA_H
#include <stddef.h>
#include <stdint.h>

typedef struct v8_arena v8_arena;

v8_arena* v8_arena_create(size_t block_size);
void* v8_arena_alloc(v8_arena* arena, size_t size, size_t alignment);
void v8_arena_reset(v8_arena* arena); // O(1) Teardown
void v8_arena_destroy(v8_arena* arena);

#endif
