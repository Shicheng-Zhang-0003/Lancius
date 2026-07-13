#ifndef Lancius_LANCIUS_ARENA_H
#define Lancius_LANCIUS_ARENA_H
#include <stddef.h>
#include <stdint.h>

typedef struct lancius_arena lancius_arena;

lancius_arena* lancius_arena_create(size_t block_size);
void* lancius_arena_alloc(lancius_arena* arena, size_t size, size_t alignment);
void lancius_arena_reset(lancius_arena* arena); // O(1) Teardown
void lancius_arena_destroy(lancius_arena* arena);

#endif
