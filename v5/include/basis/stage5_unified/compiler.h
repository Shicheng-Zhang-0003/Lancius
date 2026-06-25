
#ifndef basis_stage5_unified_compiler_h
#define basis_stage5_unified_compiler_h
#include "basis/stage1_atomic/scalar.h"
#include "basis/stage5_unified/symbolic.h"

typedef struct { char* name; basis_value* basis_value_pointer; } basis_variable_mapping;

// V5 Phase C: CSE Cache Entry
typedef struct {
    basis_symbol* sym;
    basis_value* val;
    size_t hash;
} cse_entry;

typedef struct {
    basis_variable_mapping* variable_mappings;
    size_t variable_mapping_count;

    cse_entry* cse_cache;
    size_t cse_count;
    size_t cse_cap;
} basis_compiler;

basis_compiler* basis_compiler_new(void);
void basis_compiler_map(basis_compiler* basis_compiler_instance, char* variable_name, basis_value* basis_value_pointer);
basis_value* basis_compiler_compile(basis_compiler* basis_compiler_instance, basis_symbol* basis_symbol_node);
void basis_compiler_free(basis_compiler* basis_compiler_instance);
#endif
