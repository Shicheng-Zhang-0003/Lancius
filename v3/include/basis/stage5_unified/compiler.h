#ifndef basis_stage5_unified_compiler_h
#define basis_stage5_unified_compiler_h
#include "basis/stage1_atomic/scalar.h"
#include "basis/stage5_unified/symbolic.h"

typedef struct {
    char* name;
    value* value_pointer;
} variable_mapping;

typedef struct {
    variable_mapping* variable_mappings;
    size_t variable_mapping_count;
} compiler;

compiler* compiler_new (void);
void compiler_map (compiler* compiler_instance, char* variable_name, value* value_pointer);
value* compiler_compile (compiler* compiler_instance, symbol* symbol_node);
void compiler_free (compiler* compiler_instance);
#endif
