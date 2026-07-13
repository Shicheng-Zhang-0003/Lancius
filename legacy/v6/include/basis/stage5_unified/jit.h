#ifndef basis_stage5_unified_jit_h
#define basis_stage5_unified_jit_h
#include "basis/stage5_unified/symbolic.h"

typedef double (*basis_jit_func_t)(double*);

typedef struct {
    void* handle;
    basis_jit_func_t func;
    char* so_path;
} basis_jit_module;

basis_jit_module* basis_jit_compile(basis_symbol* root, char** var_names, size_t var_count);
double basis_jit_execute(basis_jit_module* mod, double* var_values);
void basis_jit_free(basis_jit_module* mod);

#endif
