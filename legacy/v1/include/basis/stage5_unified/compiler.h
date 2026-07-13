#ifndef BASIS_STAGE5_COMPILER_H
#define BASIS_STAGE5_COMPILER_H
#include "basis/stage5_unified/symbolic.h"
#include "basis/stage1_atomic/scalar.h"
typedef struct {
    char* name;
    Value* value;
} VarMapping;
typedef struct {
    VarMapping* vars;
    size_t var_count;
} Compiler;
Compiler* compiler_new ();
void compiler_map (Compiler* c, char* name, Value* v);
Value* compiler_compile (Compiler* c, Symbol* s);
void compiler_free (Compiler* c);
#endif
