#ifndef BASIS_STAGE5_SYMBOLIC_H
#define BASIS_STAGE5_SYMBOLIC_H
#include <stdlib.h>
#include "basis/stage2_linear/tensor.h"
typedef enum {
    SYM_VAR,
    SYM_CONST,
    SYM_OP
} SymbolType;
typedef struct Symbol {
    SymbolType type;
    char* name;
    double value;
    char* op;
    struct Symbol** args;
    size_t arg_count;
} Symbol;
Symbol* sym_var (char* name);
Symbol* sym_const (double val);
Symbol* sym_add (Symbol* a, Symbol* b);
Symbol* sym_mul (Symbol* a, Symbol* b);
Symbol* sym_pow (Symbol* a, double exp);
Symbol* sym_log (Symbol* a);
Symbol* sym_exp (Symbol* a);
Symbol* sym_copy (Symbol* expr);
Symbol* sym_diff (Symbol* expr, char* var_name);
Symbol* sym_simplify (Symbol* expr);
void sym_print (Symbol* s);
void sym_free (Symbol* s);
#endif
