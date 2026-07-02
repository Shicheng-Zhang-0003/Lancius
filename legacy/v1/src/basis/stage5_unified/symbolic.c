#include "basis/stage5_unified/symbolic.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

static Symbol* sym_new(SymbolType type) {
    Symbol* s = (Symbol*)malloc(sizeof(Symbol));
    s->type = type;
    s->name = NULL;
    s->op = NULL;
    s->value = 0.0;
    s->args = NULL;
    s->arg_count = 0;
    return s;
}

Symbol* sym_var(char* name) {
    Symbol* s = sym_new(SYM_VAR);
    s->name = strdup(name);
    return s;
}

Symbol* sym_const(double val) {
    Symbol* s = sym_new(SYM_CONST);
    s->value = val;
    return s;
}

static Symbol* sym_op_new(char* op, size_t count) {
    Symbol* s = sym_new(SYM_OP);
    s->op = strdup(op);
    s->arg_count = count;
    s->args = (Symbol**)malloc(sizeof(Symbol*) * count);
    return s;
}

Symbol* sym_add(Symbol* a, Symbol* b) {
    Symbol* s = sym_op_new("+", 2);
    s->args[0] = a;
    s->args[1] = b;
    return s;
}

Symbol* sym_mul(Symbol* a, Symbol* b) {
    Symbol* s = sym_op_new("*", 2);
    s->args[0] = a;
    s->args[1] = b;
    return s;
}

Symbol* sym_pow(Symbol* a, double exp) {
    Symbol* s = sym_op_new("**", 1);
    s->args[0] = a;
    s->value = exp;
    return s;
}

Symbol* sym_log(Symbol* a) {
    Symbol* s = sym_op_new("log", 1);
    s->args[0] = a;
    return s;
}

Symbol* sym_exp(Symbol* a) {
    Symbol* s = sym_op_new("exp", 1);
    s->args[0] = a;
    return s;
}

Symbol* sym_copy(Symbol* e) {
    if (!e) return NULL;
    Symbol* out;
    if (e->type == SYM_VAR) out = sym_var(e->name);
    else if (e->type == SYM_CONST) out = sym_const(e->value);
    else {
        out = sym_op_new(e->op, e->arg_count);
        out->value = e->value;
        for (size_t i = 0; i < e->arg_count; i++) out->args[i] = sym_copy(e->args[i]);
    }
    return out;
}

Symbol* sym_diff(Symbol* e, char* var) {
    if (e->type == SYM_CONST) return sym_const(0.0);
    if (e->type == SYM_VAR) return (strcmp(e->name, var) == 0) ? sym_const(1.0) : sym_const(0.0);
    if (e->type == SYM_OP) {
        if (strcmp(e->op, "+") == 0) return sym_add(sym_diff(e->args[0], var), sym_diff(e->args[1], var));
        if (strcmp(e->op, "*") == 0) return sym_add(sym_mul(sym_diff(e->args[0], var), sym_copy(e->args[1])), sym_mul(sym_copy(e->args[0]), sym_diff(e->args[1], var)));
        if (strcmp(e->op, "**") == 0) {
            double n = e->value;
            return sym_mul(sym_const(n), sym_mul(sym_pow(sym_copy(e->args[0]), n - 1.0), sym_diff(e->args[0], var)));
        }
        if (strcmp(e->op, "exp") == 0) return sym_mul(sym_exp(sym_copy(e->args[0])), sym_diff(e->args[0], var));
        if (strcmp(e->op, "log") == 0) return sym_mul(sym_diff(e->args[0], var), sym_pow(sym_copy(e->args[0]), -1.0));
    }
    return sym_const(0.0);
}

void sym_free(Symbol* s) {
    if (!s) return;
    if (s->name) free(s->name);
    if (s->op) free(s->op);
    for (size_t i = 0; i < s->arg_count; i++) sym_free(s->args[i]);
    if (s->args) free(s->args);
    free(s);
}

void sym_print(Symbol* s) {
    if (s->type == SYM_VAR) printf("%s", s->name);
    else if (s->type == SYM_CONST) printf("%.2f", s->value);
    else if (s->type == SYM_OP) {
        if (s->arg_count == 2) {
            printf("("); sym_print(s->args[0]); printf(" %s ", s->op); sym_print(s->args[1]); printf(")");
        } else {
            printf("%s(", s->op); sym_print(s->args[0]);
            if (strcmp(s->op, "**") == 0) printf(", %.2f", s->value);
            printf(")");
        }
    }
}

Symbol* sym_simplify(Symbol* e) {
    return sym_copy(e);
}
