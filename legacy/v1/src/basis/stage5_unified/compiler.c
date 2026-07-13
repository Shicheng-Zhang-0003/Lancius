#include "basis/stage5_unified/compiler.h"
#include <string.h>
#include <stdio.h>

Compiler* compiler_new() {
    Compiler* c = (Compiler*)malloc(sizeof(Compiler));
    c->vars = NULL;
    c->var_count = 0;
    return c;
}

void compiler_map(Compiler* c, char* name, Value* v) {
    c->var_count++;
    c->vars = (VarMapping*)realloc(c->vars, sizeof(VarMapping) * c->var_count);
    c->vars[c->var_count - 1].name = strdup(name);
    c->vars[c->var_count - 1].value = v;
}

static Value* find_var(Compiler* c, char* name) {
    for (size_t i = 0; i < c->var_count; i++) {
        if (strcmp(c->vars[i].name, name) == 0) return c->vars[i].value;
    }
    return NULL;
}

Value* compiler_compile(Compiler* c, Symbol* s) {
    if (s->type == SYM_CONST) return val_new(s->value);
    if (s->type == SYM_VAR) {
        Value* v = find_var(c, s->name);
        if (!v) return NULL;
        return v;
    }
    if (s->type == SYM_OP) {
        if (strcmp(s->op, "+") == 0) return val_add(compiler_compile(c, s->args[0]), compiler_compile(c, s->args[1]));
        if (strcmp(s->op, "*") == 0) return val_mul(compiler_compile(c, s->args[0]), compiler_compile(c, s->args[1]));
        if (strcmp(s->op, "**") == 0) return val_pow(compiler_compile(c, s->args[0]), s->value);
        if (strcmp(s->op, "exp") == 0) return val_exp(compiler_compile(c, s->args[0]));
        if (strcmp(s->op, "log") == 0) return val_log(compiler_compile(c, s->args[0]));
    }
    return NULL;
}

void compiler_free(Compiler* c) {
    for (size_t i = 0; i < c->var_count; i++) free(c->vars[i].name);
    if (c->vars) free(c->vars);
    free(c);
}
