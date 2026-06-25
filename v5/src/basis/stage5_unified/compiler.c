
#include "basis/stage5_unified/compiler.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

static char* basis_compiler_string_duplicate(const char* source_string) {
    size_t length = strlen(source_string) + 1;
    char* copy = (char*)malloc(length);
    if (copy) memcpy(copy, source_string, length);
    return copy;
}

// --- CSE Helper Functions ---
static size_t basis_symbol_hash(basis_symbol* sym) {
    if (!sym) return 0;
    size_t h = sym->type;
    if (sym->type == basis_symbol_type_variable) {
        const char* p = sym->name;
        while (*p) { h = h * 31 + *p++; }
    } else if (sym->type == basis_symbol_type_constant) {
        union { double d; size_t i; } u;
        u.d = sym->basis_value;
        h ^= u.i;
    } else {
        const char* p = sym->operation;
        while (*p) { h = h * 31 + *p++; }
        h ^= sym->argument_count;
        for (size_t i = 0; i < sym->argument_count; i++) {
            h ^= basis_symbol_hash(sym->arguments[i]) * (i + 1);
        }
        if (sym->operation && strcmp(sym->operation, "**") == 0) {
            union { double d; size_t i; } u;
            u.d = sym->basis_value;
            h ^= u.i;
        }
    }
    return h;
}

static bool basis_symbol_structural_equal(basis_symbol* a, basis_symbol* b) {
    if (!a || !b) return a == b;
    if (a->type != b->type) return false;
    if (a->type == basis_symbol_type_variable) return strcmp(a->name, b->name) == 0;
    if (a->type == basis_symbol_type_constant) return a->basis_value == b->basis_value;
    if (strcmp(a->operation, b->operation) != 0) return false;
    if (a->argument_count != b->argument_count) return false;
    if (a->operation && strcmp(a->operation, "**") == 0 && a->basis_value != b->basis_value) return false;
    for (size_t i = 0; i < a->argument_count; i++) {
        if (!basis_symbol_structural_equal(a->arguments[i], b->arguments[i])) return false;
    }
    return true;
}
// ----------------------------

basis_compiler* basis_compiler_new() {
    basis_compiler* new_compiler = (basis_compiler*)malloc(sizeof(basis_compiler));
    new_compiler->variable_mappings = NULL;
    new_compiler->variable_mapping_count = 0;
    new_compiler->cse_cache = NULL;
    new_compiler->cse_count = 0;
    new_compiler->cse_cap = 0;
    return new_compiler;
}

void basis_compiler_map(basis_compiler* basis_compiler_instance, char* variable_name, basis_value* basis_value_pointer) {
    basis_compiler_instance->variable_mapping_count++;
    basis_compiler_instance->variable_mappings = (basis_variable_mapping*)realloc(basis_compiler_instance->variable_mappings, sizeof(basis_variable_mapping) * basis_compiler_instance->variable_mapping_count);
    basis_compiler_instance->variable_mappings[basis_compiler_instance->variable_mapping_count - 1].name = basis_compiler_string_duplicate(variable_name);
    basis_compiler_instance->variable_mappings[basis_compiler_instance->variable_mapping_count - 1].basis_value_pointer = basis_value_pointer;
}

static basis_value* find_variable(basis_compiler* basis_compiler_instance, char* variable_name) {
    for (size_t i = 0; i < basis_compiler_instance->variable_mapping_count; i++) {
        if (strcmp(basis_compiler_instance->variable_mappings[i].name, variable_name) == 0) return basis_compiler_instance->variable_mappings[i].basis_value_pointer;
    }
    return NULL;
}

basis_value* basis_compiler_compile(basis_compiler* basis_compiler_instance, basis_symbol* basis_symbol_node) {
    if (!basis_compiler_instance || !basis_symbol_node) return NULL;

    // Phase C: CSE Check
    size_t h = basis_symbol_hash(basis_symbol_node);
    for (size_t i = 0; i < basis_compiler_instance->cse_count; i++) {
        if (basis_compiler_instance->cse_cache[i].hash == h &&
            basis_symbol_structural_equal(basis_compiler_instance->cse_cache[i].sym, basis_symbol_node)) {
            basis_value_retain(basis_compiler_instance->cse_cache[i].val);
            return basis_compiler_instance->cse_cache[i].val;
        }
    }

    if (basis_symbol_node->type == basis_symbol_type_constant) return basis_value_new(basis_symbol_node->basis_value);
    if (basis_symbol_node->type == basis_symbol_type_variable) {
        basis_value* found_value = find_variable(basis_compiler_instance, basis_symbol_node->name);
        if (!found_value) {
            fprintf(stderr, "Compiler: undefined variable '%s'\n", basis_symbol_node->name);
            return NULL;
        }
        basis_value_retain(found_value);
        return found_value;
    }
    if (basis_symbol_node->type == basis_symbol_type_operation) {
        basis_value** arguments = calloc(basis_symbol_node->argument_count, sizeof(basis_value*));
        if (!arguments) return NULL;
        for (size_t i = 0; i < basis_symbol_node->argument_count; i++) {
            arguments[i] = basis_compiler_compile(basis_compiler_instance, basis_symbol_node->arguments[i]);
            if (!arguments[i]) {
                for (size_t j = 0; j < i; j++) basis_value_free(arguments[j]);
                free(arguments); return NULL;
            }
        }
        basis_value* output_value = NULL;
        if (strcmp(basis_symbol_node->operation, "+") == 0) output_value = basis_value_addition(arguments[0], arguments[1]);
        else if (strcmp(basis_symbol_node->operation, "-") == 0) output_value = basis_value_subtraction(arguments[0], arguments[1]);
        else if (strcmp(basis_symbol_node->operation, "*") == 0) output_value = basis_value_multiplication(arguments[0], arguments[1]);
        else if (strcmp(basis_symbol_node->operation, "**") == 0) output_value = basis_value_power(arguments[0], basis_symbol_node->basis_value);
        else if (strcmp(basis_symbol_node->operation, "exp") == 0) output_value = basis_value_exponential(arguments[0]);
        else if (strcmp(basis_symbol_node->operation, "log") == 0) output_value = basis_value_logarithm(arguments[0]);
        else {
            fprintf(stderr, "Compiler: unknown operation '%s'\n", basis_symbol_node->operation);
            for (size_t i = 0; i < basis_symbol_node->argument_count; i++) basis_value_free(arguments[i]);
            free(arguments); return NULL;
        }
        for (size_t i = 0; i < basis_symbol_node->argument_count; i++) basis_value_free(arguments[i]);
        free(arguments);

        // Phase C: CSE Store (Weak Reference)
        if (output_value) {
            if (basis_compiler_instance->cse_count >= basis_compiler_instance->cse_cap) {
                basis_compiler_instance->cse_cap = basis_compiler_instance->cse_cap == 0 ? 16 : basis_compiler_instance->cse_cap * 2;
                basis_compiler_instance->cse_cache = realloc(basis_compiler_instance->cse_cache, sizeof(cse_entry) * basis_compiler_instance->cse_cap);
            }
            basis_compiler_instance->cse_cache[basis_compiler_instance->cse_count].sym = basis_symbol_node;
            basis_compiler_instance->cse_cache[basis_compiler_instance->cse_count].val = output_value;
            basis_compiler_instance->cse_cache[basis_compiler_instance->cse_count].hash = h;
            basis_compiler_instance->cse_count++;
        }

        return output_value;
    }
    return NULL;
}

void basis_compiler_free(basis_compiler* basis_compiler_instance) {
    if (!basis_compiler_instance) return;
    for (size_t i = 0; i < basis_compiler_instance->variable_mapping_count; i++) free(basis_compiler_instance->variable_mappings[i].name);
    if (basis_compiler_instance->variable_mappings) free(basis_compiler_instance->variable_mappings);
    if (basis_compiler_instance->cse_cache) free(basis_compiler_instance->cse_cache);
    free(basis_compiler_instance);
}
