#include "basis/stage5_unified/compiler.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
static char* compiler_string_duplicate (const char* source_string) {
    size_t length = strlen (source_string) + 1;
    char* copy = (char*) malloc (length);
    if (copy) {memcpy (copy, source_string, length);}
    return copy;
} compiler* compiler_new () {
    compiler* new_compiler = (compiler*) malloc (sizeof (compiler));
    new_compiler -> variable_mappings = NULL;
    new_compiler -> variable_mapping_count = 0;
    return new_compiler;
} void compiler_map (compiler* compiler_instance, char* variable_name, value* value_pointer) {
    compiler_instance -> variable_mapping_count ++;
    compiler_instance -> variable_mappings = (variable_mapping*) realloc (compiler_instance -> variable_mappings, sizeof (variable_mapping) * compiler_instance -> variable_mapping_count);
    compiler_instance -> variable_mappings [compiler_instance -> variable_mapping_count - 1].name = compiler_string_duplicate (variable_name);
    compiler_instance -> variable_mappings [compiler_instance -> variable_mapping_count - 1].value_pointer = value_pointer;
} static value* find_variable (compiler* compiler_instance, char* variable_name) {
    for (size_t i = 0; i < compiler_instance -> variable_mapping_count; i ++) {
        if (strcmp (compiler_instance -> variable_mappings [i].name, variable_name) == 0) {return compiler_instance -> variable_mappings [i].value_pointer;}
    }
    return NULL;
} value* compiler_compile (compiler* compiler_instance, symbol* symbol_node) {
    if ((!compiler_instance) || (!symbol_node)) {return NULL;}
    if (symbol_node -> type == symbol_type_constant) {return value_new (symbol_node -> value);}
    if (symbol_node -> type == symbol_type_variable) {
        value* found_value = find_variable (compiler_instance, symbol_node -> name);
        if (!found_value) {
            fprintf (stderr, "Compiler: undefined variable '%s'\n", symbol_node -> name);
            return NULL;
        }
        value_retain (found_value);
        return found_value;
    }
    if (symbol_node -> type == symbol_type_operation) {
        value** arguments = calloc (symbol_node -> argument_count, sizeof (value*));
        if (!arguments) {return NULL;}
        for (size_t i = 0; i < symbol_node -> argument_count; i ++) {
            arguments [i] = compiler_compile (compiler_instance, symbol_node -> arguments [i]);
            if (!arguments [i]) {
                for (size_t j = 0; j < i; j ++) {value_free (arguments [j]);}
                free (arguments);
                return NULL;
            }
        }
        value* output_value = NULL;
        if (strcmp (symbol_node -> operation, "+") == 0) {output_value = value_addition (arguments [0], arguments [1]);}
        else if (strcmp (symbol_node -> operation, "*") == 0) {output_value = value_multiplication (arguments [0], arguments [1]);}
        else if (strcmp (symbol_node -> operation, "**") == 0) {output_value = value_power (arguments [0], symbol_node -> value);}
        else if (strcmp (symbol_node -> operation, "exp") == 0) {output_value = value_exponential (arguments [0]);}
        else if (strcmp (symbol_node -> operation, "log") == 0) {output_value = value_logarithm (arguments [0]);}
        else {
            fprintf (stderr, "Compiler: unknown operation '%s'\n", symbol_node -> operation);
            for (size_t i = 0; i < symbol_node -> argument_count; i ++) {value_free (arguments [i]);}
            free (arguments);
            return NULL;
        }
        for (size_t i = 0; i < symbol_node -> argument_count; i ++) {value_free (arguments [i]);}
        free (arguments);
        return output_value;
    }
    return NULL;
} void compiler_free (compiler* compiler_instance) {
    if (!compiler_instance) {return;}
    for (size_t i = 0; i < compiler_instance -> variable_mapping_count; i ++) {free (compiler_instance -> variable_mappings [i].name);}
    if (compiler_instance -> variable_mappings) {free (compiler_instance -> variable_mappings);}
    free (compiler_instance);
}
