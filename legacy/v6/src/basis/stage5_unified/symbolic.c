
#include "basis/stage5_unified/symbolic.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

static char* basis_symbol_string_duplicate(const char* source_string) {
    size_t length = strlen(source_string) + 1;
    char* copy = (char*)malloc(length);
    if (copy) memcpy(copy, source_string, length);
    return copy;
}

static basis_symbol* basis_symbol_new(basis_symbol_type type) {
    basis_symbol* new_symbol = (basis_symbol*)malloc(sizeof(basis_symbol));
    new_symbol->type = type; new_symbol->name = NULL; new_symbol->operation = NULL;
    new_symbol->basis_value = 0.0; new_symbol->arguments = NULL; new_symbol->argument_count = 0;
    return new_symbol;
}

basis_symbol* basis_symbol_variable(char* variable_name) {
    basis_symbol* new_symbol = basis_symbol_new(basis_symbol_type_variable);
    new_symbol->name = basis_symbol_string_duplicate(variable_name);
    return new_symbol;
}

basis_symbol* basis_symbol_constant(double constant_value) {
    basis_symbol* new_symbol = basis_symbol_new(basis_symbol_type_constant);
    new_symbol->basis_value = constant_value;
    return new_symbol;
}

static basis_symbol* basis_symbol_operation_new(char* operation, size_t argument_count) {
    basis_symbol* new_symbol = basis_symbol_new(basis_symbol_type_operation);
    new_symbol->operation = basis_symbol_string_duplicate(operation);
    new_symbol->argument_count = argument_count;
    new_symbol->arguments = (basis_symbol**)malloc(sizeof(basis_symbol*) * argument_count);
    return new_symbol;
}

basis_symbol* basis_symbol_addition(basis_symbol* first_operand, basis_symbol* second_operand) {
    basis_symbol* new_symbol = basis_symbol_operation_new("+", 2);
    new_symbol->arguments[0] = first_operand; new_symbol->arguments[1] = second_operand;
    return new_symbol;
}

basis_symbol* basis_symbol_subtraction(basis_symbol* first_operand, basis_symbol* second_operand) {
    basis_symbol* new_symbol = basis_symbol_operation_new("-", 2);
    new_symbol->arguments[0] = first_operand; new_symbol->arguments[1] = second_operand;
    return new_symbol;
}

basis_symbol* basis_symbol_multiplication(basis_symbol* first_operand, basis_symbol* second_operand) {
    basis_symbol* new_symbol = basis_symbol_operation_new("*", 2);
    new_symbol->arguments[0] = first_operand; new_symbol->arguments[1] = second_operand;
    return new_symbol;
}

basis_symbol* basis_symbol_power(basis_symbol* base, double exponent) {
    basis_symbol* new_symbol = basis_symbol_operation_new("**", 1);
    new_symbol->arguments[0] = base; new_symbol->basis_value = exponent;
    return new_symbol;
}

basis_symbol* basis_symbol_logarithm(basis_symbol* input_symbol) {
    basis_symbol* new_symbol = basis_symbol_operation_new("log", 1);
    new_symbol->arguments[0] = input_symbol;
    return new_symbol;
}

basis_symbol* basis_symbol_exponential(basis_symbol* input_symbol) {
    basis_symbol* new_symbol = basis_symbol_operation_new("exp", 1);
    new_symbol->arguments[0] = input_symbol;
    return new_symbol;
}

basis_symbol* basis_symbol_copy(basis_symbol* source_symbol) {
    if (!source_symbol) return NULL;
    basis_symbol* output_symbol;
    if (source_symbol->type == basis_symbol_type_variable) {
        output_symbol = basis_symbol_variable(source_symbol->name);
    } else if (source_symbol->type == basis_symbol_type_constant) {
        output_symbol = basis_symbol_constant(source_symbol->basis_value);
    } else {
        output_symbol = basis_symbol_operation_new(source_symbol->operation, source_symbol->argument_count);
        output_symbol->basis_value = source_symbol->basis_value;
        for (size_t i = 0; i < source_symbol->argument_count; i++) {
            output_symbol->arguments[i] = basis_symbol_copy(source_symbol->arguments[i]);
        }
    }
    return output_symbol;
}

basis_symbol* basis_symbol_differentiation(basis_symbol* source_symbol, char* variable_name) {
    if (source_symbol->type == basis_symbol_type_constant) return basis_symbol_constant(0.0);
    if (source_symbol->type == basis_symbol_type_variable) return (strcmp(source_symbol->name, variable_name) == 0) ? basis_symbol_constant(1.0) : basis_symbol_constant(0.0);
    if (source_symbol->type == basis_symbol_type_operation) {
        if (strcmp(source_symbol->operation, "+") == 0) return basis_symbol_addition(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_differentiation(source_symbol->arguments[1], variable_name));
        if (strcmp(source_symbol->operation, "-") == 0) return basis_symbol_subtraction(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_differentiation(source_symbol->arguments[1], variable_name));
        if (strcmp(source_symbol->operation, "*") == 0) return basis_symbol_addition(basis_symbol_multiplication(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_copy(source_symbol->arguments[1])), basis_symbol_multiplication(basis_symbol_copy(source_symbol->arguments[0]), basis_symbol_differentiation(source_symbol->arguments[1], variable_name)));
        if (strcmp(source_symbol->operation, "**") == 0) {
            double exponent = source_symbol->basis_value;
            return basis_symbol_multiplication(basis_symbol_constant(exponent), basis_symbol_multiplication(basis_symbol_power(basis_symbol_copy(source_symbol->arguments[0]), exponent - 1.0), basis_symbol_differentiation(source_symbol->arguments[0], variable_name)));
        }
        if (strcmp(source_symbol->operation, "exp") == 0) return basis_symbol_multiplication(basis_symbol_exponential(basis_symbol_copy(source_symbol->arguments[0])), basis_symbol_differentiation(source_symbol->arguments[0], variable_name));
        if (strcmp(source_symbol->operation, "log") == 0) return basis_symbol_multiplication(basis_symbol_differentiation(source_symbol->arguments[0], variable_name), basis_symbol_power(basis_symbol_copy(source_symbol->arguments[0]), -1.0));
    }
    return basis_symbol_constant(0.0);
}

void basis_symbol_free(basis_symbol* target_symbol) {
    if (!target_symbol) return;
    if (target_symbol->name) free(target_symbol->name);
    if (target_symbol->operation) free(target_symbol->operation);
    for (size_t i = 0; i < target_symbol->argument_count; i++) basis_symbol_free(target_symbol->arguments[i]);
    if (target_symbol->arguments) free(target_symbol->arguments);
    free(target_symbol);
}

void basis_symbol_print(basis_symbol* target_symbol) {
    if (!target_symbol) { printf("NULL"); return; }
    if (target_symbol->type == basis_symbol_type_variable) printf("%s", target_symbol->name);
    else if (target_symbol->type == basis_symbol_type_constant) printf("%.2f", target_symbol->basis_value);
    else if (target_symbol->type == basis_symbol_type_operation) {
        if (target_symbol->argument_count == 2) {
            printf("("); basis_symbol_print(target_symbol->arguments[0]); printf(" %s ", target_symbol->operation); basis_symbol_print(target_symbol->arguments[1]); printf(")");
        } else {
            printf("%s(", target_symbol->operation); basis_symbol_print(target_symbol->arguments[0]);
            if (strcmp(target_symbol->operation, "**") == 0) printf(", %.2f", target_symbol->basis_value);
            printf(")");
        }
    }
}

basis_symbol* basis_symbol_simplify(basis_symbol* source_symbol) {
    if (!source_symbol) return NULL;
    if (source_symbol->type == basis_symbol_type_variable || source_symbol->type == basis_symbol_type_constant) return basis_symbol_copy(source_symbol);

    basis_symbol* first_argument = source_symbol->argument_count > 0 ? basis_symbol_simplify(source_symbol->arguments[0]) : NULL;
    basis_symbol* second_argument = source_symbol->argument_count > 1 ? basis_symbol_simplify(source_symbol->arguments[1]) : NULL;

    if (strcmp(source_symbol->operation, "+") == 0) {
        if (first_argument->type == basis_symbol_type_constant && first_argument->basis_value == 0.0) { basis_symbol_free(first_argument); return second_argument; }
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 0.0) { basis_symbol_free(second_argument); return first_argument; }
        if (first_argument->type == basis_symbol_type_constant && second_argument->type == basis_symbol_type_constant) {
            double result = first_argument->basis_value + second_argument->basis_value;
            basis_symbol_free(first_argument); basis_symbol_free(second_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "-") == 0) {
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 0.0) { basis_symbol_free(second_argument); return first_argument; }
        if (first_argument->type == basis_symbol_type_constant && second_argument->type == basis_symbol_type_constant) {
            double result = first_argument->basis_value - second_argument->basis_value;
            basis_symbol_free(first_argument); basis_symbol_free(second_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "*") == 0) {
        if (first_argument->type == basis_symbol_type_constant && first_argument->basis_value == 0.0) { basis_symbol_free(second_argument); return first_argument; }
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 0.0) { basis_symbol_free(first_argument); return second_argument; }
        if (first_argument->type == basis_symbol_type_constant && first_argument->basis_value == 1.0) { basis_symbol_free(first_argument); return second_argument; }
        if (second_argument->type == basis_symbol_type_constant && second_argument->basis_value == 1.0) { basis_symbol_free(second_argument); return first_argument; }
        if (first_argument->type == basis_symbol_type_constant && second_argument->type == basis_symbol_type_constant) {
            double result = first_argument->basis_value * second_argument->basis_value;
            basis_symbol_free(first_argument); basis_symbol_free(second_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "**") == 0) {
        if (source_symbol->basis_value == 0.0) { basis_symbol_free(first_argument); return basis_symbol_constant(1.0); }
        if (source_symbol->basis_value == 1.0) { return first_argument; }
        if (first_argument->type == basis_symbol_type_constant) {
            double result = pow(first_argument->basis_value, source_symbol->basis_value);
            basis_symbol_free(first_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "exp") == 0) {
        if (first_argument->type == basis_symbol_type_constant) {
            double result = exp(first_argument->basis_value);
            basis_symbol_free(first_argument); return basis_symbol_constant(result);
        }
    } else if (strcmp(source_symbol->operation, "log") == 0) {
        if (first_argument->type == basis_symbol_type_constant) {
            double result = log(first_argument->basis_value);
            basis_symbol_free(first_argument); return basis_symbol_constant(result);
        }
    }

    basis_symbol* output_symbol = basis_symbol_operation_new(source_symbol->operation, source_symbol->argument_count);
    output_symbol->basis_value = source_symbol->basis_value;
    if (source_symbol->argument_count > 0 && first_argument) output_symbol->arguments[0] = first_argument;
    if (source_symbol->argument_count > 1 && second_argument) output_symbol->arguments[1] = second_argument;
    for (size_t i = 2; i < source_symbol->argument_count; i++) output_symbol->arguments[i] = basis_symbol_copy(source_symbol->arguments[i]);
    return output_symbol;
}
