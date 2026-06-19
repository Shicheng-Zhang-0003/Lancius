#ifndef basis_stage5_symbolic_h
#define basis_stage5_symbolic_h
#include <stdlib.h>
typedef enum {
    symbol_type_variable,
    symbol_type_constant,
    symbol_type_operation
} symbol_type;
typedef struct symbol {
    symbol_type type;
    char* name;
    double value;
    char* operation;
    struct symbol** arguments;
    size_t argument_count;
} symbol;
symbol* symbol_variable (char* variable_name);
symbol* symbol_constant (double constant_value);
symbol* symbol_addition (symbol* first_operand, symbol* second_operand);
symbol* symbol_multiplication (symbol* first_operand, symbol* second_operand);
symbol* symbol_power (symbol* base, double exponent);
symbol* symbol_logarithm (symbol* input_symbol);
symbol* symbol_exponential (symbol* input_symbol);
symbol* symbol_copy (symbol* source_symbol);
symbol* symbol_differentiation (symbol* source_symbol, char* variable_name);
symbol* symbol_simplify (symbol* source_symbol);
void symbol_print (symbol* target_symbol);
void symbol_free (symbol* target_symbol);
#endif
