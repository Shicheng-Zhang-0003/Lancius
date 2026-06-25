
#ifndef basis_stage5_symbolic_h
#define basis_stage5_symbolic_h
#include <stdlib.h>
typedef enum { basis_symbol_type_variable, basis_symbol_type_constant, basis_symbol_type_operation } basis_symbol_type;
typedef struct basis_symbol {
    basis_symbol_type type; char* name; double basis_value; char* operation;
    struct basis_symbol** arguments; size_t argument_count;
} basis_symbol;
basis_symbol* basis_symbol_variable(char* variable_name);
basis_symbol* basis_symbol_constant(double constant_value);
basis_symbol* basis_symbol_addition(basis_symbol* first_operand, basis_symbol* second_operand);
basis_symbol* basis_symbol_subtraction(basis_symbol* first_operand, basis_symbol* second_operand);
basis_symbol* basis_symbol_multiplication(basis_symbol* first_operand, basis_symbol* second_operand);
basis_symbol* basis_symbol_power(basis_symbol* base, double exponent);
basis_symbol* basis_symbol_logarithm(basis_symbol* input_symbol);
basis_symbol* basis_symbol_exponential(basis_symbol* input_symbol);
basis_symbol* basis_symbol_copy(basis_symbol* source_symbol);
basis_symbol* basis_symbol_differentiation(basis_symbol* source_symbol, char* variable_name);
basis_symbol* basis_symbol_simplify(basis_symbol* source_symbol);
void basis_symbol_print(basis_symbol* target_symbol);
void basis_symbol_free(basis_symbol* target_symbol);
#endif
