
#ifndef basis_stage1_scalar_h
#define basis_stage1_scalar_h
#include <stdlib.h>
#include <stdbool.h>

typedef struct basis_value {
    double data;
    double gradient;
    void (*backward_function)(struct basis_value* output_value);
    struct basis_value** previous_nodes;
    size_t previous_node_count;
    char* operation;
    double constant_value;
    size_t reference_count;
    bool visited;
} basis_value;

basis_value* basis_value_new(double initial_data);
void basis_value_retain(basis_value* target_value);
void basis_value_free(basis_value* target_value);
basis_value* basis_value_addition(basis_value* first_operand, basis_value* second_operand);
basis_value* basis_value_subtraction(basis_value* first_operand, basis_value* second_operand);
basis_value* basis_value_multiplication(basis_value* first_operand, basis_value* second_operand);
basis_value* basis_value_power(basis_value* base, double exponent);
basis_value* basis_value_exponential(basis_value* input_value);
basis_value* basis_value_logarithm(basis_value* input_value);
basis_value* basis_value_sine(basis_value* input_value);
basis_value* basis_value_cosine(basis_value* input_value);
basis_value* basis_value_tanh(basis_value* input_value);
basis_value* basis_value_rectified_linear_unit(basis_value* input_value);
void basis_value_zero_gradient(basis_value* root_value);
void basis_value_backward_propagation(basis_value* root_value);
#endif
