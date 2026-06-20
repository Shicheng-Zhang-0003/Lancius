#ifndef basis_stage1_scalar_h
#define basis_stage1_scalar_h
#include <stdlib.h>
#include <stdbool.h>

typedef struct value {
    double data;
    double gradient;
    void (*backward_function) (struct value* output_value);
    struct value** previous_nodes;
    size_t previous_node_count;
    char* operation;
    double constant_value;
    size_t reference_count;
    bool visited;
} value;

value* value_new (double initial_data);
void value_retain (value* target_value);
void value_free (value* target_value);
value* value_addition (value* first_operand, value* second_operand);
value* value_multiplication (value* first_operand, value* second_operand);
value* value_power (value* base, double exponent);
value* value_exponential (value* input_value);
value* value_logarithm (value* input_value);
value* value_sine (value* input_value);
value* value_cosine (value* input_value);
value* value_rectified_linear_unit (value* input_value);
void value_zero_gradient (value* root_value);
void value_backward_propagation (value* root_value);
#endif
