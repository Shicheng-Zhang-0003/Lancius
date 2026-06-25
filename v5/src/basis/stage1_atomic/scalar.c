
#include "basis/stage1_atomic/scalar.h"
#include "basis/core/error.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static char* basis_value_string_duplicate(const char* source) {
    if (!source) return NULL;
    size_t len = strlen(source) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, source, len);
    return copy;
}

basis_value* basis_value_new(double initial_data) {
    basis_value* new_value = (basis_value*)malloc(sizeof(basis_value));
    if (!new_value) { BASIS_SET_ERROR(BASIS_OUT_OF_MEMORY, "Failed to allocate basis_value"); return NULL; }
    new_value->data = initial_data;
    new_value->gradient = 0.0;
    new_value->backward_function = NULL;
    new_value->previous_nodes = NULL;
    new_value->previous_node_count = 0;
    new_value->operation = basis_value_string_duplicate("");
    new_value->constant_value = 0.0;
    new_value->reference_count = 1;
    new_value->visited = false;
    return new_value;
}

void basis_value_retain(basis_value* target_value) {
    if (target_value) target_value->reference_count++;
}

static basis_value* basis_value_operation_initialize(basis_value* first_operand, basis_value* second_operand, char* operation, size_t operand_count) {
    basis_value* output_value = basis_value_new(0.0);
    if (!output_value) return NULL;
    free(output_value->operation);
    output_value->operation = basis_value_string_duplicate(operation);
    output_value->previous_node_count = operand_count;
    output_value->previous_nodes = (basis_value**)malloc(sizeof(basis_value*) * operand_count);
    if (first_operand) output_value->previous_nodes[0] = first_operand;
    if (second_operand && operand_count > 1) output_value->previous_nodes[1] = second_operand;
    return output_value;
}

static void backward_addition(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += 1.0 * output_value->gradient;
    output_value->previous_nodes[1]->gradient += 1.0 * output_value->gradient;
}

basis_value* basis_value_addition(basis_value* first_operand, basis_value* second_operand) {
    BASIS_CHECK_NULL(first_operand); BASIS_CHECK_NULL(second_operand);
    basis_value* output_value = basis_value_operation_initialize(first_operand, second_operand, "+", 2);
    output_value->data = first_operand->data + second_operand->data;
    output_value->backward_function = backward_addition;
    basis_value_retain(first_operand); basis_value_retain(second_operand);
    return output_value;
}

static void backward_subtraction(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += 1.0 * output_value->gradient;
    output_value->previous_nodes[1]->gradient += -1.0 * output_value->gradient;
}

basis_value* basis_value_subtraction(basis_value* first_operand, basis_value* second_operand) {
    BASIS_CHECK_NULL(first_operand); BASIS_CHECK_NULL(second_operand);
    basis_value* output_value = basis_value_operation_initialize(first_operand, second_operand, "-", 2);
    output_value->data = first_operand->data - second_operand->data;
    output_value->backward_function = backward_subtraction;
    basis_value_retain(first_operand); basis_value_retain(second_operand);
    return output_value;
}

static void backward_multiplication(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += output_value->previous_nodes[1]->data * output_value->gradient;
    output_value->previous_nodes[1]->gradient += output_value->previous_nodes[0]->data * output_value->gradient;
}

basis_value* basis_value_multiplication(basis_value* first_operand, basis_value* second_operand) {
    BASIS_CHECK_NULL(first_operand); BASIS_CHECK_NULL(second_operand);
    basis_value* output_value = basis_value_operation_initialize(first_operand, second_operand, "*", 2);
    output_value->data = first_operand->data * second_operand->data;
    output_value->backward_function = backward_multiplication;
    basis_value_retain(first_operand); basis_value_retain(second_operand);
    return output_value;
}

static void backward_power(basis_value* output_value) {
    double base = output_value->previous_nodes[0]->data;
    double exponent = output_value->constant_value;
    output_value->previous_nodes[0]->gradient += (exponent * pow(base, exponent - 1.0)) * output_value->gradient;
}

basis_value* basis_value_power(basis_value* base, double exponent) {
    BASIS_CHECK_NULL(base);
    basis_value* output_value = basis_value_operation_initialize(base, NULL, "**", 1);
    output_value->data = pow(base->data, exponent);
    output_value->constant_value = exponent;
    output_value->backward_function = backward_power;
    basis_value_retain(base);
    return output_value;
}

static void backward_exponential(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += output_value->data * output_value->gradient;
}

basis_value* basis_value_exponential(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "exp", 1);
    output_value->data = exp(input_value->data);
    output_value->backward_function = backward_exponential;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_logarithm(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += (1.0 / output_value->previous_nodes[0]->data) * output_value->gradient;
}

basis_value* basis_value_logarithm(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    if (input_value->data <= 0.0) {
        BASIS_SET_ERROR(BASIS_DOMAIN_ERROR, "Logarithm of non-positive number");
        return NULL;
    }
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "log", 1);
    output_value->data = log(input_value->data);
    output_value->backward_function = backward_logarithm;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_sine(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += cos(output_value->previous_nodes[0]->data) * output_value->gradient;
}

basis_value* basis_value_sine(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "sin", 1);
    output_value->data = sin(input_value->data);
    output_value->backward_function = backward_sine;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_cosine(basis_value* output_value) {
    output_value->previous_nodes[0]->gradient += -sin(output_value->previous_nodes[0]->data) * output_value->gradient;
}

basis_value* basis_value_cosine(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "cos", 1);
    output_value->data = cos(input_value->data);
    output_value->backward_function = backward_cosine;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_tanh(basis_value* output_value) {
    double tanh_val = output_value->data;
    output_value->previous_nodes[0]->gradient += (1.0 - tanh_val * tanh_val) * output_value->gradient;
}

basis_value* basis_value_tanh(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "tanh", 1);
    output_value->data = tanh(input_value->data);
    output_value->backward_function = backward_tanh;
    basis_value_retain(input_value);
    return output_value;
}

static void backward_rectified_linear_unit(basis_value* output_value) {
    double gradient_multiplier = (output_value->data > 0) ? 1.0 : 0.0;
    output_value->previous_nodes[0]->gradient += gradient_multiplier * output_value->gradient;
}

basis_value* basis_value_rectified_linear_unit(basis_value* input_value) {
    BASIS_CHECK_NULL(input_value);
    basis_value* output_value = basis_value_operation_initialize(input_value, NULL, "ReLU", 1);
    output_value->data = (input_value->data > 0) ? input_value->data : 0.0;
    output_value->backward_function = backward_rectified_linear_unit;
    basis_value_retain(input_value);
    return output_value;
}

static void build_topological_order(basis_value* root, basis_value*** topological_order, size_t* order_size, size_t* order_capacity) {
    size_t stack_cap = 256;
    basis_value** stack = (basis_value**)malloc(sizeof(basis_value*) * stack_cap);
    size_t* edge_index = (size_t*)malloc(sizeof(size_t) * stack_cap);
    size_t stack_ptr = 0;

    stack[stack_ptr] = root;
    edge_index[stack_ptr] = 0;
    root->visited = true;
    stack_ptr++;

    while (stack_ptr > 0) {
        basis_value* current = stack[stack_ptr - 1];
        size_t idx = edge_index[stack_ptr - 1];

        if (idx < current->previous_node_count) {
            edge_index[stack_ptr - 1]++;
            basis_value* child = current->previous_nodes[idx];
            if (!child->visited) {
                child->visited = true;
                if (stack_ptr >= stack_cap) {
                    stack_cap *= 2;
                    stack = (basis_value**)realloc(stack, sizeof(basis_value*) * stack_cap);
                    edge_index = (size_t*)realloc(edge_index, sizeof(size_t) * stack_cap);
                }
                stack[stack_ptr] = child;
                edge_index[stack_ptr] = 0;
                stack_ptr++;
            }
        } else {
            if (*order_size >= *order_capacity) {
                *order_capacity *= 2;
                *topological_order = (basis_value**)realloc(*topological_order, sizeof(basis_value*) * (*order_capacity));
            }
            (*topological_order)[(*order_size)++] = current;
            stack_ptr--;
        }
    }
    free(stack);
    free(edge_index);
}

void basis_value_zero_gradient(basis_value* root_value) {
    if (!root_value) return;
    size_t order_capacity = 16;
    size_t order_size = 0;
    basis_value** topological_order = (basis_value**)malloc(sizeof(basis_value*) * order_capacity);
    build_topological_order(root_value, &topological_order, &order_size, &order_capacity);
    for (size_t i = 0; i < order_size; i++) {
        topological_order[i]->gradient = 0.0;
        topological_order[i]->visited = false;
    }
    free(topological_order);
}

void basis_value_backward_propagation(basis_value* root_value) {
    if (!root_value) return;
    size_t order_capacity = 16;
    size_t order_size = 0;
    basis_value** topological_order = (basis_value**)malloc(sizeof(basis_value*) * order_capacity);
    build_topological_order(root_value, &topological_order, &order_size, &order_capacity);
    for (size_t i = 0; i < order_size; i++) {
        topological_order[i]->gradient = 0.0;
        topological_order[i]->visited = false;
    }
    root_value->gradient = 1.0;
    for (int i = (int)order_size - 1; i >= 0; i--) {
        if (topological_order[i]->backward_function) {
            topological_order[i]->backward_function(topological_order[i]);
        }
    }
    free(topological_order);
}

void basis_value_free(basis_value* target_value) {
    if (!target_value) return;

    // Phase L: Iterative DAG Teardown to prevent stack overflow on deep graphs
    size_t stack_cap = 256;
    basis_value** stack = (basis_value**)malloc(sizeof(basis_value*) * stack_cap);
    size_t stack_ptr = 0;

    stack[stack_ptr++] = target_value;

    while (stack_ptr > 0) {
        basis_value* current = stack[--stack_ptr];
        current->reference_count--;

        if (current->reference_count > 0) continue;

        if (current->previous_nodes) {
            for (size_t i = 0; i < current->previous_node_count; i++) {
                if (current->previous_nodes[i]) {
                    if (stack_ptr >= stack_cap) {
                        stack_cap *= 2;
                        stack = (basis_value**)realloc(stack, sizeof(basis_value*) * stack_cap);
                    }
                    stack[stack_ptr++] = current->previous_nodes[i];
                }
            }
            free(current->previous_nodes);
        }
        if (current->operation) free(current->operation);
        free(current);
    }
    free(stack);
}
