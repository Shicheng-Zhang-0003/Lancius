#include "basis/stage1_atomic/scalar.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

static char* value_string_duplicate(const char* source) {
    if (!source) return NULL;
    size_t len = strlen(source) + 1;
    char* copy = (char*)malloc(len);
    if (copy) memcpy(copy, source, len);
    return copy;
}

value* value_new (double initial_data) {
    value* new_value = (value*) malloc (sizeof (value));
    new_value -> data = initial_data;
    new_value -> gradient = 0.0;
    new_value -> backward_function = NULL;
    new_value -> previous_nodes = NULL;
    new_value -> previous_node_count = 0;
    new_value -> operation = value_string_duplicate("");
    new_value -> constant_value = 0.0;
    new_value -> reference_count = 1;
    new_value -> visited = false;
    return new_value;
}

void value_retain (value* target_value) {
    if (target_value) {target_value -> reference_count ++;}
}

static value* value_operation_initialize (value* first_operand, value* second_operand, char* operation, size_t operand_count) {
    value* output_value = value_new (0.0);
    free(output_value -> operation);
    output_value -> operation = value_string_duplicate(operation);
    output_value -> previous_node_count = operand_count;
    output_value -> previous_nodes = (value **) malloc (sizeof (value*) * operand_count);
    if (first_operand) {output_value -> previous_nodes [0] = first_operand;}
    if ((second_operand) && (operand_count > 1)) {output_value -> previous_nodes [1] = second_operand;}
    return output_value;
}

static void backward_addition (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += 1.0 * output_value -> gradient;
    output_value -> previous_nodes [1] -> gradient += 1.0 * output_value -> gradient;
}

value* value_addition (value* first_operand, value* second_operand) {
    value* output_value = value_operation_initialize (first_operand, second_operand, "+", 2);
    output_value -> data = first_operand -> data + second_operand -> data;
    output_value -> backward_function = backward_addition;
    value_retain (first_operand);
    value_retain (second_operand);
    return output_value;
}

static void backward_multiplication (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += output_value -> previous_nodes [1] -> data * output_value -> gradient;
    output_value -> previous_nodes [1] -> gradient += output_value -> previous_nodes [0] -> data * output_value -> gradient;
}

value* value_multiplication (value* first_operand, value* second_operand) {
    value* output_value = value_operation_initialize (first_operand, second_operand, "*", 2);
    output_value -> data = first_operand -> data * second_operand -> data;
    output_value -> backward_function = backward_multiplication;
    value_retain (first_operand);
    value_retain (second_operand);
    return output_value;
}

static void backward_power (value* output_value) {
    double base = output_value -> previous_nodes [0] -> data;
    double exponent = output_value -> constant_value;
    output_value -> previous_nodes [0] -> gradient += (exponent * pow (base, exponent - 1.0)) * output_value -> gradient;
}

value* value_power (value* base, double exponent) {
    value* output_value = value_operation_initialize (base, NULL, "**", 1);
    output_value -> data = pow (base -> data, exponent);
    output_value -> constant_value = exponent;
    output_value -> backward_function = backward_power;
    value_retain (base);
    return output_value;
}

static void backward_exponential (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += output_value -> data * output_value -> gradient;
}

value* value_exponential (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "exp", 1);
    output_value -> data = exp (input_value -> data);
    output_value -> backward_function = backward_exponential;
    value_retain (input_value);
    return output_value;
}

static void backward_logarithm (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += (1.0 / output_value -> previous_nodes [0] -> data) * output_value -> gradient;
}

value* value_logarithm (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "log", 1);
    output_value -> data = log (input_value -> data);
    output_value -> backward_function = backward_logarithm;
    value_retain (input_value);
    return output_value;
}

static void backward_sine (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += cos (output_value -> previous_nodes [0] -> data) * output_value -> gradient;
}

value* value_sine (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "sin", 1);
    output_value -> data = sin (input_value -> data);
    output_value -> backward_function = backward_sine;
    value_retain (input_value);
    return output_value;
}

static void backward_cosine (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += -sin (output_value -> previous_nodes [0] -> data) * output_value -> gradient;
}

value* value_cosine (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "cos", 1);
    output_value -> data = cos (input_value -> data);
    output_value -> backward_function = backward_cosine;
    value_retain (input_value);
    return output_value;
}

static void backward_rectified_linear_unit (value* output_value) {
    double gradient_multiplier;
    if (output_value -> data > 0) {
        gradient_multiplier = 1.0;
    } else {
        gradient_multiplier = 0.0;
    }
    output_value -> previous_nodes [0] -> gradient += gradient_multiplier * output_value -> gradient;
}

value* value_rectified_linear_unit (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "ReLU", 1);
    if (input_value -> data > 0) {
        output_value -> data = input_value -> data;
    } else {
        output_value -> data = 0.0;
    }
    output_value -> backward_function = backward_rectified_linear_unit;
    value_retain (input_value);
    return output_value;
}

static void build_topological_order (value* current_node, value *** topological_order, size_t* order_size, size_t* order_capacity) {
    if (current_node->visited) return;
    current_node->visited = true;

    for (size_t i = 0; i < current_node->previous_node_count; i++) {
        build_topological_order(current_node->previous_nodes[i], topological_order, order_size, order_capacity);
    }

    if (*order_size >= *order_capacity) {
        *order_capacity *= 2;
        *topological_order = (value**)realloc(*topological_order, sizeof(value*) * (*order_capacity));
    }
    (*topological_order)[(*order_size)++] = current_node;
}

void value_zero_gradient (value* root_value) {
    size_t order_capacity = 16;
    size_t order_size = 0;
    value** topological_order = (value**)malloc(sizeof(value*) * order_capacity);

    build_topological_order(root_value, &topological_order, &order_size, &order_capacity);

    for (size_t i = 0; i < order_size; i++) {
        topological_order[i]->gradient = 0.0;
        topological_order[i]->visited = false;
    }
    free(topological_order);
}

void value_backward_propagation (value* root_value) {
    size_t order_capacity = 16;
    size_t order_size = 0;
    value** topological_order = (value**)malloc(sizeof(value*) * order_capacity);

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

void value_free (value* target_value) {
    if (!target_value) {return;}
    target_value -> reference_count --;
    if (target_value -> reference_count > 0) {return;}

    if (target_value -> previous_nodes) {
        for (size_t i = 0; i < target_value -> previous_node_count; i ++) {value_free (target_value -> previous_nodes [i]);}
        free (target_value -> previous_nodes);
    }
    if (target_value -> operation) {
        free (target_value -> operation);
    }
    free (target_value);
}
