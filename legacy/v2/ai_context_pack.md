# PROJECT BUNDLE: v2
Root Directory: /home/magi-01/Desktop/work/programming/llm_maths/v2
Generated: Sat Jun 20 01:34:01 PM MDT 2026

## 1. DIRECTORY HIERARCHY
```text
v2/
├── include/
│   └── basis/
│       ├── stage1_atomic/
│       │   └── scalar.h
│       ├── stage2_linear/
│       │   └── tensor.h
│       ├── stage3_sequence/
│       │   └── sequence.h
│       ├── stage4_learning/
│       │   ├── learning.h
│       │   └── optim.h
│       ├── stage5_unified/
│       │   ├── compiler.h
│       │   ├── geometry.h
│       │   └── symbolic.h
│       ├── compiler.h
│       ├── geometry.h
│       ├── learning.h
│       ├── optim.h
│       ├── scalar.h
│       ├── sequence.h
│       ├── symbolic.h
│       └── tensor.h
├── src/
│   └── basis/
│       ├── stage1_atomic/
│       │   ├── scalar.c
│       │   └── scalar.o
│       ├── stage2_linear/
│       │   ├── tensor.c
│       │   └── tensor.o
│       ├── stage3_sequence/
│       │   ├── sequence.c
│       │   └── sequence.o
│       ├── stage4_learning/
│       │   ├── learning.c
│       │   ├── learning.o
│       │   ├── optim.c
│       │   └── optim.o
│       └── stage5_unified/
│           ├── compiler.c
│           ├── compiler.o
│           ├── geometry.c
│           ├── geometry.o
│           ├── symbolic.c
│           └── symbolic.o
├── tests/
│   ├── test_attention.c
│   ├── test_compiler.c
│   ├── test_frontier.c
│   ├── test_grand_unified.c
│   ├── test_scalar.c
│   ├── test_symbolic.c
│   └── test_tensor.c
├── makefile
├── test_attention
├── test_compiler
├── test_frontier
├── test_grand_unified
├── test_scalar
├── test_symbolic
└── test_tensor
```

## 2. FILE CONTENTS

### FILE: include/basis/compiler.h
Location: `include/basis/compiler.h`
```h
#include "basis/stage5_unified/compiler.h"

```

---

### FILE: include/basis/geometry.h
Location: `include/basis/geometry.h`
```h
#include "basis/stage5_unified/geometry.h"

```

---

### FILE: include/basis/learning.h
Location: `include/basis/learning.h`
```h
#include "basis/stage4_learning/learning.h"

```

---

### FILE: include/basis/optim.h
Location: `include/basis/optim.h`
```h
#include "basis/stage4_learning/optim.h"

```

---

### FILE: include/basis/scalar.h
Location: `include/basis/scalar.h`
```h
#include "basis/stage1_atomic/scalar.h"

```

---

### FILE: include/basis/sequence.h
Location: `include/basis/sequence.h`
```h
#include "basis/stage3_sequence/sequence.h"

```

---

### FILE: include/basis/stage1_atomic/scalar.h
Location: `include/basis/stage1_atomic/scalar.h`
```h
#ifndef basis_stage1_scalar_h
#define basis_stage1_scalar_h
#include <stdlib.h>
typedef struct value {
    double data;
    double gradient;
    void (*backward_function) (struct value* output_value);
    struct value** previous_nodes;
    size_t previous_node_count;
    char* operation;
    double constant_value;
    size_t reference_count;
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

```

---

### FILE: include/basis/stage2_linear/tensor.h
Location: `include/basis/stage2_linear/tensor.h`
```h
#ifndef basis_stage2_tensor_h
#define basis_stage2_tensor_h
#include "basis/stage1_atomic/scalar.h"
typedef struct tensor {
    value** data;
    size_t row_count;
    size_t column_count;
} tensor;
tensor* tensor_new (size_t row_count, size_t column_count);
void tensor_free (tensor* target_tensor);
void tensor_fill (tensor* target_tensor, double fill_value);
void tensor_set (tensor* target_tensor, size_t row_index, size_t column_index, double data_value);
tensor* tensor_addition (tensor* first_tensor, tensor* second_tensor);
tensor* tensor_matrix_multiplication (tensor* first_tensor, tensor* second_tensor);
tensor* tensor_scalar_multiplication (tensor* target_tensor, double scalar_value);
tensor* tensor_rectified_linear_unit (tensor* target_tensor);
void tensor_print (tensor* target_tensor, const char* name);
#endif

```

---

### FILE: include/basis/stage3_sequence/sequence.h
Location: `include/basis/stage3_sequence/sequence.h`
```h
#ifndef basis_stage3_sequence_h
#define basis_stage3_sequence_h
#include "basis/stage2_linear/tensor.h"
tensor* tensor_transpose (tensor* target_tensor);
tensor* tensor_softmax (tensor* target_tensor);
tensor* tensor_attention (tensor* query_tensor, tensor* key_tensor, tensor* value_tensor);
#endif

```

---

### FILE: include/basis/stage4_learning/learning.h
Location: `include/basis/stage4_learning/learning.h`
```h
#ifndef basis_stage4_learning_h
#define basis_stage4_learning_h
#include "basis/stage2_linear/tensor.h"
#include "basis/stage4_learning/optim.h"
tensor* tensor_layer_normalization (tensor* target_tensor, double epsilon);
tensor* tensor_rotary_positional_embedding (tensor* target_tensor, int position, double base);
value* tensor_sum (tensor* target_tensor);
tensor* tensor_logarithm (tensor* target_tensor);
tensor* tensor_multiplication (tensor* first_tensor, tensor* second_tensor);
#endif

```

---

### FILE: include/basis/stage4_learning/optim.h
Location: `include/basis/stage4_learning/optim.h`
```h
#ifndef basis_stage4_optim_h
#define basis_stage4_optim_h
#include "basis/stage2_linear/tensor.h"
typedef struct adam {
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    size_t time_step;
    tensor* first_moment;
    tensor* second_moment;
} adam;
adam* adam_new (size_t row_count, size_t column_count, double learning_rate);
void adam_optimization_step (adam* optimizer, tensor* weight_tensor);
void adam_free (adam* optimizer);
#endif

```

---

### FILE: include/basis/stage5_unified/compiler.h
Location: `include/basis/stage5_unified/compiler.h`
```h
#ifndef basis_stage5_unified_compiler_h
#define basis_stage5_unified_compiler_h
#include "basis/stage1_atomic/scalar.h"
#include "basis/stage5_unified/symbolic.h"
typedef struct {
    char* name;
    value* value_pointer;
} variable_mapping;
typedef struct {
    variable_mapping* variable_mappings;
    size_t variable_mapping_count;
} compiler;
compiler* compiler_new (void);
void compiler_map (compiler* compiler_instance, char* variable_name, value* value_pointer);
value* compiler_compile (compiler* compiler_instance, symbol* symbol_node);
void compiler_free (compiler* compiler_instance);
#endif

```

---

### FILE: include/basis/stage5_unified/geometry.h
Location: `include/basis/stage5_unified/geometry.h`
```h
#ifndef basis_stage5_unified_geometry_h
#define basis_stage5_unified_geometry_h
#include "basis/stage2_linear/tensor.h"
typedef struct {
    tensor* matrix;
} metric;
metric* metric_fisher_information (tensor* weight_tensor, tensor* input_tensor, tensor* output_tensor);
void metric_free (metric* target_metric);
#endif

```

---

### FILE: include/basis/stage5_unified/symbolic.h
Location: `include/basis/stage5_unified/symbolic.h`
```h
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

```

---

### FILE: include/basis/symbolic.h
Location: `include/basis/symbolic.h`
```h
#include "basis/stage5_unified/symbolic.h"

```

---

### FILE: include/basis/tensor.h
Location: `include/basis/tensor.h`
```h
#include "basis/stage2_linear/tensor.h"

```

---

### FILE: makefile
Location: `makefile`
```text
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -I./include
LDFLAGS = -lm

SRCS = src/basis/stage1_atomic/scalar.c \
       src/basis/stage2_linear/tensor.c \
       src/basis/stage3_sequence/sequence.c \
       src/basis/stage4_learning/learning.c \
       src/basis/stage4_learning/optim.c \
       src/basis/stage5_unified/symbolic.c \
       src/basis/stage5_unified/compiler.c \
       src/basis/stage5_unified/geometry.c

OBJS = $(SRCS:.c=.o)

TESTS = test_scalar test_tensor test_attention test_symbolic test_compiler test_frontier test_grand_unified

all: $(TESTS)

test_scalar: tests/test_scalar.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_tensor: tests/test_tensor.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_attention: tests/test_attention.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_symbolic: tests/test_symbolic.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_compiler: tests/test_compiler.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_frontier: tests/test_frontier.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_grand_unified: tests/test_grand_unified.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TESTS)

.PHONY: all clean

```

---

### FILE: src/basis/stage1_atomic/scalar.c
Location: `src/basis/stage1_atomic/scalar.c`
```cpp
#include "basis/stage1_atomic/scalar.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
value* value_new (double initial_data) {
    value* new_value = (value*) malloc (sizeof (value));
    new_value -> data = initial_data;
    new_value -> gradient = 0.0;
    new_value -> backward_function = NULL;
    new_value -> previous_nodes = NULL;
    new_value -> previous_node_count = 0;
    new_value -> operation = "";
    new_value -> constant_value = 0.0;
    new_value -> reference_count = 1;
    return new_value;
} void value_retain (value* target_value) {
    if (target_value) {target_value -> reference_count ++;}
} static value* value_operation_initialize (value* first_operand, value* second_operand, char* operation, size_t operand_count) {
    value* output_value = value_new (0.0);
    output_value -> operation = operation;
    output_value -> previous_node_count = operand_count;
    output_value -> previous_nodes = (value **) malloc (sizeof (value*) * operand_count);
    if (first_operand) {output_value -> previous_nodes [0] = first_operand;}
    if ((second_operand) && (operand_count > 1)) {output_value -> previous_nodes [1] = second_operand;}
    return output_value;
} static void backward_addition (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += 1.0 * output_value -> gradient;
    output_value -> previous_nodes [1] -> gradient += 1.0 * output_value -> gradient;
} value* value_addition (value* first_operand, value* second_operand) {
    value* output_value = value_operation_initialize (first_operand, second_operand, "+", 2);
    output_value -> data = first_operand -> data + second_operand -> data;
    output_value -> backward_function = backward_addition;
    value_retain (first_operand);
    value_retain (second_operand);
    return output_value;
} static void backward_multiplication (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += output_value -> previous_nodes [1] -> data * output_value -> gradient;
    output_value -> previous_nodes [1] -> gradient += output_value -> previous_nodes [0] -> data * output_value -> gradient;
} value* value_multiplication (value* first_operand, value* second_operand) {
    value* output_value = value_operation_initialize (first_operand, second_operand, "*", 2);
    output_value -> data = first_operand -> data * second_operand -> data;
    output_value -> backward_function = backward_multiplication;
    value_retain (first_operand);
    value_retain (second_operand);
    return output_value;
} static void backward_power (value* output_value) {
    double base = output_value -> previous_nodes [0] -> data;
    double exponent = output_value -> constant_value;
    output_value -> previous_nodes [0] -> gradient += (exponent * pow (base, exponent - 1.0)) * output_value -> gradient;
} value* value_power (value* base, double exponent) {
    value* output_value = value_operation_initialize (base, NULL, "**", 1);
    output_value -> data = pow (base -> data, exponent);
    output_value -> constant_value = exponent;
    output_value -> backward_function = backward_power;
    value_retain (base);
    return output_value;
} static void backward_exponential (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += output_value -> data * output_value -> gradient;
} value* value_exponential (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "exp", 1);
    output_value -> data = exp (input_value -> data);
    output_value -> backward_function = backward_exponential;
    value_retain (input_value);
    return output_value;
} static void backward_logarithm (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += (1.0 / output_value -> previous_nodes [0] -> data) * output_value -> gradient;
} value* value_logarithm (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "log", 1);
    output_value -> data = log (input_value -> data);
    output_value -> backward_function = backward_logarithm;
    value_retain (input_value);
    return output_value;
} static void backward_sine (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += cos (output_value -> previous_nodes [0] -> data) * output_value -> gradient;
} value* value_sine (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "sin", 1);
    output_value -> data = sin (input_value -> data);
    output_value -> backward_function = backward_sine;
    value_retain (input_value);
    return output_value;
} static void backward_cosine (value* output_value) {
    output_value -> previous_nodes [0] -> gradient += -sin (output_value -> previous_nodes [0] -> data) * output_value -> gradient;
} value* value_cosine (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "cos", 1);
    output_value -> data = cos (input_value -> data);
    output_value -> backward_function = backward_cosine;
    value_retain (input_value);
    return output_value;
} static void backward_rectified_linear_unit (value* output_value) {
    double gradient_multiplier;
    if (output_value -> data > 0) {
        gradient_multiplier = 1.0;
    } else {
        gradient_multiplier = 0.0;
    }
    output_value -> previous_nodes [0] -> gradient += gradient_multiplier * output_value -> gradient;
} value* value_rectified_linear_unit (value* input_value) {
    value* output_value = value_operation_initialize (input_value, NULL, "ReLU", 1);
    if (input_value -> data > 0) {
        output_value -> data = input_value -> data;
    } else {
        output_value -> data = 0.0;
    }
    output_value -> backward_function = backward_rectified_linear_unit;
    value_retain (input_value);
    return output_value;
} static void build_topological_order (value* current_node, value *** topological_order, size_t* order_size, size_t* order_capacity, value *** visited_nodes, size_t* visited_count, size_t* visited_capacity) {
    for (size_t i = 0; i < *visited_count; i ++) {if ((*visited_nodes) [i] == current_node) {return;}}
    if (*visited_count >= *visited_capacity) {
        *visited_capacity *= 2;
        *visited_nodes = (value **) realloc (*visited_nodes, sizeof (value*) * (*visited_capacity));
    }
    (*visited_nodes) [(*visited_count) ++] = current_node;
    for (size_t i = 0; i < current_node -> previous_node_count; i ++) {build_topological_order (current_node -> previous_nodes [i], topological_order, order_size, order_capacity, visited_nodes, visited_count, visited_capacity);}
    if (*order_size >= *order_capacity) {
        *order_capacity *= 2;
        *topological_order = (value **) realloc (*topological_order, sizeof (value*) * (*order_capacity));
    }
    (*topological_order) [(*order_size) ++] = current_node;
} void value_zero_gradient (value* root_value) {
    size_t order_capacity = 128;
    size_t order_size = 0;
    value** topological_order = (value **) malloc (sizeof (value*) * order_capacity);
    size_t visited_capacity = 128;
    size_t visited_count = 0;
    value** visited_nodes = (value **) malloc (sizeof (value*) * visited_capacity);
    build_topological_order (root_value, &topological_order, &order_size, &order_capacity, &visited_nodes, &visited_count, &visited_capacity);
    for (size_t i = 0; i < order_size; i ++) {topological_order [i] -> gradient = 0.0;}
    free (topological_order);
    free (visited_nodes);
} void value_backward_propagation (value* root_value) {
    size_t order_capacity = 128;
    size_t order_size = 0;
    value** topological_order = (value **) malloc (sizeof (value*) * order_capacity);
    size_t visited_capacity = 128;
    size_t visited_count = 0;
    value** visited_nodes = (value **) malloc (sizeof (value*) * visited_capacity);
    build_topological_order (root_value, &topological_order, &order_size, &order_capacity, &visited_nodes, &visited_count, &visited_capacity);
    value_zero_gradient (root_value);
    root_value -> gradient = 1.0;
    for (int i = (int) order_size - 1; i >= 0; i --) {if (topological_order [i] -> backward_function) {topological_order [i] -> backward_function (topological_order [i]);}}
    free (topological_order);
    free (visited_nodes);
} void value_free (value* target_value) {
    if (!target_value) {return;}
    target_value -> reference_count --;
    if (target_value -> reference_count > 0) {return;}
    if (target_value -> previous_nodes) {
        for (size_t i = 0; i < target_value -> previous_node_count; i ++) {value_free (target_value -> previous_nodes [i]);}
        free (target_value -> previous_nodes);
    }
    free (target_value);
}

```

---

### FILE: src/basis/stage2_linear/tensor.c
Location: `src/basis/stage2_linear/tensor.c`
```cpp
#include "basis/stage2_linear/tensor.h"
#include <stdio.h>
#include <assert.h>
tensor* tensor_new (size_t row_count, size_t column_count) {
    tensor* new_tensor = (tensor*) malloc (sizeof (tensor));
    new_tensor -> row_count = row_count;
    new_tensor -> column_count = column_count;
    new_tensor -> data = (value **) malloc (sizeof (value*) * row_count * column_count);
    for (size_t i = 0; i < row_count * column_count; i ++) {new_tensor -> data [i] = value_new (0.0);}
    return new_tensor;
} void tensor_free (tensor* target_tensor) {
    if (!target_tensor) {return;}
    if (target_tensor -> data) {
        for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {value_free (target_tensor -> data [i]);}
        free (target_tensor -> data);
    }
    free (target_tensor);
} void tensor_fill (tensor* target_tensor, double fill_value) {
    if (!target_tensor) {return;}
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {target_tensor -> data [i] -> data = fill_value;}
} void tensor_set (tensor* target_tensor, size_t row_index, size_t column_index, double data_value) {
    if ((!target_tensor) || (row_index >= target_tensor -> row_count) || (column_index >= target_tensor -> column_count)) {return;}
    target_tensor -> data [row_index * target_tensor -> column_count + column_index] -> data = data_value;
} tensor* tensor_addition (tensor* first_tensor, tensor* second_tensor) {
    if ((!first_tensor) || (!second_tensor) || ((first_tensor -> row_count) != (second_tensor -> row_count)) || ((first_tensor -> column_count) != (second_tensor -> column_count))) {return NULL;}
    tensor* output_tensor = tensor_new (first_tensor -> row_count, first_tensor -> column_count);
    for (size_t i = 0; i < first_tensor -> row_count * first_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_addition (first_tensor -> data [i], second_tensor -> data [i]);
    }
    return output_tensor;
} tensor* tensor_matrix_multiplication (tensor* first_tensor, tensor* second_tensor) {
    if ((!first_tensor) || (!second_tensor) || (first_tensor -> column_count != second_tensor -> row_count)) {return NULL;}
    tensor* output_tensor = tensor_new (first_tensor -> row_count, second_tensor -> column_count);
    for (size_t i = 0; i < first_tensor -> row_count; i ++) {
        for (size_t j = 0; j < second_tensor -> column_count; j ++) {
            value* sum = value_new (0.0);
            for (size_t k = 0; k < first_tensor -> column_count; k ++) {
                value* product = value_multiplication (first_tensor -> data [i * first_tensor -> column_count + k], second_tensor -> data [k * second_tensor -> column_count + j]);
                value* new_sum = value_addition (sum, product);
                value_free (sum);
                value_free (product);
                sum = new_sum;
            }
            value_free (output_tensor -> data [i * second_tensor -> column_count + j]);
            output_tensor -> data [i * second_tensor -> column_count + j] = sum;
        }
    }
    return output_tensor;
} tensor* tensor_scalar_multiplication (tensor* target_tensor, double scalar_value) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    value* scalar = value_new (scalar_value);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_multiplication (target_tensor -> data [i], scalar);
    }
    value_free (scalar);
    return output_tensor;
} tensor* tensor_rectified_linear_unit (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_rectified_linear_unit (target_tensor -> data [i]);
    }
    return output_tensor;
} void tensor_print (tensor* target_tensor, const char* name) {
    if (!target_tensor) {return;}
    printf ("Tensor %s (%zu x %zu):\n", name, target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        printf ("  [ ");
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {printf ("%8.4f ", target_tensor -> data [i * target_tensor -> column_count + j] -> data);}
        printf ("]\n");
    }
}

```

---

### FILE: src/basis/stage3_sequence/sequence.c
Location: `src/basis/stage3_sequence/sequence.c`
```cpp
#include "basis/stage3_sequence/sequence.h"
#include <math.h>
#include <stdlib.h>
tensor* tensor_transpose (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = (tensor*) malloc (sizeof (tensor));
    output_tensor -> row_count = target_tensor -> column_count;
    output_tensor -> column_count = target_tensor -> row_count;
    output_tensor -> data = (value **) malloc (sizeof (value*) * output_tensor -> row_count * output_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* current_value = target_tensor -> data [i * target_tensor -> column_count + j];
            value_retain (current_value);
            output_tensor -> data [j * output_tensor -> column_count + i] = current_value;
        }
    }
    return output_tensor;
} tensor* tensor_softmax (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        double maximum_data = target_tensor -> data [i * target_tensor -> column_count] -> data;
        for (size_t j = 1; j < target_tensor -> column_count; j ++) {
            if (target_tensor -> data [i * target_tensor -> column_count + j] -> data > maximum_data) {maximum_data = target_tensor -> data [i * target_tensor -> column_count + j] -> data;}
        }
        value* negative_maximum = value_new (-maximum_data);
        value* sum = value_new (0.0);
        value** exponentials = (value **) malloc (sizeof (value*) * target_tensor -> column_count);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* shifted = value_addition (target_tensor -> data [i * target_tensor -> column_count + j], negative_maximum);
            exponentials [j] = value_exponential (shifted);
            value* new_sum = value_addition (sum, exponentials [j]);
            value_free (sum);
            value_free (shifted);
            sum = new_sum;
        }
        value_free (negative_maximum);
        value* inverse_sum = value_power (sum, -1.0);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value_free (output_tensor -> data [i * target_tensor -> column_count + j]);
            output_tensor -> data [i * target_tensor -> column_count + j] = value_multiplication (exponentials [j], inverse_sum);
            value_free (exponentials [j]);
        }
        value_free (sum);
        value_free (inverse_sum);
        free (exponentials);
    }
    return output_tensor;
} tensor* tensor_attention (tensor* query_tensor, tensor* key_tensor, tensor* value_tensor) {
    if ((!query_tensor) || (!key_tensor) || (!value_tensor)) {return NULL;}
    if ((query_tensor -> column_count != key_tensor -> column_count) || (key_tensor -> row_count != value_tensor -> row_count)) {return NULL;}
    tensor* transposed_key_tensor = tensor_transpose (key_tensor);
    if (!transposed_key_tensor) {return NULL;}
    tensor* query_key_product = tensor_matrix_multiplication (query_tensor, transposed_key_tensor);
    if (!query_key_product) {
        tensor_free (transposed_key_tensor);
        return NULL;
    }
    tensor* scaled_tensor = tensor_scalar_multiplication (query_key_product, 1.0 / sqrt ((double) key_tensor -> column_count));
    if (!scaled_tensor) {
        tensor_free (query_key_product);
        tensor_free (transposed_key_tensor);
        return NULL;
    }
    tensor* attention_weights = tensor_softmax (scaled_tensor);
    if (!attention_weights) {
        tensor_free (scaled_tensor);
        tensor_free (query_key_product);
        tensor_free (transposed_key_tensor);
        return NULL;
    }
    tensor* output_tensor = tensor_matrix_multiplication (attention_weights, value_tensor);
    if (!output_tensor) {
        tensor_free (attention_weights);
        tensor_free (scaled_tensor);
        tensor_free (query_key_product);
        tensor_free (transposed_key_tensor);
        return NULL;
    }
    tensor_free (transposed_key_tensor);
    tensor_free (query_key_product);
    tensor_free (scaled_tensor);
    tensor_free (attention_weights);
    return output_tensor;
}

```

---

### FILE: src/basis/stage4_learning/learning.c
Location: `src/basis/stage4_learning/learning.c`
```cpp
#include "basis/stage4_learning/learning.h"
#include <math.h>
#include <stdlib.h>
tensor* tensor_layer_normalization (tensor* target_tensor, double epsilon) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        value* sum = value_new (0.0);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* new_sum = value_addition (sum, target_tensor -> data [i * target_tensor -> column_count + j]);
            value_free (sum);
            sum = new_sum;
        }
        value* mean = value_multiplication (sum, value_new (1.0 / target_tensor -> column_count));
        value_free (sum);
        value* variance_sum = value_new (0.0);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* negative_mean = value_multiplication (mean, value_new (-1.0));
            value* difference = value_addition (target_tensor -> data [i * target_tensor -> column_count + j], negative_mean);
            value_free (negative_mean);
            value* squared = value_power (difference, 2.0);
            value_free (difference);
            value* new_variance_sum = value_addition (variance_sum, squared);
            value_free (variance_sum);
            value_free (squared);
            variance_sum = new_variance_sum;
        }
        value* variance = value_multiplication (variance_sum, value_new (1.0 / target_tensor -> column_count));
        value_free (variance_sum);
        value* epsilon_value = value_new (epsilon);
        value* variance_epsilon = value_addition (variance, epsilon_value);
        value_free (variance);
        value* inverse_standard_deviation = value_power (variance_epsilon, -0.5);
        value_free (variance_epsilon);
        value_free (epsilon_value);
        for (size_t j = 0; j < target_tensor -> column_count; j ++) {
            value* negative_mean = value_multiplication (mean, value_new (-1.0));
            value* difference = value_addition (target_tensor -> data [i * target_tensor -> column_count + j], negative_mean);
            value_free (negative_mean);
            value_free (output_tensor -> data [i * target_tensor -> column_count + j]);
            output_tensor -> data [i * target_tensor -> column_count + j] = value_multiplication (difference, inverse_standard_deviation);
            value_free (difference);
        }
        value_free (mean);
        value_free (inverse_standard_deviation);
    }
    return output_tensor;
} tensor* tensor_rotary_positional_embedding (tensor* target_tensor, int position, double base) {
    if ((!target_tensor) || (target_tensor -> column_count % 2 != 0)) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count; i ++) {
        for (size_t j = 0; j < target_tensor -> column_count; j += 2) {
            double theta = pow (base, -((double) j / target_tensor -> column_count));
            double angle = position * theta;
            value* angle_value = value_new (angle);
            value* cosine_value = value_cosine (angle_value);
            value* sine_value = value_sine (angle_value);
            value_free (angle_value);
            value* first_element = target_tensor -> data [i * target_tensor -> column_count + j];
            value* second_element = target_tensor -> data [i * target_tensor -> column_count + j + 1];
            value_free (output_tensor -> data [i * target_tensor -> column_count + j]);
            value_free (output_tensor -> data [i * target_tensor -> column_count + j + 1]);
            value* negative_sine = value_multiplication (sine_value, value_new (-1.0));
            value* term1 = value_multiplication (first_element, cosine_value);
            value* term2 = value_multiplication (second_element, negative_sine);
            output_tensor -> data [i * target_tensor -> column_count + j] = value_addition (term1, term2);
            value_free (term1);
            value_free (term2);
            value_free (negative_sine);
            value* term3 = value_multiplication (first_element, sine_value);
            value* term4 = value_multiplication (second_element, cosine_value);
            output_tensor -> data [i * target_tensor -> column_count + j + 1] = value_addition (term3, term4);
            value_free (term3);
            value_free (term4);
            value_free (cosine_value);
            value_free (sine_value);
        }
    }
    return output_tensor;
} value* tensor_sum (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    value* sum = value_new (0.0);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value* new_sum = value_addition (sum, target_tensor -> data [i]);
        value_free (sum);
        sum = new_sum;
    }
    return sum;
} tensor* tensor_logarithm (tensor* target_tensor) {
    if (!target_tensor) {return NULL;}
    tensor* output_tensor = tensor_new (target_tensor -> row_count, target_tensor -> column_count);
    for (size_t i = 0; i < target_tensor -> row_count * target_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_logarithm (target_tensor -> data [i]);
    }
    return output_tensor;
} tensor* tensor_multiplication (tensor* first_tensor, tensor* second_tensor) {
    if ((!first_tensor) || (!second_tensor) || ((first_tensor -> row_count) != (second_tensor -> row_count)) || ((first_tensor -> column_count) != (second_tensor -> column_count))) {return NULL;}
    tensor* output_tensor = tensor_new (first_tensor -> row_count, first_tensor -> column_count);
    for (size_t i = 0; i < first_tensor -> row_count * first_tensor -> column_count; i ++) {
        value_free (output_tensor -> data [i]);
        output_tensor -> data [i] = value_multiplication (first_tensor -> data [i], second_tensor -> data [i]);
    }
    return output_tensor;
}

```

---

### FILE: src/basis/stage4_learning/optim.c
Location: `src/basis/stage4_learning/optim.c`
```cpp
#include "basis/stage4_learning/optim.h"
#include <math.h>
#include <stdlib.h>
adam* adam_new (size_t row_count, size_t column_count, double learning_rate) {
    adam* optimizer = (adam*) malloc (sizeof (adam));
    optimizer -> learning_rate = learning_rate;
    optimizer -> beta1 = 0.9;
    optimizer -> beta2 = 0.999;
    optimizer -> epsilon = 1e-8;
    optimizer -> time_step = 0;
    optimizer -> first_moment = tensor_new (row_count, column_count);
    optimizer -> second_moment = tensor_new (row_count, column_count);
    tensor_fill (optimizer -> first_moment, 0.0);
    tensor_fill (optimizer -> second_moment, 0.0);
    return optimizer;
} void adam_optimization_step (adam* optimizer, tensor* weight_tensor) {
    optimizer -> time_step ++;
    for (size_t i = 0; i < weight_tensor -> row_count * weight_tensor -> column_count; i ++) {
        double gradient = weight_tensor -> data [i] -> gradient;
        optimizer -> first_moment -> data [i] -> data = optimizer -> beta1 * optimizer -> first_moment -> data [i] -> data + (1.0 - optimizer -> beta1) * gradient;
        optimizer -> second_moment -> data [i] -> data = optimizer -> beta2 * optimizer -> second_moment -> data [i] -> data + (1.0 - optimizer -> beta2) * gradient * gradient;
        double first_moment_hat = optimizer -> first_moment -> data [i] -> data / (1.0 - pow (optimizer -> beta1, optimizer -> time_step));
        double second_moment_hat = optimizer -> second_moment -> data [i] -> data / (1.0 - pow (optimizer -> beta2, optimizer -> time_step));
        weight_tensor -> data [i] -> data -= optimizer -> learning_rate * first_moment_hat / (sqrt (second_moment_hat) + optimizer -> epsilon);
        weight_tensor -> data [i] -> gradient = 0.0;
    }
} void adam_free (adam* optimizer) {
    if (!optimizer) {return;}
    tensor_free (optimizer -> first_moment);
    tensor_free (optimizer -> second_moment);
    free (optimizer);
}

```

---

### FILE: src/basis/stage5_unified/compiler.c
Location: `src/basis/stage5_unified/compiler.c`
```cpp
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

```

---

### FILE: src/basis/stage5_unified/geometry.c
Location: `src/basis/stage5_unified/geometry.c`
```cpp
#include "basis/stage5_unified/geometry.h"
#include "basis/stage3_sequence/sequence.h"
#include <stdlib.h>
metric* metric_fisher_information (tensor* weight_tensor, tensor* input_tensor, tensor* output_tensor) {
    (void) weight_tensor; (void) output_tensor;
    if (!input_tensor) {return NULL;}
    tensor* transposed_input = tensor_transpose (input_tensor);
    if (!transposed_input) {return NULL;}
    tensor* fisher_matrix = tensor_matrix_multiplication (transposed_input, input_tensor);
    if (!fisher_matrix) {
        tensor_free (transposed_input);
        return NULL;
    }
    metric* new_metric = (metric*) malloc (sizeof (metric));
    if (!new_metric) {
        tensor_free (fisher_matrix);
        tensor_free (transposed_input);
        return NULL;
    }
    new_metric -> matrix = fisher_matrix;
    tensor_free (transposed_input);
    return new_metric;
} void metric_free (metric* target_metric) {
    if (!target_metric) {return;}
    tensor_free (target_metric -> matrix);
    free (target_metric);
}

```

---

### FILE: src/basis/stage5_unified/symbolic.c
Location: `src/basis/stage5_unified/symbolic.c`
```cpp
#include "basis/stage5_unified/symbolic.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
static char* symbol_string_duplicate (const char* source_string) {
    size_t length = strlen (source_string) + 1;
    char* copy = (char*) malloc (length);
    if (copy) {memcpy (copy, source_string, length);}
    return copy;
} static symbol* symbol_new (symbol_type type) {
    symbol* new_symbol = (symbol*) malloc (sizeof (symbol));
    new_symbol -> type = type;
    new_symbol -> name = NULL;
    new_symbol -> operation = NULL;
    new_symbol -> value = 0.0;
    new_symbol -> arguments = NULL;
    new_symbol -> argument_count = 0;
    return new_symbol;
} symbol* symbol_variable (char* variable_name) {
    symbol* new_symbol = symbol_new (symbol_type_variable);
    new_symbol -> name = symbol_string_duplicate (variable_name);
    return new_symbol;
} symbol* symbol_constant (double constant_value) {
    symbol* new_symbol = symbol_new (symbol_type_constant);
    new_symbol -> value = constant_value;
    return new_symbol;
} static symbol* symbol_operation_new (char* operation, size_t argument_count) {
    symbol* new_symbol = symbol_new (symbol_type_operation);
    new_symbol -> operation = symbol_string_duplicate (operation);
    new_symbol -> argument_count = argument_count;
    new_symbol -> arguments = (symbol **) malloc (sizeof (symbol*) * argument_count);
    return new_symbol;
} symbol* symbol_addition (symbol* first_operand, symbol* second_operand) {
    symbol* new_symbol = symbol_operation_new ("+", 2);
    new_symbol -> arguments [0] = first_operand;
    new_symbol -> arguments [1] = second_operand;
    return new_symbol;
} symbol* symbol_multiplication (symbol* first_operand, symbol* second_operand) {
    symbol* new_symbol = symbol_operation_new ("*", 2);
    new_symbol -> arguments [0] = first_operand;
    new_symbol -> arguments [1] = second_operand;
    return new_symbol;
} symbol* symbol_power (symbol* base, double exponent) {
    symbol* new_symbol = symbol_operation_new ("**", 1);
    new_symbol -> arguments [0] = base;
    new_symbol -> value = exponent;
    return new_symbol;
} symbol* symbol_logarithm (symbol* input_symbol) {
    symbol* new_symbol = symbol_operation_new ("log", 1);
    new_symbol -> arguments [0] = input_symbol;
    return new_symbol;
} symbol* symbol_exponential (symbol* input_symbol) {
    symbol* new_symbol = symbol_operation_new ("exp", 1);
    new_symbol -> arguments [0] = input_symbol;
    return new_symbol;
} symbol* symbol_copy (symbol* source_symbol) {
    if (!source_symbol) {return NULL;}
    symbol* output_symbol;
    if (source_symbol -> type == symbol_type_variable) {output_symbol = symbol_variable (source_symbol -> name);}
    else if (source_symbol -> type == symbol_type_constant) {output_symbol = symbol_constant (source_symbol -> value);}
    else {
        output_symbol = symbol_operation_new (source_symbol -> operation, source_symbol -> argument_count);
        output_symbol -> value = source_symbol -> value;
        for (size_t i = 0; i < source_symbol -> argument_count; i ++) {output_symbol -> arguments [i] = symbol_copy (source_symbol -> arguments [i]);}
    }
    return output_symbol;
} symbol* symbol_differentiation (symbol* source_symbol, char* variable_name) {
    if (source_symbol -> type == symbol_type_constant) {return symbol_constant (0.0);}
    if (source_symbol -> type == symbol_type_variable) {return (strcmp (source_symbol -> name, variable_name) == 0) ? symbol_constant (1.0) : symbol_constant (0.0);}
    if (source_symbol -> type == symbol_type_operation) {
        if (strcmp (source_symbol -> operation, "+") == 0) {return symbol_addition (symbol_differentiation (source_symbol -> arguments [0], variable_name), symbol_differentiation (source_symbol -> arguments [1], variable_name));}
        if (strcmp (source_symbol -> operation, "*") == 0) {return symbol_addition (symbol_multiplication (symbol_differentiation (source_symbol -> arguments [0], variable_name), symbol_copy (source_symbol -> arguments [1])), symbol_multiplication (symbol_copy (source_symbol -> arguments [0]), symbol_differentiation (source_symbol -> arguments [1], variable_name)));}
        if (strcmp (source_symbol -> operation, "**") == 0) {
            double exponent = source_symbol -> value;
            return symbol_multiplication (symbol_constant (exponent), symbol_multiplication (symbol_power (symbol_copy (source_symbol -> arguments [0]), exponent - 1.0), symbol_differentiation (source_symbol -> arguments [0], variable_name)));
        }
        if (strcmp (source_symbol -> operation, "exp") == 0) {return symbol_multiplication (symbol_exponential (symbol_copy (source_symbol -> arguments [0])), symbol_differentiation (source_symbol -> arguments [0], variable_name));}
        if (strcmp (source_symbol -> operation, "log") == 0) {return symbol_multiplication (symbol_differentiation (source_symbol -> arguments [0], variable_name), symbol_power (symbol_copy (source_symbol -> arguments [0]), -1.0));}
    }
    return symbol_constant (0.0);
} void symbol_free (symbol* target_symbol) {
    if (!target_symbol) {return;}
    if (target_symbol -> name) {free (target_symbol -> name);}
    if (target_symbol -> operation) {free (target_symbol -> operation);}
    for (size_t i = 0; i < target_symbol -> argument_count; i ++) {symbol_free (target_symbol -> arguments [i]);}
    if (target_symbol -> arguments) {free (target_symbol -> arguments);}
    free (target_symbol);
} void symbol_print (symbol* target_symbol) {
    if (!target_symbol) {printf ("NULL"); return;}
    if (target_symbol -> type == symbol_type_variable) {printf ("%s", target_symbol -> name);}
    else if (target_symbol -> type == symbol_type_constant) {printf ("%.2f", target_symbol -> value);}
    else if (target_symbol -> type == symbol_type_operation) {
        if (target_symbol -> argument_count == 2) {
            printf ("("); symbol_print (target_symbol -> arguments [0]); printf (" %s ", target_symbol -> operation); symbol_print (target_symbol -> arguments [1]); printf (")");
        } else {
            printf ("%s(", target_symbol -> operation); symbol_print (target_symbol -> arguments [0]);
            if (strcmp (target_symbol -> operation, "**") == 0) {printf (", %.2f", target_symbol -> value);}
            printf (")");
        }
    }
} symbol* symbol_simplify (symbol* source_symbol) {
    if (!source_symbol) {return NULL;}
    if ((source_symbol -> type == symbol_type_variable) || (source_symbol -> type == symbol_type_constant)) {return symbol_copy (source_symbol);}
    symbol* first_argument = source_symbol -> argument_count > 0 ? symbol_simplify (source_symbol -> arguments [0]) : NULL;
    symbol* second_argument = source_symbol -> argument_count > 1 ? symbol_simplify (source_symbol -> arguments [1]) : NULL;
    if (strcmp (source_symbol -> operation, "+") == 0) {
        if ((first_argument -> type == symbol_type_constant) && (first_argument -> value == 0.0)) {symbol_free (first_argument); return second_argument;}
        if ((second_argument -> type == symbol_type_constant) && (second_argument -> value == 0.0)) {symbol_free (second_argument); return first_argument;}
        if ((first_argument -> type == symbol_type_constant) && (second_argument -> type == symbol_type_constant)) {
            double result = first_argument -> value + second_argument -> value;
            symbol_free (first_argument); symbol_free (second_argument);
            return symbol_constant (result);
        }
        symbol* output_symbol = symbol_operation_new ("+", 2); output_symbol -> arguments [0] = first_argument; output_symbol -> arguments [1] = second_argument; return output_symbol;
    }
    if (strcmp (source_symbol -> operation, "*") == 0) {
        if ((first_argument -> type == symbol_type_constant) && (first_argument -> value == 0.0)) {symbol_free (second_argument); return first_argument;}
        if ((second_argument -> type == symbol_type_constant) && (second_argument -> value == 0.0)) {symbol_free (first_argument); return second_argument;}
        if ((first_argument -> type == symbol_type_constant) && (first_argument -> value == 1.0)) {symbol_free (first_argument); return second_argument;}
        if ((second_argument -> type == symbol_type_constant) && (second_argument -> value == 1.0)) {symbol_free (second_argument); return first_argument;}
        if ((first_argument -> type == symbol_type_constant) && (second_argument -> type == symbol_type_constant)) {
            double result = first_argument -> value* second_argument -> value;
            symbol_free (first_argument); symbol_free (second_argument);
            return symbol_constant (result);
        }
        symbol* output_symbol = symbol_operation_new ("*", 2); output_symbol -> arguments [0] = first_argument; output_symbol -> arguments [1] = second_argument; return output_symbol;
    }
    if (strcmp (source_symbol -> operation, "**") == 0) {
        if (source_symbol -> value == 0.0) {symbol_free (first_argument); return symbol_constant (1.0);}
        if (source_symbol -> value == 1.0) {return first_argument;}
        if (first_argument -> type == symbol_type_constant) {
            double result = pow (first_argument -> value, source_symbol -> value);
            symbol_free (first_argument);
            return symbol_constant (result);
        }
        symbol* output_symbol = symbol_operation_new ("**", 1); output_symbol -> arguments [0] = first_argument; output_symbol -> value = source_symbol -> value; return output_symbol;
    }
    if (strcmp (source_symbol -> operation, "exp") == 0) {
        if (first_argument -> type == symbol_type_constant) {
            double result = exp (first_argument -> value);
            symbol_free (first_argument);
            return symbol_constant (result);
        }
        symbol* output_symbol = symbol_operation_new ("exp", 1); output_symbol -> arguments [0] = first_argument; return output_symbol;
    }
    if (strcmp (source_symbol -> operation, "log") == 0) {
        if (first_argument -> type == symbol_type_constant) {
            double result = log (first_argument -> value);
            symbol_free (first_argument);
            return symbol_constant (result);
        }
        symbol* output_symbol = symbol_operation_new ("log", 1); output_symbol -> arguments [0] = first_argument; return output_symbol;
    }
    symbol* output_symbol = symbol_operation_new (source_symbol -> operation, source_symbol -> argument_count);
    output_symbol -> value = source_symbol -> value;
    output_symbol -> arguments [0] = first_argument;
    if (source_symbol -> argument_count > 1) {output_symbol -> arguments [1] = second_argument;}
    for (size_t i = 2; i < source_symbol -> argument_count; i ++) {output_symbol -> arguments [i] = symbol_copy (source_symbol -> arguments [i]);}
    return output_symbol;
}

```

---

### FILE: tests/test_attention.c
Location: `tests/test_attention.c`
```cpp
#include "basis/tensor.h"
#include "basis/sequence.h"
#include "basis/learning.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf ("Testing Transformer Attention Math...\n");
    tensor* query_tensor = tensor_new (1, 2);
    tensor_set (query_tensor, 0, 0, 1.0);
    tensor_set (query_tensor, 0, 1, 0.0);
    tensor* key_tensor = tensor_new (1, 2);
    tensor_set (key_tensor, 0, 0, 1.0);
    tensor_set (key_tensor, 0, 1, 0.0);
    tensor* value_tensor = tensor_new (1, 2);
    tensor_set (value_tensor, 0, 0, 10.0);
    tensor_set (value_tensor, 0, 1, 20.0);
    tensor* output_tensor = tensor_attention (query_tensor, key_tensor, value_tensor);
    tensor_print (output_tensor, "Attention Output");
    assert (fabs (output_tensor -> data [0] -> data - 10.0) < 1e-6);
    assert (fabs (output_tensor -> data [1] -> data - 20.0) < 1e-6);
    value* output_sum = tensor_sum (output_tensor);
    value_backward_propagation (output_sum);
    value_free (output_sum);
    printf ("Gradient dOutput/dV[0,0] = %f (expected 1.0)\n", value_tensor -> data [0] -> gradient);
    assert (fabs (value_tensor -> data [0] -> gradient - 1.0) < 1e-6);
    printf ("All Attention tests passed!\n");
    tensor_free (output_tensor);
    tensor_free (query_tensor);
    tensor_free (key_tensor);
    tensor_free (value_tensor);
    return 0;
}

```

---

### FILE: tests/test_compiler.c
Location: `tests/test_compiler.c`
```cpp
#include "basis/compiler.h"
#include "basis/symbolic.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf ("Testing Symbolic-to-Numeric Compiler...\n");
    symbol* x_symbol = symbol_variable ("x");
    symbol* w_symbol = symbol_variable ("w");
    symbol* b_symbol = symbol_variable ("b");
    symbol* L_symbol = symbol_addition (symbol_multiplication (x_symbol, w_symbol), b_symbol);
    symbol* dLdw_symbol = symbol_differentiation (L_symbol, "w");
    printf ("Symbolic dL/dw: ");
    symbol_print (dLdw_symbol);
    printf ("\n");
    value* x_value = value_new (2.0);
    value* w_value = value_new (3.0);
    value* b_value = value_new (10.0);
    compiler* compiler_instance = compiler_new ();
    compiler_map (compiler_instance, "x", x_value);
    compiler_map (compiler_instance, "w", w_value);
    compiler_map (compiler_instance, "b", b_value);
    value* dLdw_numeric = compiler_compile (compiler_instance, dLdw_symbol);
    printf ("Numeric dL/dw (Compiled): %f (Expected 2.0)\n", dLdw_numeric -> data);
    assert (fabs (dLdw_numeric -> data - 2.0) < 1e-6);
    value* L_numeric = value_addition (value_multiplication (x_value, w_value), b_value);
    value_backward_propagation (L_numeric);
    printf ("Numeric dL/dw (Autograd): %f (Expected 2.0)\n", w_value -> gradient);
    assert (fabs (w_value -> gradient - 2.0) < 1e-6);
    printf ("Compiler Test Passed! Symbolic Math == Numeric Code.\n");
    value_free (dLdw_numeric);
    value_free (L_numeric);
    value_free (x_value);
    value_free (w_value);
    value_free (b_value);
    symbol_free (L_symbol);
    symbol_free (dLdw_symbol);
    compiler_free (compiler_instance);
    return 0;
}

```

---

### FILE: tests/test_frontier.c
Location: `tests/test_frontier.c`
```cpp
#include "basis/tensor.h"
#include "basis/learning.h"
#include "basis/optim.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
int main () {
    printf ("Frontier Convergence Test: Learning x * 2 = 10\n");
    tensor* weight_tensor = tensor_new (1, 1);
    tensor_set (weight_tensor, 0, 0, 0.5);
    tensor* input_tensor = tensor_new (1, 1);
    tensor_set (input_tensor, 0, 0, 5.0);
    adam* optimizer = adam_new (1, 1, 0.1);
    for (int epoch = 0; epoch < 50; epoch ++) {
        tensor* output_tensor = tensor_matrix_multiplication (input_tensor, weight_tensor);
        value* target_value = value_new (-10.0);
        value* difference = value_addition (output_tensor -> data [0], target_value);
        value* loss = value_power (difference, 2.0);
        if (epoch % 10 == 0) {
            printf ("Epoch %d: Loss = %f, W = %f\n", epoch, loss -> data, weight_tensor -> data [0] -> data);
        }
        value_backward_propagation (loss);
        value_free (loss);
        value_free (difference);
        value_free (target_value);
        adam_optimization_step (optimizer, weight_tensor);
        tensor_free (output_tensor);
    }
    printf ("Final W: %f (Expected ~2.0)\n", weight_tensor -> data [0] -> data);
    assert (fabs (weight_tensor -> data [0] -> data - 2.0) < 0.1);
    adam_free (optimizer);
    tensor_free (weight_tensor);
    tensor_free (input_tensor);
    return 0;
}

```

---

### FILE: tests/test_grand_unified.c
Location: `tests/test_grand_unified.c`
```cpp
#include "basis/compiler.h"
#include "basis/geometry.h"
#include "basis/symbolic.h"
#include "basis/tensor.h"
#include <stdio.h>
int main () {
    printf ("Final Grand Unified Frontier Demo\n");
    printf ("--------------------------------\n");
    symbol* x_symbol = symbol_variable ("x");
    symbol* w_symbol = symbol_variable ("w");
    symbol* L_symbol = symbol_multiplication (x_symbol, w_symbol);
    printf ("Step 1: Symbolic Math Defined: ");
    symbol_print (L_symbol);
    printf ("\n");
    symbol* gradient_symbol = symbol_differentiation (L_symbol, "w");
    printf ("Step 2: Symbolic Gradient (dL/dw): ");
    symbol_print (gradient_symbol);
    printf ("\n");
    value* x_value = value_new (5.0);
    value* w_value = value_new (2.0);
    compiler* compiler_instance = compiler_new ();
    compiler_map (compiler_instance, "x", x_value);
    compiler_map (compiler_instance, "w", w_value);
    value* Y_numeric = compiler_compile (compiler_instance, L_symbol);
    printf ("Step 3: Compiled Numeric Output (Y = 5 * 2): %f\n", Y_numeric -> data);
    tensor* X_tensor = tensor_new (1, 1);
    tensor_set (X_tensor, 0, 0, 5.0);
    tensor* W_tensor = tensor_new (1, 1);
    tensor_set (W_tensor, 0, 0, 2.0);
    metric* fisher_information_metric = metric_fisher_information (W_tensor, X_tensor, NULL);
    printf ("Step 4: Information Geometry (Fisher Metric F): %f\n", fisher_information_metric -> matrix -> data [0] -> data);
    printf ("       (F represents the 'curvature' of the model at this point)\n");
    printf ("\nMission Accomplished: Math, Programming, and Geometry are unified.\n");
    value_free (Y_numeric);
    symbol_free (L_symbol);
    symbol_free (gradient_symbol);
    compiler_free (compiler_instance);
    metric_free (fisher_information_metric);
    tensor_free (X_tensor);
    tensor_free (W_tensor);
    value_free (x_value);
    value_free (w_value);
    return 0;
}

```

---

### FILE: tests/test_scalar.c
Location: `tests/test_scalar.c`
```cpp
#include "basis/scalar.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf ("Testing Scalar Autograd...\n");
    value* first_value = value_new (2.0);
    value* second_value = value_new (-3.0);
    value* third_value = value_new (10.0);
    value* product_value = value_multiplication (first_value, second_value);
    value* loss_value = value_addition (product_value, third_value);
    printf ("Forward Pass: L = %f (expected 4.0)\n", loss_value -> data);
    assert (fabs (loss_value -> data - 4.0) < 1e-6);
    value_backward_propagation (loss_value);
    printf ("Gradients:\n");
    printf ("  dL/da = %f (expected -3.0)\n", first_value -> gradient);
    printf ("  dL/db = %f (expected 2.0)\n", second_value -> gradient);
    printf ("  dL/dc = %f (expected 1.0)\n", third_value -> gradient);
    assert (fabs (first_value -> gradient - (-3.0)) < 1e-6);
    assert (fabs (second_value -> gradient - 2.0) < 1e-6);
    assert (fabs (third_value -> gradient - 1.0) < 1e-6);
    printf ("All Scalar tests passed!\n");
    value_free (loss_value);
    value_free (product_value);
    value_free (first_value);
    value_free (second_value);
    value_free (third_value);
    return 0;
}

```

---

### FILE: tests/test_symbolic.c
Location: `tests/test_symbolic.c`
```cpp
#include "basis/symbolic.h"
#include <stdio.h>
int main () {
    printf ("Testing Symbolic Algebra Engine...\n");
    symbol* x = symbol_variable ("x");
    symbol* w = symbol_variable ("w");
    symbol* b = symbol_variable ("b");
    symbol* xw = symbol_multiplication (x, w);
    symbol* L = symbol_addition (xw, b);
    printf ("Expression L: ");
    symbol_print (L);
    printf ("\n");
    symbol* dLdw = symbol_differentiation (L, "w");
    printf ("Raw dL/dw: ");
    symbol_print (dLdw);
    printf ("\n");
    symbol* dLdw_simplified = symbol_simplify (dLdw);
    printf ("Simplified dL/dw: ");
    symbol_print (dLdw_simplified);
    printf ("\n");
    symbol_free (L);
    symbol_free (dLdw);
    symbol_free (dLdw_simplified);
    return 0;
}

```

---

### FILE: tests/test_tensor.c
Location: `tests/test_tensor.c`
```cpp
#include "basis/tensor.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
int main () {
    printf ("Testing Tensor MatMul and Autograd...\n");
    tensor* input_tensor = tensor_new (1, 2);
    tensor_set (input_tensor, 0, 0, 1.0);
    tensor_set (input_tensor, 0, 1, 2.0);
    tensor* weight_tensor = tensor_new (2, 2);
    tensor_set (weight_tensor, 0, 0, 3.0);
    tensor_set (weight_tensor, 0, 1, 4.0);
    tensor_set (weight_tensor, 1, 0, 5.0);
    tensor_set (weight_tensor, 1, 1, 6.0);
    tensor* output_tensor = tensor_matrix_multiplication (input_tensor, weight_tensor);
    tensor_print (output_tensor, "Y = X * W");
    assert (fabs (output_tensor -> data [0] -> data - 13.0) < 1e-6);
    assert (fabs (output_tensor -> data [1] -> data - 16.0) < 1e-6);
    value* loss = value_addition (output_tensor -> data [0], output_tensor -> data [1]);
    printf ("Loss: %f (expected 29.0)\n", loss -> data);
    assert (fabs (loss -> data - 29.0) < 1e-6);
    value_backward_propagation (loss);
    printf ("Gradients of Weights (W):\n");
    for (size_t i = 0; i < weight_tensor -> row_count; i ++) {
        for (size_t j = 0; j < weight_tensor -> column_count; j ++) {
            double gradient = weight_tensor -> data [i * weight_tensor -> column_count + j] -> gradient;
            printf ("  dW[%zu,%zu] = %f\n", i, j, gradient);
            double expected;
            if (i == 0) {expected = 1.0;} else {expected = 2.0;}
            assert (fabs (gradient - expected) < 1e-6);
        }
    }
    printf ("All Tensor tests passed!\n");
    value_free (loss);
    tensor_free (output_tensor);
    tensor_free (input_tensor);
    tensor_free (weight_tensor);
    return 0;
}

```

---

