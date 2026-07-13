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
