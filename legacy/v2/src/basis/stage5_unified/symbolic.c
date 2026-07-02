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
