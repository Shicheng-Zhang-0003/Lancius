#ifndef V8_BYTECODE_H
#define V8_BYTECODE_H
#include "basis/v8_ir.h"
#include "basis/v8_arena.h"
#include <stdint.h>

typedef enum {
    V8_BC_NOP = 0,
    V8_BC_MATMUL,
    V8_BC_ADD,
    V8_BC_RELU,
    V8_BC_BROADCAST,
    V8_BC_SOFTMAX,
    V8_BC_SUB,
    V8_BC_MUL,
    V8_BC_SUM,
    V8_BC_HALT
} v8_bc_opcode;

typedef struct {
    uint32_t* code;
    size_t code_len;
    uint32_t num_regs;
    uint32_t out_reg;
    uint32_t* input_regs;
    uint32_t input_count;
    size_t* rows;
    size_t* cols;
} v8_program;

v8_program* v8_compile_graph(v8_graph* g);
void v8_vm_execute(v8_program* prog, double** inputs, double* out, v8_arena* scratch);
void v8_program_destroy(v8_program* prog);
#endif
