/**
 * @file lancius_bytecode.h
 * @brief Bytecode VM: AOT compilation of the SSA graph into a flat register-based instruction tape.
 */
#ifndef LANCIUS_BYTECODE_H
#define LANCIUS_BYTECODE_H
#include "lancius/lancius_ir.h"
#include "lancius/lancius_arena.h"
#include <stdint.h>

typedef enum {
    LANCIUS_BC_NOP = 0,
    LANCIUS_BC_MATMUL,
    LANCIUS_BC_ADD,
    LANCIUS_BC_RELU,
    LANCIUS_BC_BROADCAST,
    LANCIUS_BC_SOFTMAX,
    LANCIUS_BC_SUB,
    LANCIUS_BC_MUL,
    LANCIUS_BC_SUM,
    LANCIUS_BC_HALT
} lancius_bc_opcode;

typedef struct {
    uint32_t* code;
    size_t code_len;
    uint32_t num_regs;
    uint32_t out_reg;
    uint32_t* input_regs;
    uint32_t input_count;
    size_t* rows;
    size_t* cols;
} lancius_program;

lancius_program* lancius_compile_graph(lancius_graph* g);
void lancius_vm_execute(lancius_program* prog, double** inputs, double* out, lancius_arena* scratch);
void lancius_program_destroy(lancius_program* prog);
#endif
