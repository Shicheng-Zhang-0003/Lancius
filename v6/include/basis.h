/**
 * @file basis.h
 * @brief BASIS v5: Bare-metal Autograd & Symbolic Inference System
 *
 * @section intro_sec Introduction
 * BASIS is a from-scratch, pure-C implementation of a reverse-mode automatic
 * differentiation engine, tensor mathematics, symbolic calculus, and a JIT
 * machine-code compiler.
 *
 * @section memory_sec Strict Memory Ownership
 * BASIS uses manual reference counting.
 * - Functions that create new tensors/values (e.g., basis_tensor_new, basis_value_addition)
 *   return objects with a reference count of 1. The caller owns them and MUST free them.
 * - Views (e.g., basis_tensor_transpose, basis_tensor_broadcast_view) share underlying
 *   data and increment the parent's reference count. Freeing a view is safe and required.
 * - Optimizer states (basis_adam) are decoupled from the autograd graph and must be
 *   freed independently.
 *
 * @section compile_sec Compilation
 * Link against the shared library: `gcc main.c -lbasis -lm -ldl`
 */
#ifndef BASIS_UMBRELLA_H
#define BASIS_UMBRELLA_H

#include "basis/core/error.h"
#include "basis/stage1_atomic/scalar.h"
#include "basis/stage2_linear/tensor.h"
#include "basis/stage3_sequence/sequence.h"
#include "basis/stage4_learning/learning.h"
#include "basis/stage4_learning/optim.h"
#include "basis/stage5_unified/symbolic.h"
#include "basis/stage5_unified/compiler.h"
#include "basis/stage5_unified/geometry.h"
#include "basis/stage5_unified/jit.h"
#include "basis/stage6_ir/graph.h"
#include "basis/stage6_ir/lowering.h"

#endif
