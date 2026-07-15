/**
 * @file lancius_vision_ops.h
 * @brief Vision Ops Router: Dynamic dispatch and mixed-precision quantization for Conv2D/MaxPool.
 */
#ifndef LANCIUS_VISION_OPS_H
#define LANCIUS_VISION_OPS_H
#include "lancius/lancius_ir.h"

// Routes heavy 4D math to the dedicated vision engine
void lancius_execute_vision_op(lancius_node* n);

#endif
