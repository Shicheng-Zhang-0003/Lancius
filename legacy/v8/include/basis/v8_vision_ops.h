#ifndef V8_VISION_OPS_H
#define V8_VISION_OPS_H
#include "basis/v8_ir.h"

// Routes heavy 4D math to the dedicated vision engine
void v8_execute_vision_op(v8_node* n);

#endif
