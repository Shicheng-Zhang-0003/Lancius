import onnx
from onnx import numpy_helper, TensorProto
import struct
import numpy as np
import sys

LANCIUS_MAGIC = 0x21434E41
OP_MAP = {
    'Conv': 16, 'Relu': 7, 'MaxPool': 17, 'Flatten': 18,
    'MatMul': 6, 'Add': 3, 'Reshape': 29
}

def get_shape(tensor_type):
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0: shape.append(dim.dim_value)
        else: shape.append(1) # Fallback for dynamic batch
    while len(shape) < 4: shape.append(1)
    return shape[:4]

def convert(onnx_path, lancius_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)

    graph = model.graph
    nodes = []
    name_to_id = {}
    next_id = 0

    # 1. Register Initializers (Weights) as INPUT nodes
    for init in graph.initializer:
        data = numpy_helper.to_array(init).astype(np.float64)
        shape = list(data.shape)
        while len(shape) < 4: shape.append(1)
        shape = shape[:4]

        nodes.append({
            'id': next_id, 'op': 1, 'ndim': len(data.shape), 'shape': shape,
            'inputs': [], 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
            'weights': data.tobytes(), 'dtype': 0, 'scale': 1.0
        })
        name_to_id[init.name] = next_id
        next_id += 1

    # 2. Register Graph Inputs (Skip if already in initializers)
    for inp in graph.input:
        if inp.name not in name_to_id:
            shape = get_shape(inp.type.tensor_type)
            nodes.append({
                'id': next_id, 'op': 1, 'ndim': len([s for s in shape if s>0]), 'shape': shape,
                'inputs': [], 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
                'weights': None, 'dtype': 0, 'scale': 1.0
            })
            name_to_id[inp.name] = next_id
            next_id += 1

    # 3. Map Operations
    for node in graph.node:
        if node.op_type not in OP_MAP:
            print(f"⚠️ Skipping unsupported op: {node.op_type}")
            continue

        op = OP_MAP[node.op_type]
        inputs = [name_to_id[i] for i in node.input if i in name_to_id]

        # Extract shapes from output
        out_shape = [1,1,1,1]
        for vi in graph.value_info:
            if vi.name == node.output[0]:
                out_shape = get_shape(vi.type.tensor_type)
                break

        meta = [0,0,0,0]
        if node.op_type == 'Conv':
            for attr in node.attribute:
                if attr.name == 'kernel_shape': meta[0], meta[1] = attr.ints[0], attr.ints[1]
                if attr.name == 'strides': meta[2] = attr.ints[0]
                if attr.name == 'pads': meta[3] = attr.ints[0]
        elif node.op_type == 'MaxPool':
            for attr in node.attribute:
                if attr.name == 'kernel_shape': meta[0], meta[2] = attr.ints[0], attr.ints[0]
                if attr.name == 'strides': meta[2] = attr.ints[0]

        nodes.append({
            'id': next_id, 'op': op, 'ndim': len([s for s in out_shape if s>0]), 'shape': out_shape,
            'inputs': inputs, 'attr': 0.0, 'meta': meta, 'axes': [0,0,0,0],
            'weights': None, 'dtype': 0, 'scale': 1.0
        })
        name_to_id[node.output[0]] = next_id
        next_id += 1

    # 4. Write Binary
    with open(lancius_path, 'wb') as f:
        f.write(struct.pack('<I', LANCIUS_MAGIC))
        f.write(struct.pack('<I', len(nodes)))
        for n in nodes:
            f.write(struct.pack('<I', n['id']))
            f.write(struct.pack('<I', n['op']))
            f.write(struct.pack('<B', n['ndim']))
            for s in n['shape']: f.write(struct.pack('<Q', s))
            f.write(struct.pack('<I', len(n['inputs'])))
            for i in n['inputs']: f.write(struct.pack('<I', i))
            f.write(struct.pack('<d', n['attr']))
            for m in n['meta']: f.write(struct.pack('<I', m))
            for a in n['axes']: f.write(struct.pack('<I', a))

            has_w = 1 if n['weights'] else 0
            f.write(struct.pack('<B', has_w))
            if has_w:
                f.write(struct.pack('<B', n['dtype']))
                f.write(struct.pack('<d', n['scale']))
                f.write(n['weights'])
    print(f"✅ Translated {len(nodes)} nodes to {lancius_path}")

if __name__ == "__main__":
    convert("pytorch_lenet.onnx", "pytorch_lenet.lancius")
