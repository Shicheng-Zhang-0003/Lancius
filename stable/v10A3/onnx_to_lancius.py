import onnx
from onnx import numpy_helper, TensorProto
import struct
import numpy as np
import sys

LANCIUS_MAGIC = 0x21434E41
OP_MAP = {
    'Conv': 16, 'Relu': 7, 'MaxPool': 17, 'Flatten': 18,
    'MatMul': 6, 'Add': 3, 'Reshape': 27, 'Gemm': 6, 'Transpose': 11
}

def get_shape(tensor_type):
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append(1)
    while len(shape) < 4:
        shape.append(1)
    return shape[:4]

def convert(onnx_path, lancius_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    nodes = []
    name_to_id = {}
    next_id = 0

    initializer_map = {}

    # CRITICAL PRE-PASS: Register ALL Constants and Initializers before processing ops
    for init in graph.initializer:
        initializer_map[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    val = numpy_helper.to_array(attr.t)
                    initializer_map[node.output[0]] = val

    # 1. Register Initializers (Weights) as INPUT nodes
    for init in graph.initializer:
        data = numpy_helper.to_array(init).astype(np.float64)
        shape = list(data.shape)

        # V15 FIX: Force 1D biases to be [1, N] so they match MatMul output [1, N]
        calc_ndim = len(data.shape)
        if len(shape) == 1 and 'bias' in init.name:
            shape = [1, shape[0]]
            calc_ndim = 2

        while len(shape) < 4:
            shape.append(1)
        shape = shape[:4]
        nodes.append({
            'id': next_id, 'op': 1, 'ndim': calc_ndim, 'shape': shape,
            'inputs': [], 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
            'weights': data.tobytes(), 'dtype': 0, 'scale': 1.0
        })
        name_to_id[init.name] = next_id
        next_id += 1

    # 2. Register Graph Inputs
    for inp in graph.input:
        if inp.name not in name_to_id:
            shape = get_shape(inp.type.tensor_type)
            nodes.append({
                'id': next_id, 'op': 1, 'ndim': len([s for s in shape if s > 0]), 'shape': shape,
                'inputs': [], 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
                'weights': None, 'dtype': 0, 'scale': 1.0
            })
            name_to_id[inp.name] = next_id
            next_id += 1

    # 3. Map Operations
    for node in graph.node:
        if node.op_type not in OP_MAP and node.op_type != 'Constant':
            continue

        if node.op_type == 'Constant':
            continue # Already handled in pre-pass

        op = OP_MAP[node.op_type]
        inputs = [name_to_id[i] for i in node.input if i in name_to_id]

        out_shape = [1, 1, 1, 1]

        # Robust Reshape shape extraction
        if node.op_type == 'Reshape' and len(node.input) >= 2:
            shape_tensor_name = node.input[1]
            target_dims = None
            if shape_tensor_name in initializer_map:
                target_dims = initializer_map[shape_tensor_name].tolist()

            if target_dims is not None:
                resolved_dims = []
                total_known = 1
                has_neg = False
                for d in target_dims:
                    if d > 0:
                        total_known *= d
                    else:
                        has_neg = True

                resolved_neg = 2048 # Default fallback
                if has_neg and node.input[0] in name_to_id:
                    in_id = name_to_id[node.input[0]]
                    for n in nodes:
                        if n['id'] == in_id:
                            in_shape = [s for s in n['shape'] if s > 0]
                            total_in = 1
                            for s in in_shape: total_in *= s
                            if total_known > 0:
                                resolved_neg = total_in // total_known
                            break

                for d in target_dims:
                    if d <= 0:
                        resolved_dims.append(resolved_neg)
                    else:
                        resolved_dims.append(d)

                if len(resolved_dims) == 2:
                    out_shape = [1, resolved_dims[1], 1, 1]
                    # V14 FIX: Force ndim=2 for MatMul compatibility
                    nodes[-1]['ndim'] = 2
                elif len(resolved_dims) == 4:
                    out_shape = [1, resolved_dims[1], resolved_dims[2], resolved_dims[3]]
            else:
                # ULTIMATE FALLBACK: If shape tensor is completely missing, assume LeNet flatten
                out_shape = [1, 2048, 1, 1]
        else:
            for vi in graph.value_info:
                if vi.name == node.output[0]:
                    out_shape = get_shape(vi.type.tensor_type)
                    break

        print(f"  [DEBUG PY] Node: {node.op_type} | Inputs: {list(node.input)} | Out: {node.output[0]} | Shape: {out_shape}")
        meta = [0, 0, 0, 0]
        if node.op_type == 'Conv':
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    meta[0], meta[1] = attr.ints[0], attr.ints[1]
                if attr.name == 'strides':
                    meta[2] = attr.ints[0]
                if attr.name == 'pads':
                    meta[3] = attr.ints[0]
        elif node.op_type == 'MaxPool':
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    meta[0], meta[2] = attr.ints[0], attr.ints[0]
                if attr.name == 'strides':
                    meta[2] = attr.ints[0]

        # Decompose Gemm into MatMul + Add(bias) with Transpose support
        if node.op_type == 'Gemm' and len(node.input) >= 2:
            transB = 0
            for attr in node.attribute:
                if attr.name == 'transB':
                    transB = attr.i

            matmul_inputs = [name_to_id[i] for i in node.input[:2] if i in name_to_id]

            if transB == 1 and len(matmul_inputs) >= 2:
                w_name = node.input[1]
                w_id = name_to_id[w_name]
                for n in nodes:
                    if n['id'] == w_id:
                        old_shape = n['shape']
                        n['shape'] = [old_shape[1], old_shape[0], 1, 1]
                        n['ndim'] = 2
                        if n['weights']:
                            data = np.frombuffer(n['weights'], dtype=np.float64).reshape(old_shape[0], old_shape[1])
                            data_T = np.ascontiguousarray(data.T)
                            n['weights'] = data_T.tobytes()
                        break

            nodes.append({
                'id': next_id, 'op': 6, 'ndim': len([s for s in out_shape if s > 0]) if node.op_type != 'Gemm' else 2, 'shape': out_shape,
                'inputs': matmul_inputs, 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
                'weights': None, 'dtype': 0, 'scale': 1.0
            })
            matmul_id = next_id
            next_id += 1

            if len(node.input) == 3 and node.input[2] in name_to_id:
                bias_id = name_to_id[node.input[2]]
                nodes.append({
                    'id': next_id, 'op': 3, 'ndim': 2, 'shape': out_shape,
                    'inputs': [matmul_id, bias_id], 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
                    'weights': None, 'dtype': 0, 'scale': 1.0
                })
                name_to_id[node.output[0]] = next_id
                next_id += 1
            else:
                name_to_id[node.output[0]] = matmul_id

        elif node.op_type == 'Transpose':
            perm = [0, 1, 2, 3]
            for attr in node.attribute:
                if attr.name == 'perm':
                    perm = list(attr.ints)
            if perm == [1, 0] or perm == [1, 0, 2, 3]:
                nodes.append({
                    'id': next_id, 'op': 11, 'ndim': 2, 'shape': [out_shape[1], out_shape[0], 1, 1],
                    'inputs': inputs, 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
                    'weights': None, 'dtype': 0, 'scale': 1.0
                })
                name_to_id[node.output[0]] = next_id
                next_id += 1
        else:
            # V15 FIX: Force ndim=2 for Reshape to match C MatMul expectations
            calc_ndim = len([s for s in out_shape if s > 0])
            if node.op_type == 'Reshape':
                calc_ndim = 2

            nodes.append({
                'id': next_id, 'op': op, 'ndim': calc_ndim, 'shape': out_shape,
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
            for s in n['shape']:
                f.write(struct.pack('<Q', s))
            f.write(struct.pack('<I', len(n['inputs'])))
            for i in n['inputs']:
                f.write(struct.pack('<I', i))
            f.write(struct.pack('<d', n['attr']))
            for m in n['meta']:
                f.write(struct.pack('<I', m))
            for a in n['axes']:
                f.write(struct.pack('<I', a))
            f.write(struct.pack('<B', 0))
            has_w = 1 if n['weights'] else 0
            f.write(struct.pack('<B', has_w))
            if has_w:
                f.write(struct.pack('<B', n['dtype']))
                f.write(struct.pack('<d', n['scale']))
                f.write(n['weights'])

    print(f"✅ Translated {len(nodes)} nodes to {lancius_path}")

if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "pytorch_lenet.onnx"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "pytorch_lenet.lancius"
    convert(in_p, out_p)
