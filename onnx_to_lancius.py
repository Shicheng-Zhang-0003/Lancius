import onnx
from onnx import numpy_helper, TensorProto
import struct
import io
import zlib
import numpy as np
import sys

# v11A1 Task 6c: this converter writes v2. C runtime writes/loads v2 and loads v1.
LANCIUS_MAGIC = 0x21434E41
LANCIUS_MAGIC_V2 = 0x32434E41
LANCIUS_VERSION_V2 = 2
LANCIUS_FLAGS_V2 = 3  # little-endian | static-graph
OP_MAP = {
    'Conv': 16, 'Relu': 7, 'MaxPool': 17, 'Flatten': 18,
    'MatMul': 6, 'Add': 3, 'Reshape': 27, 'Gemm': 6, 'Transpose': 11
}

def get_shape(tensor_type):
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            # Hostile fix: symbolic dim (e.g. dynamic batch) was silently frozen to 1.
            # Fail loud instead of emitting a silently wrong static shape.
            raise ValueError(f"Symbolic dim '{dim.dim_param}' requires explicit batch resolution; refusing silent freeze to 1.")
        else:
            shape.append(1)
    while len(shape) < 4:
        shape.append(1)
    if len(tensor_type.shape.dim) > 4:
        raise ValueError(f"Rank {len(tensor_type.shape.dim)} > 4 not supported; refusing silent truncation.")
    return shape[:4]

def convert(onnx_path, lancius_path):
    import os
    # Despot V6 truth: size cap before load (was unbounded onnx.load).
    try:
        if os.path.getsize(onnx_path) > 2 * 1024 * 1024 * 1024:
            raise ValueError(f"ONNX file exceeds 2GB cap: {onnx_path}")
    except OSError as e:
        raise ValueError(f"cannot stat ONNX file: {e}")
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
    # External audit V8: Reshape shape tensors (e.g. [2] int data) were
    # emitted as 1D INPUT nodes, which the v2 loader rejects (INPUT is
    # 2/3/4-D only) — every converted LeNet was unloadable. Skip them:
    # Reshape consumes shape from resolved dims, never as a graph input.
    shape_helper_names = set()
    for _n in graph.node:
        if _n.op_type == 'Reshape' and len(_n.input) >= 2 and _n.input[1]:
            shape_helper_names.add(_n.input[1])
    for init in graph.initializer:
        if init.name in shape_helper_names:
            continue
        data = numpy_helper.to_array(init).astype(np.float64)
        shape = list(data.shape)

        # V10S FIX: Force 1D biases to be [1, N] so they match MatMul output [1, N]
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
    # Despot truth: ndim is the TRUE rank (was: counted >0 on 4-padded shape,
    # always 4, so [N,C] inputs became 4D and broke 2D MatMul).
    for inp in graph.input:
        if inp.name not in name_to_id:
            true_rank = len(inp.type.tensor_type.shape.dim)
            if true_rank < 1 or true_rank > 4:
                raise ValueError(f"Graph input '{inp.name}' has unsupported rank {true_rank}.")
            shape = get_shape(inp.type.tensor_type)
            nodes.append({
                'id': next_id, 'op': 1, 'ndim': true_rank, 'shape': shape,
                'inputs': [], 'attr': 0.0, 'meta': [0,0,0,0], 'axes': [0,0,0,0],
                'weights': None, 'dtype': 0, 'scale': 1.0
            })
            name_to_id[inp.name] = next_id
            next_id += 1

    # 3. Map Operations
    for node in graph.node:
        if node.op_type not in OP_MAP and node.op_type != 'Constant':
            raise ValueError(f"Unsupported ONNX op '{node.op_type}' (output '{node.output[0] if node.output else '?'}'). Converter supports only {sorted(OP_MAP)}.")

        if node.op_type == 'Constant':
            continue # Already handled in pre-pass

        op = OP_MAP[node.op_type]
        if node.op_type == 'Reshape':
            # Shape comes from resolved dims; the shape tensor is not data.
            if not node.input or node.input[0] not in name_to_id:
                raise ValueError(f"Node 'Reshape' has unmapped data input {list(node.input[:1])}. Refusing to emit partial inputs.")
            inputs = [name_to_id[node.input[0]]]
        else:
            missing = [i for i in node.input if i and i not in name_to_id]
            if missing:
                raise ValueError(f"Node '{node.op_type}' has unmapped inputs {missing}. Refusing to emit partial inputs.")
            inputs = [name_to_id[i] for i in node.input if i in name_to_id]

        out_shape = [1, 1, 1, 1]
        reshape_rank = 2

        # Robust Reshape shape extraction
        if node.op_type == 'Reshape' and len(node.input) >= 2:
            shape_tensor_name = node.input[1]
            target_dims = None
            if shape_tensor_name in initializer_map:
                target_dims = initializer_map[shape_tensor_name].tolist()

            if target_dims is not None:
                # External audit V8: ONNX Reshape allowzero (default 0) selects
                # 0-means-copy vs 0-means-explicit-zero (unsupported here).
                allowzero = 0
                for _attr in node.attribute:
                    if _attr.name == 'allowzero':
                        allowzero = int(_attr.i)
                if allowzero not in (0, 1):
                    raise ValueError(f"Reshape '{node.output[0]}' has illegal allowzero={allowzero}.")
                if allowzero == 1 and any(d == 0 for d in target_dims):
                    raise ValueError(f"Reshape '{node.output[0]}' uses allowzero=1 explicit 0 dims {target_dims}; Lancius shapes must be >0, refusing silent mis-shape.")
                if 0 in target_dims and -1 in target_dims:
                    raise ValueError(f"Reshape '{node.output[0]}' mixes 0 and -1 in {target_dims}; ambiguous, refusing emit.")
                # Hostile fix: ONNX semantics — 0 means copy input dim, -1 means infer.
                # Previous code conflated both as infer (d<=0 -> resolved_neg). Correct:
                if node.input[0] not in name_to_id:
                    raise ValueError(f"Reshape '{node.output[0]}' input not mapped; cannot resolve copy dims.")
                in_id = name_to_id[node.input[0]]
                in_shape_full = None
                for n in nodes:
                    if n['id'] == in_id:
                        in_shape_full = list(n['shape'])
                        break
                # total_known over positive dims only; -1 inferred; 0 copied
                resolved_dims = []
                total_known = 1
                neg_count = 0
                for d in target_dims:
                    if d > 0:
                        total_known *= d
                    elif d == -1:
                        neg_count += 1
                    elif d == 0:
                        pass
                    else:
                        raise ValueError(f"Reshape '{node.output[0]}' has illegal dim {d}.")
                if neg_count > 1:
                    raise ValueError(f"Reshape '{node.output[0]}' has {neg_count} infer dims; at most one -1 allowed.")
                # resolve copy dims first
                tmp_dims = []
                for idx, d in enumerate(target_dims):
                    if d == 0:
                        # copy input dim at same rank position; supported ranks 2/4 only.
                        # Use positional copy when ranks match, else fail loud.
                        # (Despot truth: no dead rank_in computation; copy is positional.)
                        if in_shape_full is None or idx >= len(target_dims):
                            raise ValueError(f"Reshape copy-dim failed for '{node.output[0]}'.")
                        # input dims in converter are 4-padded; target rank may be 2 or 4
                        # Align: for rank-2 target [d0,d1], d0 copies in_shape_full[0] if present
                        src = None
                        if len(target_dims) == 2:
                            src = [in_shape_full[0], in_shape_full[1]][idx] if idx < 2 else None
                        elif len(target_dims) == 4:
                            src = in_shape_full[idx] if idx < 4 else None
                        if src is None or src <= 0:
                            raise ValueError(f"Reshape copy-dim 0 at pos {idx} unresolvable for '{node.output[0]}'.")
                        tmp_dims.append(src)
                        total_known *= src
                    elif d == -1:
                        tmp_dims.append(-1)
                    else:
                        tmp_dims.append(d)
                resolved_neg = None
                if neg_count == 1 and node.input[0] in name_to_id:
                    in_id = name_to_id[node.input[0]]
                    for n in nodes:
                        if n['id'] == in_id:
                            in_shape = [s for s in n['shape'] if s != 0]
                            total_in = 1
                            for s in in_shape: total_in *= s
                            if total_known > 0 and total_in % total_known == 0:
                                resolved_neg = total_in // total_known
                            break
                if neg_count == 1 and resolved_neg is None:
                    raise ValueError(f"Reshape '{node.output[0]}' has dynamic dim {target_dims} that cannot be resolved from input.")
                resolved_dims = [resolved_neg if d == -1 else d for d in tmp_dims]

                if len(resolved_dims) == 2:
                    out_shape = [resolved_dims[0], resolved_dims[1], 1, 1]
                    reshape_rank = 2
                elif len(resolved_dims) == 4:
                    out_shape = [resolved_dims[0], resolved_dims[1], resolved_dims[2], resolved_dims[3]]
                    reshape_rank = 4
                else:
                    raise ValueError(f"Reshape '{node.output[0]}' resolved to unsupported rank {len(resolved_dims)}: {resolved_dims}.")
            else:
                raise ValueError(f"Reshape '{node.output[0]}' has no resolvable shape tensor '{shape_tensor_name}'. Refusing LeNet fallback.")
        else:
            for vi in graph.value_info:
                if vi.name == node.output[0]:
                    out_shape = get_shape(vi.type.tensor_type)
                    break

        meta = [0, 0, 0, 0]
        if node.op_type == 'Conv':
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    if len(attr.ints) != 2:
                        raise ValueError(f"Conv '{node.output[0]}' kernel_shape rank {len(attr.ints)} != 2.")
                    meta[0], meta[1] = attr.ints[0], attr.ints[1]
                if attr.name == 'strides':
                    if len(attr.ints) >= 2 and attr.ints[0] != attr.ints[1]:
                        raise ValueError(f"Conv '{node.output[0]}' asymmetric strides {list(attr.ints)} not supported; refusing silent [0]-only.")
                    meta[2] = attr.ints[0]
                if attr.name == 'pads':
                    # ONNX pads = [begin_h,begin_w,...,end_h,end_w] or [h,w]; require symmetric
                    pads = list(attr.ints)
                    if len(pads) == 4 and not (pads[0] == pads[2] and pads[1] == pads[3]):
                        raise ValueError(f"Conv '{node.output[0]}' asymmetric pads {pads} not supported.")
                    if len(pads) >= 2 and pads[0] != pads[1] and len(pads) == 2:
                        # 2-elem pads [h,w] may differ per axis; C supports single pad -> require equal
                        raise ValueError(f"Conv '{node.output[0]}' asymmetric pads {pads} need single pad.")
                    meta[3] = pads[0]
                if attr.name == 'dilations':
                    dil = list(attr.ints)
                    if any(d != 1 for d in dil):
                        raise ValueError(f"Conv '{node.output[0]}' dilations {dil} != 1 not supported; refusing silent dense compute.")
                if attr.name == 'group':
                    if attr.i != 1:
                        raise ValueError(f"Conv '{node.output[0]}' group={attr.i} != 1 not supported; refusing silent dense compute.")
                if attr.name == 'auto_pad':
                    ap = attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
                    if ap not in ('NOTSET', ''):
                        raise ValueError(f"Conv '{node.output[0]}' auto_pad='{ap}' not supported; export explicit pads.")
            if (meta[0] <= 0 or meta[1] <= 0) and len(node.input) >= 2 and node.input[1] in initializer_map:
                # Torch-slim exports may omit kernel_shape; recover from weight [Cout,Cin,Kh,Kw].
                w = initializer_map[node.input[1]]
                if len(w.shape) == 4 and w.shape[2] > 0 and w.shape[3] > 0:
                    meta[0], meta[1] = int(w.shape[2]), int(w.shape[3])
            if meta[0] <= 0 or meta[1] <= 0:
                raise ValueError(f"Conv '{node.output[0]}' has no kernel_shape and it cannot be recovered from weights; refusing zero-kernel.")
            if meta[2] <= 0:
                meta[2] = 1  # ONNX default stride when the attribute is omitted
        elif node.op_type == 'MaxPool':
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    if len(attr.ints) != 2 or attr.ints[0] != attr.ints[1]:
                        raise ValueError(f"MaxPool '{node.output[0]}' non-square kernel {list(attr.ints)} not supported.")
                    meta[0], meta[1] = attr.ints[0], attr.ints[1]
                if attr.name == 'strides':
                    if len(attr.ints) >= 2 and attr.ints[0] != attr.ints[1]:
                        raise ValueError(f"MaxPool '{node.output[0]}' asymmetric strides {list(attr.ints)}.")
                    meta[2] = attr.ints[0]
                if attr.name == 'pads':
                    pads = list(attr.ints)
                    if any(p != 0 for p in pads):
                        raise ValueError(f"MaxPool '{node.output[0]}' pads {pads} != 0 not supported; C MaxPool is valid-only.")
                if attr.name == 'dilations':
                    dil = list(attr.ints)
                    if any(d != 1 for d in dil):
                        raise ValueError(f"MaxPool '{node.output[0]}' dilations {dil} != 1 not supported.")
                if attr.name == 'ceil_mode':
                    if attr.i != 0:
                        raise ValueError(f"MaxPool '{node.output[0]}' ceil_mode=1 not supported; C uses floor (H-K)/s+1.")
                if attr.name == 'auto_pad':
                    ap = attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
                    if ap not in ('NOTSET', ''):
                        raise ValueError(f"MaxPool '{node.output[0]}' auto_pad='{ap}' not supported.")

        # Decompose Gemm into MatMul + Add(bias) with Transpose support
        if node.op_type == 'Gemm' and len(node.input) >= 2:
            transB = 0
            alpha = 1.0
            beta = 1.0
            transA = 0
            for attr in node.attribute:
                if attr.name == 'transB':
                    transB = attr.i
                if attr.name == 'transA':
                    transA = attr.i
                if attr.name == 'alpha':
                    alpha = attr.f
                if attr.name == 'beta':
                    beta = attr.f
            if transA != 0:
                raise ValueError(f"Gemm '{node.output[0]}' transA=1 not supported; refusing silent wrong math.")
            if abs(alpha - 1.0) > 1e-12 or abs(beta - 1.0) > 1e-12:
                raise ValueError(f"Gemm '{node.output[0]}' alpha={alpha} beta={beta} != 1; refusing silent scale drop.")

            matmul_inputs = [name_to_id[i] for i in node.input[:2]]
            if len(matmul_inputs) != 2:
                raise ValueError(f"Gemm '{node.output[0]}' needs 2 mapped inputs, got {node.input[:2]}.")

            if transB == 1 and len(matmul_inputs) >= 2:
                w_name = node.input[1]
                w_id = name_to_id[w_name]
                # Despot truth: NEVER mutate the shared initializer in place.
                # Always transpose a fresh clone so N Gemm uses of one weight stay correct.
                w_node = None
                for n in nodes:
                    if n['id'] == w_id:
                        w_node = n
                        break
                if w_node is None:
                    raise ValueError(f"Gemm '{node.output[0]}' weight '{w_name}' not found.")
                import copy as _copy
                nn = _copy.deepcopy(w_node)
                nn['id'] = next_id
                next_id += 1
                # transpose the clone from the ORIGINAL orientation every time
                orig_shape = list(w_node['shape'])
                # original initializer shape is [R,C,1,1] padded; transpose first two dims
                nn['shape'] = [orig_shape[1], orig_shape[0], 1, 1]
                nn['ndim'] = 2
                if nn['weights']:
                    # Despot truth: raw reshape raised context-free ValueError.
                    try:
                        _dt = {0: np.float64, 1: np.int8, 2: np.float32, 3: np.int32}.get(w_node['dtype'])
                        if _dt is None:
                            raise ValueError(f"Gemm '{node.output[0]}' weight '{w_name}' has unsupported dtype {w_node['dtype']}.")
                        data = np.frombuffer(w_node['weights'], dtype=_dt).reshape(orig_shape[0], orig_shape[1])
                    except ValueError as e:
                        raise ValueError(f"Gemm '{node.output[0]}' weight '{w_name}' bytes {len(w_node['weights'])} mismatch shape [{orig_shape[0]},{orig_shape[1]}]; refusing emit.") from e
                    nn['weights'] = np.ascontiguousarray(data.T).tobytes()
                nn['_transposed_for'] = node.output[0]
                nn['_used'] = True
                nodes.append(nn)
                matmul_inputs[1] = nn['id']
                name_to_id[w_name + f"__T_{node.output[0]}"] = nn['id']

            # Gemm out shape: value_info is often absent for torch Gemm outputs
            # (would silently store [1,1,1,1]). Recover from A rows x B cols.
            a_shape = w_shape = None
            for n in nodes:
                if n['id'] == matmul_inputs[0]:
                    a_shape = list(n['shape'])
                if n['id'] == matmul_inputs[1]:
                    w_shape = list(n['shape'])
            if a_shape is not None and w_shape is not None and a_shape[0] > 0 and w_shape[1] > 0:
                out_shape = [a_shape[0], w_shape[1], 1, 1]
            elif out_shape == [1, 1, 1, 1]:
                raise ValueError(f"Gemm '{node.output[0]}' output shape unresolvable (no value_info, inputs unclear); refusing [1,1] guess.")

            nodes.append({
                'id': next_id, 'op': 6, 'ndim': 2, 'shape': out_shape,
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
                raise ValueError(f"Transpose '{node.output[0]}' has unsupported perm {perm}. Supports only [1,0] / [1,0,2,3].")
        else:
            # Despot truth: Reshape ndim follows the resolved rank (was: forced
            # to 2 even for 4D targets, dropping dims in C validation).
            # External audit V8: elementwise/Flatten/MatMul counted padded
            # nonzeros ([1,120,1,1] -> 4) so a 2D RELU was emitted 4D and
            # failed validation. Follow the data rank instead.
            if node.op_type == 'Reshape':
                calc_ndim = reshape_rank
            elif node.op_type in ('Relu', 'Add', 'Sub', 'Mul', 'Flatten', 'MatMul'):
                _in_ndim = None
                for _iid in inputs[:1]:
                    for _n in nodes:
                        if _n['id'] == _iid:
                            _in_ndim = _n['ndim']
                            break
                if _in_ndim in (1, 2, 3, 4):
                    calc_ndim = _in_ndim
                    # shape must agree on the data dims; keep padded storage
                    if calc_ndim == 2:
                        out_shape = [out_shape[0], out_shape[1], 1, 1]
                    elif calc_ndim == 4:
                        pass
                    else:
                        raise ValueError(f"{node.op_type} '{node.output[0]}' input rank {_in_ndim} not 2/4 on the LeNet path; refusing emit.")
                else:
                    calc_ndim = len([s for s in out_shape if s > 0])
            else:
                calc_ndim = len([s for s in out_shape if s > 0])

            nodes.append({
                'id': next_id, 'op': op, 'ndim': calc_ndim, 'shape': out_shape,
                'inputs': inputs, 'attr': 0.0, 'meta': meta, 'axes': [0,0,0,0],
                'weights': None, 'dtype': 0, 'scale': 1.0
            })
            name_to_id[node.output[0]] = next_id
            next_id += 1

    # 4. Write Binary (v2 format with CRC32 over body bytes 48..EOF)
    # Despot truth: every packed field is range-checked (was: huge dims raised
    # raw struct.error traceback); output open is guarded (was: traceback +
    # partial file on bad dir/permission).
    def _u32(v, what):
        if not isinstance(v, int) or v < 0 or v > 0xFFFFFFFF:
            raise ValueError(f"Node field {what}={v!r} out of u32 range; refusing emit.")
        return v

    def _u64(v, what):
        if not isinstance(v, int) or v < 0 or v > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"Node field {what}={v!r} out of u64 range; refusing emit.")
        return v

    body = io.BytesIO()
    for n in nodes:
        has_w = 1 if n['weights'] else 0

        if has_w:
            if n['dtype'] == 1:
                weight_elems = len(n['weights'])
            else:
                weight_elems = len(n['weights']) // 8
        else:
            weight_elems = 0
        if weight_elems > 100000000:
            raise ValueError(f"Node id={n['id']} weight_elems={weight_elems} exceeds 100M cap; refusing emit.")

        body.write(struct.pack(
            '<IIB4QId4I4I3BdQ',
            _u32(n['id'], 'id'), _u32(n['op'], 'op'), n['ndim'],
            _u64(n['shape'][0], 'shape0'), _u64(n['shape'][1], 'shape1'),
            _u64(n['shape'][2], 'shape2'), _u64(n['shape'][3], 'shape3'),
            _u32(len(n['inputs']), 'input_count'),
            float(n['attr']),
            n['meta'][0], n['meta'][1], n['meta'][2], n['meta'][3],
            n['axes'][0], n['axes'][1], n['axes'][2], n['axes'][3],
            0, n['dtype'], has_w,
            float(n['scale']),
            _u64(weight_elems, 'weight_elems')
        ))

        for i in n['inputs']:
            body.write(struct.pack('<I', _u32(i, 'input_id')))

        if has_w and weight_elems > 0:
            body.write(n['weights'])

    body_bytes = body.getvalue()
    checksum = zlib.crc32(body_bytes) & 0xFFFFFFFF
    if checksum == 0:
        checksum = 1  # 0 means legacy/unverified; never emit it (mirrors C saver)
    try:
        with open(lancius_path, 'wb') as f:
            header = struct.pack(
                '<8IQ2I',
                LANCIUS_MAGIC_V2, LANCIUS_VERSION_V2, LANCIUS_FLAGS_V2,
                len(nodes), len(nodes), 0, 48, 0,
                0, checksum, 0
            )
            f.write(header)
            f.write(body_bytes)
    except OSError as e:
        raise ValueError(f"Cannot write '{lancius_path}': {e}.") from e

    print(f"✅ Translated {len(nodes)} nodes to {lancius_path}")

if __name__ == "__main__":
    in_p = sys.argv[1] if len(sys.argv) > 1 else "pytorch_lenet.onnx"
    out_p = sys.argv[2] if len(sys.argv) > 2 else "pytorch_lenet.lancius"
    convert(in_p, out_p)
