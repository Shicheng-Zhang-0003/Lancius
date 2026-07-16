import onnx
from onnx import helper, TensorProto
import numpy as np

def create_tensor(name, shape):
    return helper.make_tensor(
        name, TensorProto.FLOAT, shape,
        np.random.randn(*shape).astype(np.float32).tobytes(), raw=True
    )

# 1. Initializers (Weights & Biases)
initializers = [
    create_tensor('conv1.weight', [16, 3, 5, 5]),
    create_tensor('conv2.weight', [32, 16, 5, 5]),
    create_tensor('fc1.weight', [2048, 120]),
    create_tensor('fc1.bias', [1, 120]),
    create_tensor('fc2.weight', [120, 10]),
    create_tensor('fc2.bias', [1, 10]),
]

# 2. Graph Nodes (Matching Lancius IR Shape Constraints)
nodes = [
    helper.make_node('Conv', ['input', 'conv1.weight'], ['conv1_out'], kernel_shape=[5,5], pads=[2,2,2,2], strides=[1,1]),
    helper.make_node('Relu', ['conv1_out'], ['relu1_out']),
    helper.make_node('MaxPool', ['relu1_out'], ['pool1_out'], kernel_shape=[2,2], strides=[2,2]),

    helper.make_node('Conv', ['pool1_out', 'conv2.weight'], ['conv2_out'], kernel_shape=[5,5], pads=[2,2,2,2], strides=[1,1]),
    helper.make_node('Relu', ['conv2_out'], ['relu2_out']),
    helper.make_node('MaxPool', ['relu2_out'], ['pool2_out'], kernel_shape=[2,2], strides=[2,2]),

    helper.make_node('Flatten', ['pool2_out'], ['flatten_out'], axis=1),

    helper.make_node('MatMul', ['flatten_out', 'fc1.weight'], ['matmul1_out']),
    helper.make_node('Add', ['matmul1_out', 'fc1.bias'], ['add1_out']),
    helper.make_node('Relu', ['add1_out'], ['relu3_out']),

    helper.make_node('MatMul', ['relu3_out', 'fc2.weight'], ['matmul2_out']),
    helper.make_node('Add', ['matmul2_out', 'fc2.bias'], ['output']),
]

# 3. Assemble Graph
graph_def = helper.make_graph(
    nodes, 'lancius_foreign_model',
    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])],
    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])],
    initializer=initializers
)

model_def = helper.make_model(graph_def, producer_name='lancius_pure_onnx')
onnx.save(model_def, 'pytorch_lenet.onnx')
print("✅ Generated pure ONNX model: pytorch_lenet.onnx")
