import onnx
import os

model_path = "pytorch_lenet.onnx"
if not os.path.exists(model_path):
    print(f"❌ {model_path} not found. Make sure you run this inside the v10S directory.")
    exit(1)

print(f"Loading {model_path}...")
model = onnx.load(model_path)

print("Downgrading Opset version to 17...")
for opset in model.opset_import:
    if opset.domain == '' or opset.domain == 'ai.onnx':
        opset.version = 17

onnx.save(model, model_path)
print(f"✅ Successfully patched {model_path} to Opset 17.")
print("👉 Now run: python3 audit_pytorch_parity.py")
