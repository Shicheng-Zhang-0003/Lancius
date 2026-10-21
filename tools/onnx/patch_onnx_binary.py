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

patched_path = model_path.replace(".onnx", "_patched.onnx")
onnx.save(model, patched_path)
print(f"✅ Successfully patched {patched_path} to Opset 17 (original {model_path} unchanged).")
print("👉 Now run: python3 audit_pytorch_parity.py")
