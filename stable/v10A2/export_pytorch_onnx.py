import torch
import torch.nn as nn
import torch.onnx

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2, bias=False)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = LeNet()
model.eval()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "pytorch_lenet.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=17, do_constant_folding=True)
print("✅ Exported PyTorch model to pytorch_lenet.onnx")
