import torch
model = MyModel ()
script = torch.jit.script (model)
script.save ("model_test_2.pt")
