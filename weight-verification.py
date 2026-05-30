import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 1. Model Definition ───────────────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# ── 2. Config ─────────────────────────────────────────────────────────────────
FRAC_BITS = 7
SCALE     = 2 ** FRAC_BITS
HEX_DIR   = "hex_weights"

LAYER_SHAPES = {
    "conv1.weight": (6, 1, 5, 5),
    "conv1.bias":   (6,),
    "conv2.weight": (16, 6, 5, 5),
    "conv2.bias":   (16,),
    "fc1.weight":   (120, 256),
    "fc1.bias":     (120,),
    "fc2.weight":   (84, 120),
    "fc2.bias":     (84,),
    "fc3.weight":   (10, 84),
    "fc3.bias":     (10,),
}

# ── 3. Hex loader ─────────────────────────────────────────────────────────────
def load_hex_as_tensor(layer_name, shape):
    safe_name = layer_name.replace(".", "_")
    filepath  = os.path.join(HEX_DIR, f"{safe_name}.mem")
    with open(filepath, "r") as f:
        hex_lines = [l.strip() for l in f.read().splitlines() if l.strip()]
    int_vals = []
    for h in hex_lines:
        val = int(h, 16)
        if val > 127:
            val -= 256
        int_vals.append(val)
    float_vals = np.array(int_vals, dtype=np.float32) / SCALE
    return torch.tensor(float_vals).reshape(shape)

# ── 4. Load quantized weights from hex ───────────────────────────────────────
print("Loading hex weights...")
reconstructed_state_dict = {}
for layer_name, shape in LAYER_SHAPES.items():
    reconstructed_state_dict[layer_name] = load_hex_as_tensor(layer_name, shape)
    print(f"  {layer_name}: {shape} ✓")

# ── 5. Build quantized model ──────────────────────────────────────────────────
quant_model = LeNet5()
quant_model.load_state_dict(reconstructed_state_dict)
quant_model.eval()

# ── 6. Build original float model ────────────────────────────────────────────
float_model = LeNet5()
float_model.load_state_dict(torch.load("lenet5_mnist.pth", map_location="cpu"))
float_model.eval()

# ── 7. MNIST test set ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ── 8. Evaluate both models ───────────────────────────────────────────────────
def evaluate(model, loader):
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100 * correct / total

print("\nEvaluating...")
quant_acc = evaluate(quant_model, test_loader)
float_acc = evaluate(float_model, test_loader)

print(f"\nFloat32 accuracy:            {float_acc:.2f}%")
print(f"Quantized (Q1.7) accuracy:   {quant_acc:.2f}%")
print(f"Accuracy drop:               {float_acc - quant_acc:.2f}%")

# ── 9. Single image sanity check ──────────────────────────────────────────────
print("\n── Single image sanity check ──")
single_image, true_label = test_dataset[0]
single_image = single_image.unsqueeze(0)

with torch.no_grad():
    float_pred = float_model(single_image).argmax().item()
    quant_pred = quant_model(single_image).argmax().item()

print(f"True label:         {true_label}")
print(f"Float32 predicts:   {float_pred}")
print(f"Quantized predicts: {quant_pred}")

if float_acc - quant_acc < 1.0:
    print("\n✓ GREEN LIGHT — quantization loss < 1%, safe to proceed to C++ golden model")
else:
    print("\n✗ WARNING — accuracy drop > 1%, revisit quantization before proceeding")