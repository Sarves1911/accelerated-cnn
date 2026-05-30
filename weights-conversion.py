import torch
import os

state_dict = torch.load("lenet5_mnist.pth", map_location="cpu")

FRAC_BITS = 7
SCALE = 2 ** FRAC_BITS
INT8_MIN = -128
INT8_MAX = 127

output_dir = "hex_weights"
os.makedirs(output_dir, exist_ok=True)

for name, tensor in state_dict.items():
    values = tensor.detach().cpu()
    
    # Quantize
    scaled = torch.round(values * SCALE)
    quantized = scaled.clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    
    # Flatten to 1D
    flat = quantized.flatten().tolist()
    
    # Convert to hex — int8 needs masking to get unsigned 2's complement
    hex_lines = [f"{val & 0xFF:02X}" for val in flat]
    
    # Save as .mem file (Verilog $readmemh compatible)
    safe_name = name.replace(".", "_")
    filepath = os.path.join(output_dir, f"{safe_name}.mem")
    with open(filepath, "w") as f:
        f.write("\n".join(hex_lines))
    
    print(f"Saved {safe_name}.mem — {len(flat)} values")