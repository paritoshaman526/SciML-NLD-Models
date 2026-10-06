import os
import sys

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import numpy as np

print("Testing Lorenz Pipeline...")

# Test 1: Check if checkpoint exists
ckpt_path = "lorenz_sciml_mlp.pt"
print(f"Checkpoint {ckpt_path} exists: {os.path.exists(ckpt_path)}")

# Test 2: Check imports
from lorenz_mlp import LorenzSciML, lorenz_deriv, rk4_step, set_seed
model = LorenzSciML(hidden_dim=128, dt=0.01)
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print("Loaded lorenz_sciml_mlp.pt successfully!")
model.eval()

# Test 3: Check forward rollout
x0 = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
with torch.no_grad():
    x1 = model(x0)
    deriv = model.predict_derivatives(x0)
print(f"x0 = {x0.numpy()} -> x1 = {x1.numpy()}")
print(f"dx/dt = {deriv.numpy()}")

# Compare with true Lorenz derivative
true_d = lorenz_deriv(np.array([1.0, 1.0, 1.0]))
print(f"True dx/dt = {true_d}")
print(f"Derivative error = {np.abs(deriv.numpy() - true_d)}")

# Test 4: Run climate validation imports
try:
    from lorenz_climate_validation import true_jacobian, neural_jacobian_batched
    print("lorenz_climate_validation imported successfully!")
except Exception as e:
    print(f"Error importing lorenz_climate_validation: {e}")
