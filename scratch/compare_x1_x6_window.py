import os
import sys
import numpy as np
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from train_and_evaluate_lorenz_style_sciml import UnifiedLorenzStyleSciML, find_physical_steady_state, PHYSICAL_CTOT
from biochemical_global_sciml import DT

m = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
m.load_state_dict(torch.load("unified_sciml_seed_42.pt", map_location="cpu", weights_only=True))
m.eval()

CASES = [
    ("X1", (11.0, 6000.0, 0.15, 1.0)),
    ("X2", (12.0, 6000.0, 0.15, 1.0)),
    ("X3", (13.0, 6000.0, 0.25, 1.0)),
    ("X4", (10.5, 4600.0, 0.60, 1.0)),
    ("X5", (12.0, 4500.0, 0.50, 1.0)),
    ("X6", (14.0, 5000.0, 0.30, 1.0)),
]

print("=" * 115, flush=True)
print("EVALUATION WINDOW COMPARISON FOR X1-X6 (Seed 42): 4,000 steps (0.4s) vs 40,000 steps (4.0s)", flush=True)
print("=" * 115, flush=True)
print(f"{'Case':4s} | {'True m1*':8s} | {'Init m1':8s} | {'m1 at 0.4s':11s} | {'Offset 0.4s':11s} | {'m1 at 4.0s':11s} | {'Offset 4.0s':11s} | {'d(m1)/dt at 4.0s':18s}", flush=True)
print("-" * 115, flush=True)

for cid, p in CASES:
    ystar = find_physical_steady_state(p)
    # +15% perturbation on m1
    y0 = ystar.copy()
    y0[2] = ystar[2] * 1.15
    # adjust d to keep mass c_tot
    y0[0] = ystar[0] - (y0[2] - ystar[2])
    
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
    curr = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]
    
    m1_4k = None
    m1_40k = None
    deriv_40k = None
    
    with torch.no_grad():
        for step in range(40001):
            if step == 4000:
                m1_4k = curr[0, 2].item()
            if step == 40000:
                m1_40k = curr[0, 2].item()
                deriv = m.predict_derivatives(curr, p_t)
                deriv_40k = deriv[0, 2].item()
            curr = m(curr, p_t, target_ctot=c_tot)
            
    off_4k = abs(m1_4k - ystar[2])
    off_40k = abs(m1_40k - ystar[2])
    print(f"{cid:4s} | {ystar[2]:8.2f} | {y0[2]:8.2f} | {m1_4k:11.2f} | {off_4k:11.2f} | {m1_40k:11.2f} | {off_40k:11.2f} | {deriv_40k:+18.6f}", flush=True)

print("=" * 115, flush=True)
