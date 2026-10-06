import torch
import os
import sys
import numpy as np

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo")
from fhn_global_sciml import ProposedFiLMFHNMLP, simulate_trajectory, measure_period_and_amplitude, DT

m = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
ckpt = r"c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo\scratch\fhn_sciml_s42.pt"
m.load_state_dict(torch.load(ckpt, map_location="cpu"))
m.eval()

p = np.array([0.80, 0.70, 0.80, 0.08])
pt = torch.tensor([p], dtype=torch.float32)

y0 = np.array([-1.0, -0.5], dtype=np.float64)
curr = torch.tensor([[-1.0, -0.5]], dtype=torch.float32)

# Ground truth rollout
gt_traj = simulate_trajectory(y0, 3000, DT, p)
gt_per, gt_amp = measure_period_and_amplitude(gt_traj[:, 0], dt=DT)
print(f"Ground Truth RK4: Period = {gt_per:.4f} s | Amp = {gt_amp:.4f}")

# Model rollout
pred_v = []
with torch.no_grad():
    for _ in range(3000):
        curr = m.rk4_step_forward(curr, pt)
        pred_v.append(curr[0, 0].item())

pred_v = np.array(pred_v)
per, amp = measure_period_and_amplitude(pred_v, dt=DT)
print(f"Loaded fhn_sciml_s42.pt: Period = {per:.4f} s | Amp = {amp:.4f}")
