import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import numpy as np
import scipy.optimize
from biochemical_global_sciml import system

def compute_exact_ss(params, target_ctot=10002.0):
    p_arr = np.asarray(params, dtype=np.float64)
    def res(x):
        d, a1, m0 = x
        m1 = target_ctot - (d + a1)
        y = np.array([d, a1, m1, m0], dtype=np.float64)
        ydot = system(y, p_arr)
        return [ydot[0], ydot[1], ydot[3]]
    x0 = [5800.0, min(p_arr[1], 5500.0), 5.0]
    opt = scipy.optimize.least_squares(
        res, x0, bounds=([0, 0, 1e-4], [target_ctot, target_ctot, 50.0]),
        xtol=1e-12, ftol=1e-12
    )
    d, a1, m0 = opt.x
    m1 = target_ctot - (d + a1)
    return np.array([d, a1, m1, m0], dtype=np.float64)

# Test a clean neural mapper with physical constraints:
# a1* = kd2 / kp (exact)
# m0* = ka * d* / (2 * kd2) (exact)
# d* = c_avail - m1* (exact)
# Only m1* needs to be mapped as a smooth function of (ka, kd2, kd3)!
import torch.nn as nn

class SteadyStatePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.register_buffer('param_scales', torch.tensor([10., 5000., 1.5, 1.]))
    
    def forward(self, params, c_tot=10002.0):
        ka = params[:, 0:1]
        kd2 = params[:, 1:2]
        kd3 = params[:, 2:3]
        kp = params[:, 3:4]
        
        a1_star = kd2 / kp
        c_avail = torch.clamp(c_tot - a1_star, min=100.0)
        
        p_norm = params / self.param_scales
        m1_star = torch.clamp(self.net(p_norm) * 500.0, min=10.0, max=1000.0)
        d_star = torch.clamp(c_avail - m1_star, min=10.0)
        m0_star = (ka * d_star) / (2.0 * kd2)
        
        return torch.cat([d_star, a1_star, m1_star, m0_star], dim=1)

# Train on diverse parameter sets
p_train, y_train = [], []
for ka in np.linspace(10.0, 15.0, 30):
    for kd2 in [4000.0, 4500.0, 5000.0, 5500.0, 6000.0]:
        for kd3 in [0.15, 0.25, 0.35, 0.50, 0.60]:
            p = (ka, kd2, kd3, 1.0)
            y = compute_exact_ss(p)
            p_train.append(p)
            y_train.append(y[2]) # train on m1*

P_t = torch.tensor(np.array(p_train), dtype=torch.float32)
Y_t = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

model = SteadyStatePredictor()
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for epoch in range(1500):
    p_norm = P_t / model.param_scales
    pred_m1 = torch.clamp(model.net(p_norm) * 500.0, min=10.0, max=1000.0)
    loss = torch.mean((pred_m1 - Y_t)**2)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"Trained SS predictor in {epoch+1} epochs | Loss (MSE): {loss.item():.4f} mM^2 (RMSE: {np.sqrt(loss.item()):.2f} mM)")

from run_mega_parameter_sweep import SWEEP_CASES
print("\nEvaluation on Test Cases X1-X6 with ZERO Target Leakage:")
for c in SWEEP_CASES[:6]:
    p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
    y_gt = compute_exact_ss(c['params'])
    with torch.no_grad():
        y_pred = model(p_t).squeeze(0).numpy()
    err = np.abs(y_pred - y_gt)
    print(f"{c['id']}: GT m1={y_gt[2]:.2f}, Pred m1={y_pred[2]:.2f} (diff={err[2]:.2f} mM) | d diff={err[0]:.2f} mM | m0 diff={err[3]:.4f} mM")
