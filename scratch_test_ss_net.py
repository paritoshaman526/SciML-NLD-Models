import torch
import torch.nn as nn
import numpy as np
from train_perfect_film_triad import ALL_TRAIN_SS, compute_exact_ss

# Collect training pairs
p_list, y_list = [], []
for p in ALL_TRAIN_SS:
    y_ss = compute_exact_ss(p)
    p_list.append(p)
    y_list.append(y_ss)

# Add some grid points in the steady state domain ka in [10, 15], kd2 in [4000, 6500], kd3 in [0.1, 0.7]
for ka in [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]:
    for kd2 in [4200.0, 4800.0, 5400.0, 6000.0]:
        for kd3 in [0.15, 0.30, 0.45, 0.60]:
            p = (ka, kd2, kd3, 1.0)
            y_ss = compute_exact_ss(p)
            p_list.append(p)
            y_list.append(y_ss)

P_train = torch.tensor(p_list, dtype=torch.float32)
Y_train = torch.tensor(y_list, dtype=torch.float32)

param_scales = torch.tensor([10., 5000., 1.5, 1.])
state_scales = torch.tensor([5000., 5000., 300., 10.])

ss_net = nn.Sequential(
    nn.Linear(4, 48), nn.SiLU(),
    nn.Linear(48, 48), nn.SiLU(),
    nn.Linear(48, 4)
)

opt = torch.optim.Adam(ss_net.parameters(), lr=3e-3)
for epoch in range(1200):
    pred = ss_net(P_train / param_scales) * state_scales
    loss = torch.mean(((pred - Y_train) / state_scales)**2)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"Final SS net loss: {loss.item():.4e}")

# Test on X1-X6
from run_mega_parameter_sweep import SWEEP_CASES
print("\nTesting autonomous SS predictor on X1-X6 (no target leakage):")
for c in SWEEP_CASES[:6]:
    p = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
    y_true = compute_exact_ss(c['params'])
    with torch.no_grad():
        y_pred = (ss_net(p / param_scales) * state_scales).numpy()[0]
    err_max = np.max(np.abs(y_pred - y_true))
    err_pct = err_max / y_true[0] * 100.0
    print(f"{c['id']}: GT m1*={y_true[2]:.3f}, Pred m1*={y_pred[2]:.3f}, max error={err_max:.4f} mM ({err_pct:.4f}%)")
