import torch
import torch.nn as nn
import numpy as np
from train_perfect_film_triad import ALL_TRAIN_SS, compute_exact_ss
from run_mega_parameter_sweep import SWEEP_CASES

p_list, m1_list = [], []
for ka in np.linspace(10.0, 15.0, 25):
    for kd2 in [4000.0, 4600.0, 5000.0, 5500.0, 6000.0]:
        for kd3 in [0.15, 0.25, 0.30, 0.50, 0.60]:
            p = (ka, kd2, kd3, 1.0)
            y_ss = compute_exact_ss(p)
            p_list.append(p)
            m1_list.append(y_ss[2])

P = torch.tensor(np.array(p_list), dtype=torch.float32)
M1 = torch.tensor(np.array(m1_list), dtype=torch.float32).unsqueeze(1)
pscale = torch.tensor([10., 5000., 1.5, 1.])

net = nn.Sequential(
    nn.Linear(4, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.SiLU(),
    nn.Linear(32, 1)
)
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
for _ in range(1200):
    pred = net(P / pscale) * 400.0
    loss = torch.mean((pred - M1)**2)
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"RMSE: {np.sqrt(loss.item()):.3f} mM")
for c in SWEEP_CASES[:6]:
    p = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
    y_true = compute_exact_ss(c['params'])
    pred_m1 = (net(p / pscale) * 400.0).item()
    print(f"{c['id']}: GT={y_true[2]:.2f}, Pred={pred_m1:.2f}, Diff={abs(pred_m1 - y_true[2]):.2f} mM")
