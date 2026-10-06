import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

import numpy as np
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from run_complete_evaluation_suite import analyze_signal
from train_perfect_triad_v2 import KineticFiLMEngine

m_film = KineticFiLMEngine()
m_film.load_state_dict(torch.load("film_v2.pt", weights_only=True))
m_film.eval()

p_arr = np.array([1.0, 3000.0, 0.80, 1.0], dtype=np.float64)
y = DEFAULT_GUESS.copy()
for _ in range(int(6.0 / DT)):
    y = rk4_step(y, DT, p_arr)
ic = y.copy()

steps = 7000
gt_traj = simulate_trajectory(ic, steps, DT, p_arr)

p_t = torch.tensor(p_arr, dtype=torch.float32).unsqueeze(0)
curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]

pred = [ic.copy()]
with torch.no_grad():
    for _ in range(steps):
        curr = m_film(curr, p_t, target_ctot=c_tot)
        pred.append(curr.squeeze(0).cpu().numpy())
pred = np.array(pred)[:steps]

gt_sig = analyze_signal(gt_traj[:, 2], prominence=30.0, min_dist=0.15)
pred_sig = analyze_signal(pred[:, 2], prominence=30.0, min_dist=0.15)

print(f"X7 GT: Amp={gt_sig['steady_amp']:.2f}, Period={gt_sig['period']:.4f}, min_m1={gt_traj[:,2].min():.2f}, max_m1={gt_traj[:,2].max():.2f}")
print(f"X7 FiLM: Amp={pred_sig['steady_amp']:.2f}, Period={pred_sig['period']:.4f}, min_m1={pred[:,2].min():.2f}, max_m1={pred[:,2].max():.2f}")
