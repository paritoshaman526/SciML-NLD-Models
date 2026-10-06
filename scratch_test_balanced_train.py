import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import numpy as np
import time

torch.set_num_threads(1)
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed
from run_mega_parameter_sweep import SWEEP_CASES
from run_complete_evaluation_suite import analyze_signal
from train_perfect_triad_v2 import KineticFiLMEngine

set_seed(42)

# Oscillatory cases to include with balanced representation
OSC_PARAMS = [
    (1.0, 3000.0, 0.80, 1.0),   # X7
    (2.5, 3200.0, 0.80, 1.0),   # X8
    (3.5, 2600.0, 1.10, 1.0),   # X9
    (4.5, 3500.0, 1.30, 1.0),   # X10
    (5.0, 5200.0, 0.35, 1.0),   # X11
    (8.5, 4000.0, 0.85, 1.0),   # X12
    # Additional interpolation anchors
    (2.0, 3000.0, 0.60, 1.0),
    (4.0, 3000.0, 1.00, 1.0),
    (6.0, 4200.0, 1.00, 1.0),
    (8.0, 4500.0, 0.60, 1.0),
]

print("Simulating balanced dataset...", flush=True)
trajs = {}
for p in OSC_PARAMS:
    p_arr = np.array(p, dtype=np.float64)
    y = DEFAULT_GUESS.copy()
    for _ in range(int(6.0 / DT)):
        y = rk4_step(y, DT, p_arr)
    # simulate 1.5 seconds (15000 steps)
    steps = int(1.5 / DT)
    traj = simulate_trajectory(y, steps, DT, p_arr)
    trajs[p] = torch.tensor(traj, dtype=torch.float32)

print("Training FiLM with balanced case sampling...", flush=True)
model = KineticFiLMEngine()
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600, eta_min=1e-5)

steps_total = 600
weights = torch.tensor([1.0, 1.0, 3.0, 8.0]) / model.state_scales

t0 = time.time()
for step in range(1, steps_total + 1):
    K = int(np.interp(step, [1, 250, steps_total], [6, 16, 25]))
    
    # Balanced: sample random case from OSC_PARAMS
    p_indices = np.random.choice(len(OSC_PARAMS), size=32, replace=True)
    states, params, futures = [], [], []
    for p_idx in p_indices:
        p = OSC_PARAMS[p_idx]
        tr = trajs[p]
        start = np.random.randint(0, tr.shape[0] - K - 1)
        states.append(tr[start])
        futures.append(tr[start + 1 : start + K + 1])
        params.append(p)
    
    s_b = torch.stack(states)
    p_b = torch.tensor(params, dtype=torch.float32)
    f_b = torch.stack(futures)
    c_tot0 = s_b[:, 0:1] + s_b[:, 1:2] + s_b[:, 2:3]

    curr = s_b
    loss = 0.0
    for k in range(K):
        curr = model(curr, p_b, target_ctot=c_tot0)
        loss = loss + torch.mean(((curr - f_b[:, k, :]) * weights)**2)
    loss = loss / K

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()
    sched.step()

    if step % 200 == 0:
        print(f"  Step {step}/{steps_total} | Loss: {loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

print("Evaluating on X7-X12...", flush=True)
model.eval()
with torch.no_grad():
    for c in SWEEP_CASES[6:]:
        p_arr = np.array(c['params'], dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(c['warmup'] / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()
        steps = min(int(c['duration'] / DT), 8000)
        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)

        p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
        curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]

        pred = [ic.copy()]
        for _ in range(steps):
            curr = model(curr, p_t, target_ctot=c_tot)
            pred.append(curr.squeeze(0).cpu().numpy())
        pred = np.array(pred)[:steps]

        sig_gt = analyze_signal(gt_traj[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
        sig_p = analyze_signal(pred[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])

        print(f"{c['id']}: GT Amp={sig_gt['steady_amp']:.1f} (ptp={np.ptp(gt_traj[:, 2]):.1f}) | Pred Amp={sig_p['steady_amp']:.1f} (ptp={np.ptp(pred[:, 2]):.1f}) | GT Per={sig_gt['period']:.4f}s | Pred Per={sig_p['period']:.4f}s")
