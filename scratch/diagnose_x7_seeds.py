import os
import sys
import numpy as np
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from train_and_evaluate_lorenz_style_sciml import UnifiedLorenzStyleSciML
from run_complete_evaluation_suite import analyze_signal

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
p = (1.0, 3000.0, 0.80, 1.0)
p_arr = np.array(p, dtype=np.float64)
p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

# Warmup to limit cycle
y = DEFAULT_GUESS.copy()
for _ in range(int(6.0 / DT)):
    y = rk4_step(y, DT, p_arr)
ic = y.copy()

steps = 15000 # 1.5s
gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
sig_gt = analyze_signal(gt_traj[:, 2], prominence=30.0, min_dist=0.15)
gt_amp = float(sig_gt['steady_amp'])

print(f"Ground Truth X7: amp = {gt_amp:.2f} mM, period = {sig_gt['period']:.4f} s")
print("-" * 75)

for s in SEEDS:
    ckpt = f"unified_sciml_seed_{s}.pt"
    m = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()

    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]

    traj = [ic.copy()]
    with torch.no_grad():
        for _ in range(steps):
            curr = m(curr, p_t, target_ctot=c_tot_t)
            traj.append(curr.squeeze(0).numpy())
    traj = np.array(traj)

    m1 = traj[:, 2]
    m0 = traj[:, 3]
    sig = analyze_signal(m1, prominence=30.0, min_dist=0.15)
    pred_amp = float(sig['steady_amp'])
    err = abs(pred_amp - gt_amp) / gt_amp * 100.0

    status = "OK"
    if pred_amp < 50.0:
        status = "COLLAPSED TO FIXED POINT"
    elif pred_amp > 1500.0:
        status = "RUNAWAY EXPLOSION"

    print(f"Seed {s:4d}: amp = {pred_amp:8.2f} mM (Err: {err:6.2f}%) | m1 range: [{m1.min():6.1f}, {m1.max():7.1f}] | m0 range: [{m0.min():6.3f}, {m0.max():6.3f}] | {status}")
