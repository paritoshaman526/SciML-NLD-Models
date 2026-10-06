import os, sys, time, json
import numpy as np
import torch

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
SCRATCH = os.path.join(PROJ, "scratch")
sys.path.insert(0, PROJ)
sys.path.insert(0, SCRATCH)

from test_x7_fix import UnifiedLorenzStyleSciML_Fixed
from fast_rk4_extinction import KD2_VALS, KD3_VALS, scan_slice_rk4

torch.set_num_threads(4)
device = torch.device("cpu")

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
DT = 0.01
WARMUP_STEPS = 1200
MEASURE_STEPS = 500
AMP_THRESH = 8.0

# 16 dense ka values spanning the entire active and extinction zone:
KA_GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.3, 9.5, 9.8, 9.975, 10.0, 10.5, 11.0, 12.0]

print("=" * 95)
print(f"DENSE HIGH-RESOLUTION BIFURCATION DEVIATION ANALYSIS (18 ka slices x 35 kd3 x 39 kd2 = 24,570 pts/seed)")
print("=" * 95)

# A. Evaluate Ground Truth RK4
print("\n[1/3] Computing Ground Truth RK4 across all 18 ka slices...")
rk4_results = {}
for ka in KA_GRID:
    t0 = time.time()
    amps = scan_slice_rk4(ka, KD2_VALS, KD3_VALS, WARMUP_STEPS, MEASURE_STEPS, DT, AMP_THRESH)
    is_lc = amps > AMP_THRESH
    kd3_max = 0.0
    for i in range(len(KD3_VALS) - 1, -1, -1):
        if np.any(is_lc[i, :]):
            kd3_max = KD3_VALS[i]
            break
    rk4_results[ka] = {
        "kd3_max": float(kd3_max),
        "lc_count": int(np.sum(is_lc)),
        "max_amp": float(np.max(amps))
    }
    print(f"  RK4 ka = {ka:6.3f} | kd3_max = {kd3_max:6.3f} s^-1 | LC Count = {rk4_results[ka]['lc_count']:4d} | Max Amp = {rk4_results[ka]['max_amp']:7.2f} mM ({time.time()-t0:.2f}s)")

# B. Evaluate All 7 Seeds (Batched in PyTorch)
print("\n[2/3] Evaluating 7 SciML Seeds on the exact same high-resolution grid (35 kd3 x 39 kd2 = 1,365 pts/slice)...")

# Build coordinate tensor (1,365 points)
pts_kd3, pts_kd2 = np.meshgrid(KD3_VALS, KD2_VALS, indexing="ij")
flat_kd3 = pts_kd3.flatten()
flat_kd2 = pts_kd2.flatten()
N_pts = len(flat_kd3)

seed_models = {}
for s in SEEDS:
    ckpt = os.path.join(SCRATCH, f"unified_sciml_fixed_s{s}.pt")
    m = UnifiedLorenzStyleSciML_Fixed(hidden_dim=128, dt=DT).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    seed_models[s] = m

seed_slice_kd3max = {s: [] for s in SEEDS}
seed_slice_lc_count = {s: [] for s in SEEDS}

# Pre-allocate init state
init_s = torch.tensor([4900.0, 4900.0, 200.0, 5.0], device=device).repeat(N_pts, 1)

for ka in KA_GRID:
    p_arr = np.column_stack([np.full(N_pts, ka), flat_kd2, flat_kd3, np.ones(N_pts)])
    p_tensor = torch.tensor(p_arr, dtype=torch.float32, device=device)
    
    per_seed_kd3max_at_ka = []
    for s in SEEDS:
        m = seed_models[s]
        with torch.no_grad():
            curr = init_s
            for _ in range(WARMUP_STEPS):
                curr = m(curr, p_tensor)
            s_min = curr[:, 2].clone()
            s_max = curr[:, 2].clone()
            for _ in range(MEASURE_STEPS):
                curr = m(curr, p_tensor)
                s_min = torch.minimum(s_min, curr[:, 2])
                s_max = torch.maximum(s_max, curr[:, 2])
            amp = ((s_max - s_min) / 2.0).cpu().numpy().reshape(len(KD3_VALS), len(KD2_VALS))
            is_lc = amp > AMP_THRESH
            kd3_max = 0.0
            for i in range(len(KD3_VALS) - 1, -1, -1):
                if np.any(is_lc[i, :]):
                    kd3_max = KD3_VALS[i]
                    break
            seed_slice_kd3max[s].append(kd3_max)
            seed_slice_lc_count[s].append(int(np.sum(is_lc)))
            per_seed_kd3max_at_ka.append(kd3_max)
    print(f"  SciML ka = {ka:6.3f} | S42={per_seed_kd3max_at_ka[0]:.3f}, S101={per_seed_kd3max_at_ka[1]:.3f}, S777={per_seed_kd3max_at_ka[2]:.3f}, S999={per_seed_kd3max_at_ka[3]:.3f}, S1234={per_seed_kd3max_at_ka[4]:.3f}, S2024={per_seed_kd3max_at_ka[5]:.3f}, S5555={per_seed_kd3max_at_ka[6]:.3f}")

# C. Compute Pointwise Deviations, Ensemble Mean, RMSE, and Save Comprehensive Summary
print("\n[3/3] Statistical Summary & Deviation Analysis:")
print("=" * 115)
print(f"{'ka':6s} | {'True RK4 kd3_max':17s} | {'7-Seed Mean':12s} | {'7-Seed Std':11s} | {'Ensemble Dev':13s} | {'Dev Status'}")
print("-" * 115)

summary_rows = []
all_devs = []

for idx, ka in enumerate(KA_GRID):
    gt = rk4_results[ka]["kd3_max"]
    per_seed = [seed_slice_kd3max[s][idx] for s in SEEDS]
    mean_val = float(np.mean(per_seed))
    std_val = float(np.std(per_seed))
    dev = mean_val - gt
    all_devs.append(dev)
    
    if abs(dev) < 1e-4:
        status = "Exact Agreement"
    elif abs(dev) <= (KD3_VALS[1] - KD3_VALS[0]) + 1e-4:
        status = f"Within 1 Grid Bin (Δ={KD3_VALS[1]-KD3_VALS[0]:.3f})"
    else:
        status = "Deviation"
        
    summary_rows.append({
        "ka": ka,
        "rk4_kd3_max": gt,
        "ensemble_mean_kd3_max": mean_val,
        "ensemble_std_kd3_max": std_val,
        "per_seed_kd3_max": {s: seed_slice_kd3max[s][idx] for s in SEEDS},
        "deviation": dev,
        "status": status
    })
    print(f"{ka:6.3f} | {gt:17.3f} | {mean_val:12.3f} | {std_val:11.4f} | {dev:+13.4f} | {status}")

rmse_dense = np.sqrt(np.mean(np.array(all_devs)**2))
max_dev_dense = np.max(np.abs(all_devs))
max_dev_ka = KA_GRID[np.argmax(np.abs(all_devs))]

print("=" * 115)
print(f"FULL-RESOLUTION DENSE GRID METRICS (Across all 18 ka slices):")
print(f"  Ensemble vs Ground-Truth RK4 RMSE : {rmse_dense:.4f} s^-1")
print(f"  Maximum Absolute Deviation         : {max_dev_dense:+.4f} s^-1 at ka = {max_dev_ka:.3f}")
print(f"  Grid discretization step Δkd3      : {KD3_VALS[1] - KD3_VALS[0]:.4f} s^-1")
print("=" * 115)

out_json = os.path.join(SCRATCH, "dense_grid_bifurcation_deviation.json")
with open(out_json, "w") as f:
    json.dump({
        "ka_grid": KA_GRID,
        "kd3_step": float(KD3_VALS[1] - KD3_VALS[0]),
        "rmse": float(rmse_dense),
        "max_dev": float(max_dev_dense),
        "max_dev_ka": float(max_dev_ka),
        "rows": summary_rows
    }, f, indent=2)
print(f"Saved dense evaluation results to: {out_json}")
