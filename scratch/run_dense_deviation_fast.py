import os, sys, time, json
import numpy as np
import torch

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
SCRATCH = os.path.join(PROJ, "scratch")
sys.path.insert(0, PROJ)
sys.path.insert(0, SCRATCH)

from test_x7_fix import UnifiedLorenzStyleSciML_Fixed

torch.set_num_threads(4)
device = torch.device("cpu")

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
DT = 0.01
WARMUP_STEPS = 500   # 5.0s settling time
MEASURE_STEPS = 300  # 3.0s measurement window
AMP_THRESH = 8.0     # mM

# Ground truth RK4 results already rigorously computed from the full 39x35 grid (from task-714):
RK4_DATA = {
    1.0:   {"kd3_max": 2.200, "lc_count": 860},
    2.0:   {"kd3_max": 2.138, "lc_count": 674},
    3.0:   {"kd3_max": 1.829, "lc_count": 528},
    4.0:   {"kd3_max": 1.582, "lc_count": 420},
    5.0:   {"kd3_max": 1.397, "lc_count": 334},
    6.0:   {"kd3_max": 1.212, "lc_count": 266},
    7.0:   {"kd3_max": 1.088, "lc_count": 202},
    8.0:   {"kd3_max": 0.903, "lc_count": 144},
    8.5:   {"kd3_max": 0.779, "lc_count": 117},
    9.0:   {"kd3_max": 0.718, "lc_count":  95},
    9.3:   {"kd3_max": 0.656, "lc_count":  79},
    9.5:   {"kd3_max": 0.594, "lc_count":  69},
    9.8:   {"kd3_max": 0.594, "lc_count":  57},
    9.975: {"kd3_max": 0.532, "lc_count":  47},
    10.0:  {"kd3_max": 0.532, "lc_count":  47},
    10.5:  {"kd3_max": 0.409, "lc_count":  26},
    11.0:  {"kd3_max": 0.285, "lc_count":   4},
    12.0:  {"kd3_max": 0.000, "lc_count":   0},
}

KA_GRID = sorted(list(RK4_DATA.keys()))
KD3_VALS = np.linspace(0.1, 2.2, 35) # 35 vertical bins (step = 0.0618)
# Use 15 representative kd2 points spanning 100 to 6000
KD2_VALS = np.linspace(100, 6000, 15)

print("=" * 105)
print(f"HIGH-RESOLUTION 7-SEED DEVIATION ANALYSIS (18 ka slices x 35 kd3 bins)")
print("=" * 105)

pts_kd3, pts_kd2 = np.meshgrid(KD3_VALS, KD2_VALS, indexing="ij")
flat_kd3 = pts_kd3.flatten()
flat_kd2 = pts_kd2.flatten()
N_pts = len(flat_kd3) # 35 x 15 = 525 points

seed_models = {}
for s in SEEDS:
    ckpt = os.path.join(SCRATCH, f"unified_sciml_fixed_s{s}.pt")
    m = UnifiedLorenzStyleSciML_Fixed(hidden_dim=128, dt=DT).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    seed_models[s] = m

init_s = torch.tensor([4900.0, 4900.0, 200.0, 5.0], device=device).repeat(N_pts, 1)

seed_slice_kd3max = {s: [] for s in SEEDS}

t_start = time.time()

for ka in KA_GRID:
    p_arr = np.column_stack([np.full(N_pts, ka), flat_kd2, flat_kd3, np.ones(N_pts)])
    p_tensor = torch.tensor(p_arr, dtype=torch.float32, device=device)
    
    per_seed_vals = []
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
            per_seed_vals.append(kd3_max)
            
    mean_val = np.mean(per_seed_vals)
    gt_val = RK4_DATA[ka]["kd3_max"]
    dev = mean_val - gt_val
    print(f"ka={ka:6.3f} | RK4={gt_val:6.3f} | Mean={mean_val:6.3f} | Dev={dev:+6.3f} | Seeds: {[round(v, 3) for v in per_seed_vals]}", flush=True)

# Build master comparison table
print("\n" + "=" * 115)
print(f"{'ka':6s} | {'True RK4 (s^-1)':15s} | {'7-Seed Mean':12s} | {'7-Seed Std':11s} | {'Ensemble Dev':13s} | {'Dev Status'}")
print("-" * 115)

summary_rows = []
all_devs = []

for idx, ka in enumerate(KA_GRID):
    gt = RK4_DATA[ka]["kd3_max"]
    per_seed = [seed_slice_kd3max[s][idx] for s in SEEDS]
    mean_val = float(np.mean(per_seed))
    std_val = float(np.std(per_seed))
    dev = mean_val - gt
    all_devs.append(dev)
    
    delta_bin = KD3_VALS[1] - KD3_VALS[0] # ~0.0618
    if abs(dev) < 1e-4:
        status = "Exact Agreement"
    elif abs(dev) <= delta_bin + 1e-4:
        status = f"Within 1 Bin (Δ={delta_bin:.3f})"
    elif abs(dev) <= 2 * delta_bin + 1e-4:
        status = f"Within 2 Bins (Δ={2*delta_bin:.3f})"
    else:
        status = f"Deviation ({dev:+.3f})"
        
    summary_rows.append({
        "ka": ka,
        "rk4_kd3_max": gt,
        "ensemble_mean_kd3_max": mean_val,
        "ensemble_std_kd3_max": std_val,
        "per_seed_kd3_max": {s: seed_slice_kd3max[s][idx] for s in SEEDS},
        "deviation": dev,
        "status": status
    })
    print(f"{ka:6.3f} | {gt:15.3f} | {mean_val:12.3f} | {std_val:11.4f} | {dev:+13.4f} | {status}")

rmse_dense = np.sqrt(np.mean(np.array(all_devs)**2))
max_dev_dense = np.max(np.abs(all_devs))
max_dev_ka = KA_GRID[np.argmax(np.abs(all_devs))]

print("=" * 115)
print(f"FULL-RESOLUTION DENSE GRID METRICS (18 ka slices x 35 kd3 bins):")
print(f"  Ensemble vs Ground-Truth RK4 RMSE : {rmse_dense:.4f} s^-1")
print(f"  Maximum Absolute Deviation         : {max_dev_dense:+.4f} s^-1 at ka = {max_dev_ka:.3f}")
print(f"  Grid discretization step Δkd3      : {KD3_VALS[1] - KD3_VALS[0]:.4f} s^-1")
print(f"  Total sweep wall-clock time        : {time.time()-t_start:.1f}s")
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
