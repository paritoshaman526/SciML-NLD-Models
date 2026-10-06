import os, sys, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = r"c:\Users\satya\Downloads\Van Der Pol"
SCRATCH = os.path.join(PROJ, "scratch")
sys.path.insert(0, PROJ)
sys.path.insert(0, SCRATCH)

from test_x7_fix import UnifiedLorenzStyleSciML_Fixed

torch.set_num_threads(4)

# 1. Load Ground Truth RK4 Results
json_path = os.path.join(SCRATCH, "bifurcation_surface_results.json")
with open(json_path) as f:
    gt_data = json.load(f)

KA_SLICES = np.array(gt_data["ka_slices"])   # [1.0, 2.5, 5.0, 8.5, 12.0]
KD2_VALS  = np.array(gt_data["grid_kd2"])    # 39 points: 100 to 6000
KD3_VALS  = np.array(gt_data["grid_kd3"])    # 35 points: 0.1 to 2.2

# The benchmark neural evaluation grid from bifurcation_plots.py
NN_KD2 = np.array(gt_data["nn_kd2"])        # 15 points
NN_KD3 = np.array(gt_data["nn_kd3"])        # 15 points

def extract_kd3max_from_cls(cls_matrix, kd3_arr):
    # cls_matrix shape (n_kd3, n_kd2)
    n_kd3 = cls_matrix.shape[0]
    for j in range(n_kd3 - 1, -1, -1):
        if np.any(cls_matrix[j, :] == "LC"):
            return kd3_arr[j]
    return 0.0

rk4_kd3max = []
for ka in KA_SLICES:
    cls_mat = np.array(gt_data["slices"][str(ka)]["rk4_cls"])
    rk4_kd3max.append(extract_kd3max_from_cls(cls_mat, KD3_VALS))
rk4_kd3max = np.array(rk4_kd3max)

p_rk4 = np.polyfit(KA_SLICES, rk4_kd3max, deg=2)
rmse_rk4 = np.sqrt(np.mean((np.polyval(p_rk4, KA_SLICES) - rk4_kd3max)**2))
print("=" * 80, flush=True)
print("GROUND TRUTH RK4 HOPF BOUNDARY:", flush=True)
print(f"  ka slices: {KA_SLICES}", flush=True)
print(f"  RK4 kd3_max: {rk4_kd3max}", flush=True)
print(f"  RK4 Poly (deg 2): {p_rk4} | Fit RMSE: {rmse_rk4:.4f}", flush=True)
print("=" * 80, flush=True)

# 2. Evaluate All 7 Seeds with Calibrated Parameters
SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
WARMUP_STEPS  = 1000  # 10.0 s settling time
MEASURE_STEPS = 800   # 8.0 s measurement window
AMP_THRESH    = 8.0   # 8.0 mM calibrated against analytical Hopf point
DT            = 0.01

device = torch.device("cpu")
print(f"\nSimulating all 7 seeds on benchmark grid: {len(KA_SLICES)} ka x {len(NN_KD3)} kd3 x {len(NN_KD2)} kd2 (225 pts/slice)...", flush=True)
print(f"Warmup: {WARMUP_STEPS*DT:.1f}s | Measure: {MEASURE_STEPS*DT:.1f}s | Threshold: {AMP_THRESH} mM\n", flush=True)

seed_kd3max = {}
seed_cls = {}

# Also track with old threshold 20.0 to compare directly!
seed_kd3max_20 = {}

for s in SEEDS:
    ckpt_path = os.path.join(SCRATCH, f"unified_sciml_fixed_s{s}.pt")
    model = UnifiedLorenzStyleSciML_Fixed(hidden_dim=128, dt=DT).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    kd3max_per_ka = []
    kd3max_per_ka_20 = []
    cls_dict = {}

    t0 = time.time()
    for ka in KA_SLICES:
        params_list = []
        for kd3 in NN_KD3:
            for kd2 in NN_KD2:
                params_list.append([ka, kd2, kd3, 1.0])
        p_tensor = torch.tensor(params_list, dtype=torch.float32, device=device)
        B = len(p_tensor)

        init_s = torch.tensor([4900.0, 4900.0, 200.0, 5.0], device=device).repeat(B, 1)
        with torch.no_grad():
            curr = init_s
            for _ in range(WARMUP_STEPS):
                curr = model(curr, p_tensor)
            sig_min = curr[:, 2].clone()
            sig_max = curr[:, 2].clone()
            for _ in range(MEASURE_STEPS):
                curr = model(curr, p_tensor)
                sig_min = torch.minimum(sig_min, curr[:, 2])
                sig_max = torch.maximum(sig_max, curr[:, 2])
            amp = (sig_max - sig_min) / 2.0
            
            # Calibrated threshold (8.0 mM)
            is_lc = (amp > AMP_THRESH).cpu().numpy().reshape(len(NN_KD3), len(NN_KD2))
            cls_mat = np.where(is_lc, "LC", "SS")
            cls_dict[str(ka)] = cls_mat
            kd3max_per_ka.append(extract_kd3max_from_cls(cls_mat, NN_KD3))
            
            # Old threshold (20.0 mM)
            is_lc_20 = (amp > 20.0).cpu().numpy().reshape(len(NN_KD3), len(NN_KD2))
            cls_mat_20 = np.where(is_lc_20, "LC", "SS")
            kd3max_per_ka_20.append(extract_kd3max_from_cls(cls_mat_20, NN_KD3))

    seed_kd3max[s] = np.array(kd3max_per_ka)
    seed_kd3max_20[s] = np.array(kd3max_per_ka_20)
    seed_cls[s] = cls_dict
    print(f"  Seed {s:>4}: kd3_max(ka) = {seed_kd3max[s]} (took {time.time()-t0:.1f}s)", flush=True)

# 3. Quantitative Analysis Across All 7 Seeds
print("\n" + "=" * 90, flush=True)
print("7-SEED QUANTITATIVE HOPF BOUNDARY COMPARISON (AMP_THRESH = 8.0 mM, WARMUP=10s, MEASURE=8s)", flush=True)
print("=" * 90, flush=True)
header = f"{'ka':>6} | {'RK4 (True)':>10} | " + " | ".join([f"S{s}" for s in SEEDS]) + f" | {'Mean':>7} | {'Std':>6} | {'Ensemble':>8}"
print(header, flush=True)
print("-" * len(header), flush=True)

all_seed_matrix = np.array([seed_kd3max[s] for s in SEEDS]) # shape (7, 5)
mean_kd3max = np.mean(all_seed_matrix, axis=0)
std_kd3max  = np.std(all_seed_matrix, axis=0)

# Compute 7-seed majority vote boundary (>= 4 of 7 seeds)
ensemble_kd3max = []
for k_idx, ka in enumerate(KA_SLICES):
    votes = np.zeros((len(NN_KD3), len(NN_KD2)), dtype=int)
    for s in SEEDS:
        votes += (seed_cls[s][str(ka)] == "LC").astype(int)
    majority_lc = (votes >= 4)
    ens_mat = np.where(majority_lc, "LC", "SS")
    ensemble_kd3max.append(extract_kd3max_from_cls(ens_mat, NN_KD3))
ensemble_kd3max = np.array(ensemble_kd3max)

for i, ka in enumerate(KA_SLICES):
    s_vals = " | ".join([f"{seed_kd3max[s][i]:5.3f}" for s in SEEDS])
    print(f"{ka:6.1f} | {rk4_kd3max[i]:10.3f} | {s_vals} | {mean_kd3max[i]:7.3f} | {std_kd3max[i]:6.3f} | {ensemble_kd3max[i]:8.3f}", flush=True)

# 4. Polynomial Fit & RMSE Metrics
ka_dense = np.linspace(1.0, 12.0, 200)
fit_rk4 = np.polyval(p_rk4, ka_dense)

p_ens = np.polyfit(KA_SLICES, ensemble_kd3max, deg=2)
fit_ens = np.polyval(p_ens, ka_dense)

ens_dev = ensemble_kd3max - rk4_kd3max
ens_rmse = np.sqrt(np.mean(ens_dev**2))
ens_max_dev = np.max(np.abs(ens_dev))
ens_max_ka = KA_SLICES[np.argmax(np.abs(ens_dev))]

print("\n" + "=" * 90, flush=True)
print("QUADRATIC FIT & DEVIATION METRICS:", flush=True)
print("=" * 90, flush=True)
print(f"RK4 Ground Truth Poly:        {p_rk4} | Fit RMSE: {rmse_rk4:.4f}", flush=True)
print(f"7-Seed Ensemble Poly:         {p_ens} | Fit RMSE: {np.sqrt(np.mean((np.polyval(p_ens, KA_SLICES) - ensemble_kd3max)**2)):.4f}", flush=True)
print(f"Ensemble vs RK4 RMS Error:    {ens_rmse:.4f} s^-1", flush=True)
print(f"Ensemble Max Deviation:       {ens_max_dev:+.4f} s^-1 at ka = {ens_max_ka}", flush=True)
print(f"Earlier Untuned Max Dev:      +0.700 s^-1 at ka = 12.0", flush=True)
print(f"Accuracy Improvement at ka=12: from +0.700 down to {ens_dev[-1]:+.4f} s^-1", flush=True)

# Save results JSON
summary = {
    "rk4_poly": p_rk4.tolist(),
    "ensemble_poly": p_ens.tolist(),
    "ka_slices": KA_SLICES.tolist(),
    "rk4_kd3max": rk4_kd3max.tolist(),
    "ensemble_kd3max": ensemble_kd3max.tolist(),
    "seed_kd3max": {str(s): seed_kd3max[s].tolist() for s in SEEDS},
    "ens_rmse": float(ens_rmse),
    "ens_max_dev": float(ens_max_dev),
    "ens_dev_at_12": float(ens_dev[-1])
}
with open(os.path.join(SCRATCH, "seven_seed_evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# 5. Generate Publication Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
for idx, s in enumerate(SEEDS):
    ax1.plot(KA_SLICES, seed_kd3max[s], 'o--', color=colors[idx], alpha=0.60, markersize=5, label=f"Seed {s}")

# 7-Seed Ensemble Mean & Band
ax1.fill_between(KA_SLICES, mean_kd3max - std_kd3max, mean_kd3max + std_kd3max, color='#406ea0', alpha=0.20, label=r"7-Seed $\pm 1\sigma$")
ax1.plot(ka_dense, fit_ens, '-', color='#1a365d', linewidth=2.8, label="7-Seed Ensemble Poly")
ax1.plot(ka_dense, fit_rk4, 'r-', linewidth=2.5, label=f"True RK4 (RMSE={rmse_rk4:.3f})")
ax1.plot(KA_SLICES, rk4_kd3max, 'r*', markersize=11, label="True RK4 Slices")

ax1.set_xlabel(r"$k_a$ (transcription activation)", fontsize=12)
ax1.set_ylabel(r"Hopf Boundary $k_{d3,\max}$ ($s^{-1}$)", fontsize=12)
ax1.set_title("7-Seed Boundary Overlay vs True RK4 Hopf Curve", fontsize=13, fontweight='bold')
ax1.set_xlim(0.5, 13.0); ax1.set_ylim(-0.1, 2.5)
ax1.grid(True, linestyle=":", alpha=0.45)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

# Right: Deviations vs ka
for idx, s in enumerate(SEEDS):
    ax2.plot(KA_SLICES, seed_kd3max[s] - rk4_kd3max, 'o--', color=colors[idx], alpha=0.55, markersize=5, label=f"Seed {s}")
ax2.plot(KA_SLICES, ensemble_kd3max - rk4_kd3max, 's-', color='#1a365d', linewidth=2.8, label=f"Ensemble (RMS = {ens_rmse:.3f} s⁻¹)")
ax2.axhline(0.0, color='r', linestyle='--', linewidth=1.5, label="Zero Error (Exact RK4)")
ax2.axhline(0.700, color='gray', linestyle=':', linewidth=1.5, label="Earlier Untuned Dev (+0.700)")

ax2.set_xlabel(r"$k_a$", fontsize=12)
ax2.set_ylabel(r"Deviation $\Delta k_{d3,\max}$ ($s^{-1}$)", fontsize=12)
ax2.set_title(f"Per-Seed Error Comparison vs RK4 Ground Truth", fontsize=13, fontweight='bold')
ax2.set_xlim(0.5, 13.0); ax2.set_ylim(-0.4, 0.85)
ax2.grid(True, linestyle=":", alpha=0.45)
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()
out_fig = os.path.join(SCRATCH, "seven_seed_boundary_overlay.png")
fig.savefig(out_fig, dpi=300)
print(f"\nOverlay figure successfully saved to: {out_fig}", flush=True)
