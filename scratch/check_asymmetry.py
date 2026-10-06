import os
import numpy as np

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
data = np.load(os.path.join(SCRATCH_DIR, "vdp_settled_bifurcation_data.npz"))

mu_sweep = data["mu_sweep"]
gt_upper = data["gt_upper"]
gt_lower = data["gt_lower"]
a_uppers = data["a_uppers"]
a_lowers = data["a_lowers"]
a_u_mean = data["a_u_mean"]
a_l_mean = data["a_l_mean"]
b_u = data["b_u"]
b_l = data["b_l"]
base_u = data["base_u"]
base_l = data["base_l"]

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

mask = mu_sweep > 0.05
mu_active = mu_sweep[mask]

print("=" * 80)
print("NUMERICAL ASYMMETRY AUDIT: Delta_sym = x_max + x_min")
print("Analytical Ground Truth Expectation: Delta_sym = 0 due to (x, v) <-> (-x, -v) symmetry")
print("=" * 80)

# Ground Truth
gt_asym = gt_upper[mask] + gt_lower[mask]
print(f"Ground Truth RK4:")
print(f"  Max  |x_max + x_min|: {np.max(np.abs(gt_asym)):.8f}")
print(f"  Mean |x_max + x_min|: {np.mean(np.abs(gt_asym)):.8f}")

# Model A Consensus
a_mean_asym = a_u_mean[mask] + a_l_mean[mask]
print(f"\nModel A (7-Seed Mean Consensus):")
print(f"  Max  |x_max + x_min|: {np.max(np.abs(a_mean_asym)):.6f}")
print(f"  Mean |x_max + x_min|: {np.mean(np.abs(a_mean_asym)):.6f}")

# Model A Per-Seed
print(f"\nModel A Per-Seed Breakdown:")
print(f"{'Seed':<8} | {'Max |x_max + x_min|':<20} | {'Mean |x_max + x_min|':<20} | {'Mean x_max':<12} | {'Mean x_min':<12}")
print("-" * 78)
for i, s in enumerate(SEEDS):
    asym_s = a_uppers[i, mask] + a_lowers[i, mask]
    print(f"{s:<8d} | {np.max(np.abs(asym_s)):<20.6f} | {np.mean(np.abs(asym_s)):<20.6f} | {np.mean(a_uppers[i, mask]):<12.4f} | {np.mean(a_lowers[i, mask]):<12.4f}")

# Model B Residual
b_asym = b_u[mask] + b_l[mask]
print(f"\nModel B (Physics-Residual Seed 42):")
print(f"  Max  |x_max + x_min|: {np.max(np.abs(b_asym)):.6f}")
print(f"  Mean |x_max + x_min|: {np.mean(np.abs(b_asym)):.6f}")

# Baseline MLP
base_asym = base_u[mask] + base_l[mask]
print(f"\nBaseline Generic MLP (Seed 42):")
print(f"  Max  |x_max + x_min|: {np.max(np.abs(base_asym)):.6f}")
print(f"  Mean |x_max + x_min|: {np.mean(np.abs(base_asym)):.6f}")

# Detailed Table across mu
print("\n" + "=" * 90)
print(f"{'mu':<6} | {'GT x_max':<10} | {'GT x_min':<10} | {'GT sum':<10} | {'Model A x_max':<14} | {'Model A x_min':<14} | {'Model A sum':<12}")
print("-" * 90)
for idx in np.where(mask)[0][::2]:
    mu_val = mu_sweep[idx]
    gt_s = gt_upper[idx] + gt_lower[idx]
    a_s = a_u_mean[idx] + a_l_mean[idx]
    print(f"{mu_val:<+6.2f} | {gt_upper[idx]:<+10.4f} | {gt_lower[idx]:<+10.4f} | {gt_s:<+10.6f} | {a_u_mean[idx]:<+14.4f} | {a_l_mean[idx]:<+14.4f} | {a_s:<+12.6f}")
print("=" * 90)
