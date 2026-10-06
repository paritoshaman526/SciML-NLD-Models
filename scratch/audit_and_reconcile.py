import json
import numpy as np
import sys
import os

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from train_and_evaluate_lorenz_style_sciml import find_physical_steady_state, PHYSICAL_CTOT, SWEEP_CASES
from biochemical_global_sciml import system, rk4_step, DT

print("=" * 80)
print("1. DEEP DIVE ON X4: EXACT NUMBERS AND METRICS")
print("=" * 80)

# Check transient_relaxation_data.json for X4
with open("transient_relaxation_data.json", "r") as f:
    t_data = json.load(f)

x4_t = t_data["X4"]
print(f"X4 Parameters: {x4_t['params']}")
print(f"X4 tau_phys: {x4_t['tau_phys_s']:.4f} s")
print(f"X4 True y*(m1): {x4_t['y_star_m1']:.4f} mM")
print(f"X4 Initial perturbation: +{x4_t['init_pert_mM']:.4f} mM (Start m1 = {x4_t['y_star_m1'] + x4_t['init_pert_mM']:.4f} mM)")
print(f"X4 at t = 0.5s: True RK4 residual |m1(0.5s) - y*| = {x4_t['rk4_final_residual_mM']:.4f} mM")
print(f"X4 at t = 0.5s: Model seeds gap to RK4 |m1_model(0.5s) - m1_rk4(0.5s)|:")
for s, gap in x4_t["model_seeds_gap_to_rk4"].items():
    err_ystar = x4_t["model_seeds_final_err"][s]
    print(f"  Seed {s:4s}: Gap to RK4 = {gap:.4f} mM | Drift from y* = {err_ystar:.4f} mM")

# Check basin_of_attraction_results.json for X4
with open("basin_of_attraction_results.json", "r") as f:
    b_data = json.load(f)

x4_b = b_data["X4"]
print(f"\nIn basin_of_attraction_results.json (t = 0.4s / 4000 steps):")
for s, s_info in x4_b["per_seed"].items():
    print(f"  Seed {s:4s}: Final m1 = {s_info['final_m1']:.4f} mM | Drift from y* = {s_info['drift_from_ystar_mM']:.4f} mM | % Recovered = {s_info['pct_recovered']:.2f}%")

print("\n" + "=" * 80)
print("2. EXACT B.2 NUMBERS FOR X10 ACROSS ALL 7 SEEDS")
print("=" * 80)
with open("unified_sciml_7seed_results.json", "r") as f:
    orig_res = json.load(f)

seeds = [42, 101, 777, 999, 1234, 2024, 5555]
print("Original Unified SciML (unified_sciml_7seed_results.json) for X10:")
for s in seeds:
    d = orig_res["per_seed"][str(s)]["X10"]
    print(f"  Seed {s:4d}: Pred Amp = {d['pred_val']:.4f} mM, GT = {d['gt_val']:.4f} mM, Error = {d['error']:.6f}%")

with open("scratch/all_cases_fixed_summary.json", "r") as f:
    fixed_res = json.load(f)

print("\nFixed Unified SciML (scratch/all_cases_fixed_summary.json) for X10:")
for s in seeds:
    d = fixed_res[str(s)]["X10"]
    print(f"  Seed {s:4d}: Pred Amp = {d['pred_amp']:.4f} mM, GT = {d['gt_amp']:.4f} mM, Error = {d['amp_err_pct']:.6f}%")

print("\n" + "=" * 80)
print("3. CONFIRMING X1-X6 PERTURBATION PROTOCOL (SCALE-MATCHING CHECK)")
print("=" * 80)
for c in SWEEP_CASES:
    if not c.get("is_steady_state", False):
        continue
    cid = c["id"]
    p = c["params"]
    ystar = find_physical_steady_state(p, c_tot=PHYSICAL_CTOT)
    # Check what perturbation formula was used in basin_of_attraction_results.json
    b_c = b_data[cid]
    init_pert = b_c["perturbation_m1_mM"]
    ratio = init_pert / ystar[2]
    print(f"{cid:5s} | Params: {str(p):25s} | True y*(m1): {ystar[2]:8.2f} mM | Delta m1: {init_pert:8.2f} mM | Ratio Delta/y*: {ratio:.6f} ({ratio*100:.1f}%)")

print("\n" + "=" * 80)
print("4. SIDE-BY-SIDE NOISE FLOOR VS MODEL ERROR TABLE")
print("=" * 80)
with open("scratch/biochemical_noise_floor.json", "r") as f:
    nf = json.load(f)

for cid in ["X7", "X8", "X9", "X10", "X11", "X12"]:
    gt_amp = nf[cid]["gt_amp"]
    floor_amp_phase = nf[cid]["amp_noise_pct_phase"]
    floor_w1_phase = nf[cid]["w1_phase"]

    # Model error (from fixed models)
    model_errs = [fixed_res[str(s)][cid]["amp_err_pct"] for s in seeds]
    mean_err = np.mean(model_errs)
    std_err = np.std(model_errs)
    med_err = np.median(model_errs)
    min_err = np.min(model_errs)
    max_err = np.max(model_errs)

    status = "AT/BELOW NOISE FLOOR" if mean_err <= max(floor_amp_phase, 0.05) else "ABOVE NOISE FLOOR"
    print(f"{cid:5s} | Floor Amp Err: {floor_amp_phase:7.4f}% | Model Mean Amp Err: {mean_err:7.4f} ± {std_err:6.4f}% | Model Range: [{min_err:.4f}%, {max_err:.4f}%] | Floor W1: {floor_w1_phase:6.3f} mM | Status: {status}")
