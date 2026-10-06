import json
import numpy as np
import scipy.stats
import torch
import sys
import os

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import system, rk4_step, DT
from train_and_evaluate_lorenz_style_sciml import (
    SWEEP_CASES
)

# -------------------------------------------------------------------------
# B.2: Panel B Amplitude Errors across X9-X12
# -------------------------------------------------------------------------
print("\n--- B.2: Panel B Amplitude Errors across X9-X12 ---")
with open("unified_sciml_7seed_results.json", "r") as f:
    res = json.load(f)

seeds = [42, 101, 777, 999, 1234, 2024, 5555]
cases_lc = ["X9", "X10", "X11", "X12"]

for cid in cases_lc:
    amp_errs = []
    print(f"\nCase {cid}:")
    for s in seeds:
        seed_data = res["per_seed"][str(s)][cid]
        pred_amp = seed_data["pred_val"]
        gt_amp = seed_data["gt_val"]
        err_pct = seed_data["error"]
        amp_errs.append((s, err_pct))
        flag = " > 0.20% [EXCEEDS]" if err_pct > 0.20 else ""
        print(f"  Seed {s:4d}: amp = {pred_amp:.3f} (GT={gt_amp:.3f}) -> err = {err_pct:.4f}%{flag}")
    
    err_vals = [e[1] for e in amp_errs]
    above_20 = [e for e in amp_errs if e[1] > 0.20]
    print(f"  Summary for {cid}: Max = {max(err_vals):.4f}% (Seed {amp_errs[np.argmax(err_vals)][0]}), Count > 0.20%: {len(above_20)} / {len(seeds)}")

# Total count > 0.20% across all X9-X12
all_lc_above = []
for cid in cases_lc:
    for s in seeds:
        err_pct = res["per_seed"][str(s)][cid]["error"]
        if err_pct > 0.20:
            all_lc_above.append((cid, s, err_pct))

print(f"\nTotal seeds > 0.20% across all X9-X12: {len(all_lc_above)} of {len(cases_lc)*len(seeds)}")
for item in all_lc_above:
    print(f"  {item[0]} Seed {item[1]}: {item[2]:.4f}%")

# Let's also check between 0.15% and 0.20%
near_line = []
for cid in cases_lc:
    for s in seeds:
        err_pct = res["per_seed"][str(s)][cid]["error"]
        if 0.15 <= err_pct <= 0.20:
            near_line.append((cid, s, err_pct))

print(f"\nSeeds near the line (0.15% <= err <= 0.20%): {len(near_line)}")
for item in near_line:
    print(f"  {item[0]} Seed {item[1]}: {item[2]:.4f}%")

# -------------------------------------------------------------------------
# B.3: Oscillation Period vs Seed Variance Correlation (X7-X12)
# -------------------------------------------------------------------------
print("\n--- B.3: Oscillation Period vs Seed Variance Correlation (X7-X12) ---")

# Let's compute true periods for X7-X12 using RK4
lc_cases = [c for c in SWEEP_CASES if not c.get('is_steady_state', False)]

def get_rk4_period_and_amp(params):
    p_arr = np.array(params, dtype=np.float64)
    # Warmup
    y = np.array([5800.0, min(params[1], 5500.0), 400.0, 6.5], dtype=np.float64)
    for _ in range(int(15.0 / DT)):
        y = rk4_step(y, DT, p_arr)
    # Simulate 5.0 seconds
    traj = []
    for _ in range(int(5.0 / DT)):
        y = rk4_step(y, DT, p_arr)
        traj.append(y[2])
    traj = np.array(traj)
    # Find peaks
    peaks = []
    for i in range(1, len(traj) - 1):
        if traj[i] > traj[i-1] and traj[i] > traj[i+1] and traj[i] > np.mean(traj):
            peaks.append(i)
    if len(peaks) >= 2:
        diffs = np.diff(peaks) * DT
        period = float(np.mean(diffs))
        amp = float(np.max(traj) - np.min(traj))
    else:
        period = 0.0
        amp = 0.0
    return period, amp

all_lc_cases = ["X7", "X8", "X9", "X10", "X11", "X12"]
periods = []
amp_stds = []
amp_cvs = []

print(f"{'Case':5s} | {'GT Period (s)':13s} | {'Freq (Hz)':10s} | {'Cycles in 1.2s':14s} | {'Amp Std (%)':12s} | {'Amp CV':10s}")
print("-" * 80)

for cid in all_lc_cases:
    c_info = next(c for c in SWEEP_CASES if c['id'] == cid)
    p_gt, a_gt = get_rk4_period_and_amp(c_info['params'])
    cycles = 1.2 / p_gt if p_gt > 0 else 0
    
    seed_amp_errs = [res["per_seed"][str(s)][cid]["error"] for s in seeds]
    seed_amps = [res["per_seed"][str(s)][cid]["pred_val"] for s in seeds]
    
    amp_std = float(np.std(seed_amp_errs))
    amp_cv = float(np.std(seed_amps) / np.mean(seed_amps))
    
    periods.append(p_gt)
    amp_stds.append(amp_std)
    amp_cvs.append(amp_cv)
    
    print(f"{cid:5s} | {p_gt:13.4f} | {1.0/p_gt:10.2f} | {cycles:14.2f} | {amp_std:12.4f} | {amp_cv:10.4f}")

# Correlations
r_all, p_val_all = scipy.stats.pearsonr(periods, amp_stds)
spearman_all, sp_p_all = scipy.stats.spearmanr(periods, amp_stds)

print("\nCorrelations across ALL 6 limit cycles (X7-X12):")
print(f"  Pearson r = {r_all:.4f} (p = {p_val_all:.4f})")
print(f"  Spearman rho = {spearman_all:.4f} (p = {sp_p_all:.4f})")

# Excluding X7 (to test if hypothesis holds across the 5 stable limit cycles X8-X12)
r_sub, p_val_sub = scipy.stats.pearsonr(periods[1:], amp_stds[1:])
spearman_sub, sp_p_sub = scipy.stats.spearmanr(periods[1:], amp_stds[1:])
print("\nCorrelations across X8-X12 (excluding X7, N=5):")
print(f"  Pearson r = {r_sub:.4f} (p = {p_val_sub:.4f})")
print(f"  Spearman rho = {spearman_sub:.4f} (p = {sp_p_sub:.4f})")

# Let's also check correlation with log(periods) or inverse cycles
cycles_arr = [1.2 / p for p in periods]
r_cyc, p_cyc = scipy.stats.pearsonr(cycles_arr, amp_stds)
print(f"\nCorrelation between Training Cycles (1.2s / T) and Amp Error Std: r = {r_cyc:.4f} (p = {p_cyc:.4f})")
