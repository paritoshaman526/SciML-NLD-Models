import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from train_and_evaluate_lorenz_style_sciml import (
    find_physical_steady_state, SWEEP_CASES, PHYSICAL_CTOT
)
from scratch.test_x7_fix import UnifiedLorenzStyleSciML_Fixed
from run_complete_evaluation_suite import analyze_signal

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

def evaluate_fixed_models():
    print("=" * 80)
    print("EVALUATING ALL 12 REGIMES ACROSS ALL 7 SEEDS (FIXED UNIFIED ARCHITECTURE)")
    print("=" * 80)

    all_results = {}
    for s in SEEDS:
        ckpt = f"scratch/unified_sciml_fixed_s{s}.pt"
        m = UnifiedLorenzStyleSciML_Fixed(hidden_dim=128, dt=DT, m0_floor=1e-4)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()

        seed_res = {}
        for c in SWEEP_CASES:
            cid = c['id']
            p_arr = np.array(c['params'], dtype=np.float64)
            p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
            is_ss = c['is_steady_state']

            if is_ss:
                ystar = find_physical_steady_state(c['params'], c_tot=PHYSICAL_CTOT)
                delta_m1 = 0.15 * ystar[2]
                y_pert = ystar.copy()
                y_pert[2] += delta_m1
                y_pert[0] -= delta_m1
                y_pert[3] *= 1.20

                curr = torch.tensor(y_pert, dtype=torch.float32).unsqueeze(0)
                c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]
                steps = 4000
                traj = [y_pert.copy()]
                with torch.no_grad():
                    for _ in range(steps):
                        curr = m(curr, p_t, target_ctot=c_tot_t)
                        traj.append(curr.squeeze(0).numpy())
                traj = np.array(traj)
                final_drift = float(abs(traj[-1, 2] - ystar[2]))
                pct_rec = float(max(0.0, 1.0 - final_drift / delta_m1) * 100.0)
                seed_res[cid] = {
                    "regime": "Stable Focus",
                    "final_m1": float(traj[-1, 2]),
                    "drift_from_ystar_mM": final_drift,
                    "pct_recovered": pct_rec
                }
            else:
                y = DEFAULT_GUESS.copy()
                for _ in range(int(c.get('warmup', 5.0) / DT)):
                    y = rk4_step(y, DT, p_arr)
                ic = y.copy()

                steps = 15000 if cid == "X7" else 6000
                gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
                curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
                c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]

                traj = [ic.copy()]
                with torch.no_grad():
                    for _ in range(steps):
                        curr = m(curr, p_t, target_ctot=c_tot_t)
                        traj.append(curr.squeeze(0).numpy())
                traj = np.array(traj)

                sig_gt = analyze_signal(gt_traj[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
                sig_pred = analyze_signal(traj[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
                gt_amp = float(sig_gt['steady_amp'])
                pred_amp = float(sig_pred['steady_amp'])
                err = float(abs(pred_amp - gt_amp) / max(gt_amp, 1e-3) * 100.0)
                seed_res[cid] = {
                    "regime": "Limit Cycle",
                    "gt_amp": gt_amp,
                    "pred_amp": pred_amp,
                    "amp_err_pct": err
                }
        all_results[str(s)] = seed_res

    # Summary table
    print("\nSUMMARY ACROSS ALL 12 CASES (MEAN ± STD ACROSS 7 SEEDS):")
    print(f"{'Case':5s} | {'Regime':13s} | {'Mean Metric':20s} | {'Std':10s} | {'Median':12s} | {'Min - Max'}")
    print("-" * 80)
    for c in SWEEP_CASES:
        cid = c['id']
        is_ss = c['is_steady_state']
        if is_ss:
            vals = [all_results[str(s)][cid]["drift_from_ystar_mM"] for s in SEEDS]
            unit = "mM (drift)"
        else:
            vals = [all_results[str(s)][cid]["amp_err_pct"] for s in SEEDS]
            unit = "% (err)"

        mean_v = np.mean(vals)
        std_v = np.std(vals)
        med_v = np.median(vals)
        min_v = np.min(vals)
        max_v = np.max(vals)
        print(f"{cid:5s} | {('Steady State' if is_ss else 'Limit Cycle'):13s} | {mean_v:8.3f} {unit:11s} | {std_v:8.3f} | {med_v:8.3f}     | ({min_v:.3f} - {max_v:.3f})")

    with open("scratch/all_cases_fixed_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved scratch/all_cases_fixed_summary.json successfully!")

if __name__ == "__main__":
    evaluate_fixed_models()
