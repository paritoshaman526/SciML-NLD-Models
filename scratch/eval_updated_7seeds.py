import os
import sys
import json
import numpy as np
import torch

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import ProposedFiLMPhysicsConstrainedMLP, DT, DEFAULT_GUESS, rk4_step, simulate_trajectory
from run_mega_parameter_sweep import SWEEP_CASES, rollout_any, find_steady_state, SEEDS
from run_complete_evaluation_suite import analyze_signal

def evaluate_all_7seeds():
    updated_results = {}
    print(f"{'Case':4s} | {'Regime':7s} | {'FiLM Bifurcation-Aware (Updated 7 Seeds)':42s}")
    print("-" * 65)

    for c_info in SWEEP_CASES:
        cid = c_info['id']
        p_arr = np.array(c_info['params'], dtype=np.float64)
        is_ss = c_info.get('is_steady_state', False)
        
        if is_ss:
            ic = find_steady_state(c_info['params'])
            gt_t = 0.0
            gt_amp = 0.0
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c_info['warmup'] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()
            steps_gt = int(c_info['duration'] / DT)
            gt_tr = simulate_trajectory(ic, steps_gt, DT, p_arr)
            gt_m = analyze_signal(gt_tr[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])
            gt_t = gt_m['period']
            gt_amp = gt_m['steady_amp']

        p_errs, ratios, drifts = [], [], []
        steps = int(c_info['duration'] / DT)

        for s in SEEDS:
            ckpt = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{s}.pt")
            m = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()

            pred, sig, drift = rollout_any(m, ic, c_info['params'], steps, c_info['prominence'], c_info['min_dist'], mode='film')

            if is_ss:
                amp_hallucinated = float(np.ptp(pred[:, 2]))
                p_err = 0.0 if amp_hallucinated < 5.0 else 100.0
                ratio = amp_hallucinated
            else:
                if gt_t > 0 and sig['period'] > 0:
                    p_err = abs(sig['period'] - gt_t) / gt_t * 100.0
                elif gt_t == 0 and sig['period'] == 0:
                    p_err = 0.0
                else:
                    p_err = 100.0
                ratio = sig['steady_amp'] / max(gt_amp, 1e-6)

            p_errs.append(p_err)
            ratios.append(ratio)
            drifts.append(drift)

        updated_results[cid] = {
            'is_ss': is_ss,
            'p_err_mean': float(np.mean(p_errs)),
            'p_err_std': float(np.std(p_errs)),
            'p_err_median': float(np.median(p_errs)),
            'p_err_min': float(np.min(p_errs)),
            'p_err_max': float(np.max(p_errs)),
            'ratio_mean': float(np.mean(ratios)),
            'ratio_std': float(np.std(ratios)),
            'ratio_median': float(np.median(ratios)),
            'ratio_min': float(np.min(ratios)),
            'ratio_max': float(np.max(ratios)),
            'drift_mean': float(np.mean(drifts)),
            'drift_std': float(np.std(drifts)),
        }

        if is_ss:
            print(f"{cid:4s} | Steady  | Amp: {np.mean(ratios):6.2f} +/- {np.std(ratios):5.2f} mM [{np.median(ratios):.2f}, {np.min(ratios):.2f}-{np.max(ratios):.2f}]")
        else:
            print(f"{cid:4s} | LimitC  | P_err: {np.mean(p_errs):5.2f} +/- {np.std(p_errs):5.2f}% | A_ratio: {np.mean(ratios):.3f} +/- {np.std(ratios):.3f}")

    with open('updated_bifurcation_multiseed_results.json', 'w') as f:
        json.dump(updated_results, f, indent=2)

if __name__ == '__main__':
    evaluate_all_7seeds()
