import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import (
    system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed, UnconstrainedBaselineMLP
)
from train_and_evaluate_lorenz_style_sciml import (
    find_physical_steady_state, SWEEP_CASES, PHYSICAL_CTOT
)
from run_complete_evaluation_suite import analyze_signal
from scratch.test_x7_fix import generate_extended_training_data, sample_batch

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

def train_baseline_seed(seed, dataset, total_steps=400):
    set_seed(seed)
    model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.5, total_steps], [2, 6, 8]))
        s0_b, p_b, future_b, is_ss_b, ystar_b = sample_batch(dataset, K=K, batch_size=64)

        curr = s0_b
        traj_loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, p_b)
            target = future_b[:, k]
            state_l = torch.mean(((curr - target) * weights) ** 2)
            traj_loss = traj_loss + state_l

        traj_loss = traj_loss / K
        optimizer.zero_grad()
        traj_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

    return model

def evaluate_baseline(model):
    model.eval()
    metrics = {}
    for c_info in SWEEP_CASES:
        cid = c_info['id']
        p_arr = np.array(c_info['params'], dtype=np.float64)
        p_t = torch.tensor(c_info['params'], dtype=torch.float32).unsqueeze(0)
        is_ss = c_info['is_steady_state']

        if is_ss:
            y_star = find_physical_steady_state(c_info['params'], c_tot=PHYSICAL_CTOT)
            delta_m1 = 0.15 * y_star[2]
            y_pert = y_star.copy()
            y_pert[2] += delta_m1
            y_pert[0] -= delta_m1
            y_pert[3] *= 1.20

            curr = torch.tensor(y_pert, dtype=torch.float32).unsqueeze(0)
            steps = 4000
            traj = [y_pert.copy()]
            with torch.no_grad():
                for _ in range(steps):
                    curr = model(curr, p_t)
                    traj.append(curr.squeeze(0).numpy())
            traj = np.array(traj)
            final_drift = float(abs(traj[-1, 2] - y_star[2]))
            pct_recovered = float(max(0.0, 1.0 - final_drift / delta_m1) * 100.0)
            metrics[cid] = {
                "regime": "Stable Focus",
                "final_m1": float(traj[-1, 2]),
                "drift_from_ystar_mM": final_drift,
                "pct_recovered": pct_recovered
            }
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c_info.get('warmup', 5.0) / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()

            steps = 15000 if cid == "X7" else 6000
            gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
            curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)

            traj = [ic.copy()]
            with torch.no_grad():
                for _ in range(steps):
                    curr = model(curr, p_t)
                    traj.append(curr.squeeze(0).numpy())
            traj = np.array(traj)

            sig_gt = analyze_signal(gt_traj[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])
            sig_pred = analyze_signal(traj[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])
            gt_amp = float(sig_gt['steady_amp'])
            pred_amp = float(sig_pred['steady_amp'])
            err = float(abs(pred_amp - gt_amp) / max(gt_amp, 1e-3) * 100.0)
            metrics[cid] = {
                "regime": "Limit Cycle",
                "gt_amp": gt_amp,
                "pred_amp": pred_amp,
                "amp_err_pct": err
            }
    return metrics

def main():
    print("=" * 80)
    print("TRAINING UNCONSTRAINED BASELINE MLP ACROSS 7 CANONICAL SEEDS")
    print("=" * 80)
    dataset = generate_extended_training_data(horizon=2.5)
    all_res = {}
    for s in SEEDS:
        print(f"  Training Baseline Seed {s}...", end=" ", flush=True)
        m = train_baseline_seed(s, dataset)
        res = evaluate_baseline(m)
        all_res[str(s)] = res
        print(f"Done! X7 err: {res['X7']['amp_err_pct']:.1f}%, X4 drift: {res['X4']['drift_from_ystar_mM']:.1f} mM", flush=True)

    with open("scratch/baseline_unconstrained_7seed_results.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print("\nSaved scratch/baseline_unconstrained_7seed_results.json successfully!")

if __name__ == "__main__":
    main()
