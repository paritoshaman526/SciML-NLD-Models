import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    DT,
    DEFAULT_GUESS,
    rk4_step,
    simulate_trajectory,
    set_seed
)
from run_mega_parameter_sweep import SWEEP_CASES, rollout_any, find_steady_state
from train_all_bifurcation_aware_models import build_bifurcation_dataset, sample_batch

def evaluate_cases(model, case_ids=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X10', 'X12'], max_steps=15000):
    model.eval()
    results = {}
    for cid in case_ids:
        c = next(item for item in SWEEP_CASES if item['id'] == cid)
        p_arr = np.array(c['params'], dtype=np.float64)
        if c['is_steady_state']:
            ic = find_steady_state(c['params'])
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c['warmup'] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y
        steps = min(int(c['duration'] / DT), max_steps)
        pred, sig, drift = rollout_any(model, ic, c['params'], steps, c['prominence'], c['min_dist'], mode='film')
        ptp = float(np.ptp(pred[:, 2]))
        results[cid] = {
            'ptp': ptp,
            'amp': float(sig['steady_amp']),
            'period': float(sig['period']),
            'drift': float(drift)
        }
    return results

def main():
    print("Testing Regime-Classification Penalty Targeting X2/X3 on Seed 42...")
    seed = 42
    set_seed(seed)

    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    ckpt = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{seed}.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    
    print("\n--- BEFORE REGIME PENALTY (Seed 42) ---")
    before = evaluate_cases(model)
    for cid, r in before.items():
        print(f"  {cid:4s}: PTP = {r['ptp']:7.2f} mM | Amp = {r['amp']:7.2f} mM | Period = {r['period']:.4f}s")

    # Build dual-regime dataset
    dataset = build_bifurcation_dataset()

    # Define X2 and X3 targets specifically
    X2_params = (12.0, 6000.0, 0.15, 1.0)
    X3_params = (13.0, 6000.0, 0.25, 1.0)
    
    y_ss_X2 = torch.tensor(find_steady_state(X2_params), dtype=torch.float32)
    y_ss_X3 = torch.tensor(find_steady_state(X3_params), dtype=torch.float32)
    p_X2 = torch.tensor(X2_params, dtype=torch.float32)
    p_X3 = torch.tensor(X3_params, dtype=torch.float32)

    # Prepare perturbated ICs for X2/X3
    targeted_ss = []
    for (p_t, y_ss) in [(p_X2, y_ss_X2), (p_X3, y_ss_X3)]:
        for ic_idx in range(6):
            ic = y_ss.clone()
            if ic_idx > 0:
                pert = (ic_idx - 2.5) * 40.0
                ic[0] = max(ic[0] + pert, 10.0)
                ic[1] = max(ic[1] - pert, 10.0)
                ic[3] = max(ic[3] * (0.95 if ic_idx % 2 == 0 else 1.05), 0.5)
            targeted_ss.append((p_t, ic, y_ss))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    total_steps = 400

    print(f"\n--- Fine-Tuning 400 steps with Regime-Classification Penalty Targeting X2/X3 ---")
    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = 20

        # Regular batch to maintain limit-cycle dynamics across all regimes
        state_b, params_b, future_b = sample_batch(dataset, K, batch_size=48)
        c_tot0 = state_b[:, 0:1] + state_b[:, 1:2] + state_b[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        traj_loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b, target_ctot=c_tot0)
            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            traj_loss = traj_loss + state_l + 0.05 * deriv_l
        traj_loss = traj_loss / K

        # Specific Regime-Classification Penalty targeting X2/X3
        # In steady-state regime, velocity must vanish, and trajectory must converge to y_ss
        regime_loss = 0.0
        for p_t, ic, y_ss in targeted_ss:
            s_curr = ic.unsqueeze(0)
            p_curr = p_t.unsqueeze(0)
            c_t0 = s_curr[:, 0:1] + s_curr[:, 1:2] + s_curr[:, 2:3]
            y_target = y_ss.unsqueeze(0)
            
            for k in range(K):
                s_prev = s_curr
                s_curr = model(s_curr, p_curr, target_ctot=c_t0)
                # Deviation from fixed point
                dist_l = torch.mean(((s_curr - y_target) * weights) ** 2)
                # Vanishing velocity penalty
                vel = (s_curr - s_prev) / model.dt
                vel_l = torch.mean((vel / (model.state_scales * 10.0)) ** 2)
                regime_loss = regime_loss + dist_l + 0.1 * vel_l
        regime_loss = regime_loss / (len(targeted_ss) * K)

        total_loss = traj_loss + 2.0 * regime_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if step % 100 == 0 or step == total_steps:
            print(f"Step {step:3d}/{total_steps} | TrajLoss: {traj_loss.item():.4e} | RegimeLoss: {regime_loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    print("\n--- AFTER REGIME PENALTY (Seed 42) ---")
    after = evaluate_cases(model)
    for cid, r in after.items():
        print(f"  {cid:4s}: PTP = {r['ptp']:7.2f} mM | Amp = {r['amp']:7.2f} mM | Period = {r['period']:.4f}s")

if __name__ == "__main__":
    main()
