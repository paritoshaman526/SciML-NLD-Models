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
    set_seed
)
from run_mega_parameter_sweep import SWEEP_CASES, rollout_any, find_steady_state
from train_all_bifurcation_aware_models import build_bifurcation_dataset, sample_batch

def evaluate_key_cases(model):
    model.eval()
    key_cases = ['X1', 'X2', 'X3', 'X4', 'X7', 'X10', 'X12']
    res = {}
    for cid in key_cases:
        c = next(item for item in SWEEP_CASES if item['id'] == cid)
        p_arr = np.array(c['params'], dtype=np.float64)
        if c['is_steady_state']:
            ic = find_steady_state(c['params'])
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c['warmup'] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y
        steps = min(int(c['duration'] / DT), 10000)
        pred, sig, drift = rollout_any(model, ic, c['params'], steps, c['prominence'], c['min_dist'], mode='film')
        ptp = float(np.ptp(pred[:, 2]))
        res[cid] = {'ptp': ptp, 'amp': float(sig['steady_amp']), 'period': float(sig['period'])}
        print(f"  {cid:4s}: PTP={ptp:7.1f} mM | Amp={sig['steady_amp']:7.1f} mM | Period={sig['period']:.4f}s", flush=True)
    return res

def run_balanced_experiment():
    print("Testing Balanced Regime Classification Penalty...")
    seed = 42
    set_seed(seed)

    # Reload clean checkpoint
    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    ckpt = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{seed}.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    
    print("\nInitial State (Seed 42):")
    evaluate_key_cases(model)

    dataset = build_bifurcation_dataset()
    
    # Target points for regime classification:
    # Stable regime: X1, X2, X3, X4, X5, X6 (specifically X2, X3)
    # Oscillatory regime: X7, X8, X9, X10, X11, X12
    ss_cases = [c for c in SWEEP_CASES if c['is_steady_state']]
    ss_tensors = []
    for c in ss_cases:
        p_t = torch.tensor(c['params'], dtype=torch.float32)
        y_ss = torch.tensor(find_steady_state(c['params']), dtype=torch.float32)
        weight = 3.0 if c['id'] in ['X2', 'X3'] else 1.0  # Extra focus on X2/X3
        ss_tensors.append((p_t, y_ss, weight))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)

    print("\nTraining with Targeted Equilibrium Dissipation Penalty (300 steps)...")
    t0 = time.time()
    for step in range(1, 301):
        model.train()
        K = 15

        # 1. Standard batch (majority of loss to protect limit cycles)
        state_b, params_b, future_b = sample_batch(dataset, K, batch_size=64)
        c_tot0 = state_b[:, 0:1] + state_b[:, 1:2] + state_b[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        loss_traj = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b, target_ctot=c_tot0)
            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            loss_traj = loss_traj + state_l + 0.05 * deriv_l
        loss_traj = loss_traj / K

        # 2. Targeted Equilibrium Vanishing Velocity Penalty at y_ss
        # Directly penalize f_theta(y_ss) != 0 for steady states (especially X2 and X3)
        regime_penalty = 0.0
        for p_t, y_ss, w in ss_tensors:
            s_in = y_ss.unsqueeze(0)
            p_in = p_t.unsqueeze(0)
            c_t0 = s_in[:, 0:1] + s_in[:, 1:2] + s_in[:, 2:3]
            s_next = model(s_in, p_in, target_ctot=c_t0)
            # Velocity at fixed point should be strictly zero
            v_ss = (s_next - s_in) / model.dt
            regime_penalty = regime_penalty + w * torch.mean((v_ss / model.state_scales) ** 2)
        regime_penalty = regime_penalty / len(ss_tensors)

        total_loss = loss_traj + 0.05 * regime_penalty

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step:3d}/300 | TrajLoss: {loss_traj.item():.4e} | RegimePenalty: {regime_penalty.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    print("\nResult After Targeted Equilibrium Penalty:")
    evaluate_key_cases(model)

if __name__ == "__main__":
    run_balanced_experiment()
