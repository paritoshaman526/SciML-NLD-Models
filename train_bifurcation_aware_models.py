"""
======================================================================================
BIFURCATION-AWARE TRAINING SUITE: DUAL-REGIME GROUND RK4 TRAINING
======================================================================================
Trains physics-constrained and baseline models across both sides of the Hopf bifurcation:
  1. Oscillatory Limit Cycles (ka in [2.0, 8.0]) -> Periodic trajectories
  2. Steady-State Regimes (ka > 10.0) -> Fixed-point straight lines & decaying trajectories

This eliminates extrapolation hallucination where models mistakenly oscillate at ka > 10.
======================================================================================
"""

import os
import sys
import time
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP,
    system,
    rk4_step,
    simulate_trajectory,
    DEFAULT_GUESS,
    DT,
    set_seed
)

SEEDS = [42, 101, 2024, 777, 999, 1234, 5555]

def find_steady_state(params):
    p_arr = np.asarray(params, dtype=np.float64)
    def res(y):
        return system(y, p_arr)
    y_guess = np.array([5800.0, min(p_arr[1], 5500.0), 400.0, 6.5], dtype=np.float64)
    opt = scipy.optimize.least_squares(res, y_guess, bounds=([0, 0, 0, 1e-5], [10002, 10002, 10002, 100]), xtol=1e-10, ftol=1e-10)
    return opt.x

OSC_PARAMS = [
    (2.0, 3000.0, 0.6, 1.0),
    (2.0, 4500.0, 1.0, 1.0),
    (4.0, 3000.0, 1.0, 1.0),
    (4.0, 4500.0, 0.6, 1.0),
    (6.0, 3000.0, 0.6, 1.0),
    (6.0, 4208.0, 1.0, 1.0),
    (6.0, 4500.0, 1.0, 1.0),
    (8.0, 4500.0, 0.6, 1.0),
]

SS_PARAMS = [
    (10.5, 4600.0, 0.60, 1.0),
    (11.0, 6000.0, 0.15, 1.0),
    (12.0, 4500.0, 0.50, 1.0),
    (14.0, 5000.0, 0.30, 1.0),
]

def build_bifurcation_dataset(traj_duration=1.2, dt=DT, num_osc_ics=6, num_ss_ics=6):
    steps = int(traj_duration / dt)
    dataset = []

    # 1. Limit cycle oscillatory regimes
    for p in OSC_PARAMS:
        p_arr = np.array(p, dtype=np.float64)
        y_warm = DEFAULT_GUESS.copy()
        for _ in range(int(12.0 / dt)):
            y_warm = rk4_step(y_warm, dt, p_arr)
        for ic_idx in range(num_osc_ics):
            ic = y_warm.copy()
            pert = (ic_idx - (num_osc_ics - 1) / 2.0) * 80.0
            ic[0] = max(ic[0] + pert, 10.0)
            ic[1] = max(ic[1] - pert, 10.0)
            ic[3] = max(ic[3] * (0.93 if ic_idx % 2 == 0 else 1.07), 0.5)
            traj = simulate_trajectory(ic, steps, dt, p_arr)
            if not np.isnan(traj).any():
                dataset.append({
                    'traj': torch.tensor(traj, dtype=torch.float32),
                    'params': torch.tensor(p, dtype=torch.float32)
                })

    # 2. Steady state regimes (ka > 10, flat line + decaying perturbations)
    for p in SS_PARAMS:
        p_arr = np.array(p, dtype=np.float64)
        y_ss = find_steady_state(p)
        for ic_idx in range(num_ss_ics):
            ic = y_ss.copy()
            if ic_idx > 0:
                pert = (ic_idx - (num_ss_ics - 1) / 2.0) * 60.0
                ic[0] = max(ic[0] + pert, 10.0)
                ic[1] = max(ic[1] - pert, 10.0)
                ic[3] = max(ic[3] * (0.95 if ic_idx % 2 == 0 else 1.05), 0.5)
            # ic_idx == 0 is exact flat horizontal straight line
            traj = simulate_trajectory(ic, steps, dt, p_arr)
            if not np.isnan(traj).any():
                dataset.append({
                    'traj': torch.tensor(traj, dtype=torch.float32),
                    'params': torch.tensor(p, dtype=torch.float32)
                })

    print(f"Built Bifurcation-Aware Ground RK4 Dataset: {len(dataset)} trajectories ({len(OSC_PARAMS)} osc + {len(SS_PARAMS)} ss)")
    return dataset

def sample_batch(dataset, K, batch_size=64):
    n = len(dataset)
    traj_len = dataset[0]['traj'].shape[0]
    t_indices = np.random.randint(0, traj_len - K, size=batch_size)
    d_indices = np.random.randint(0, n, size=batch_size)

    s0_list, p_list, fut_list = [], [], []
    for bi in range(batch_size):
        di = d_indices[bi]
        t0 = t_indices[bi]
        s0_list.append(dataset[di]['traj'][t0])
        p_list.append(dataset[di]['params'])
        fut_list.append(dataset[di]['traj'][t0 + 1 : t0 + 1 + K])

    return torch.stack(s0_list), torch.stack(p_list), torch.stack(fut_list)

def train_seed(seed, dataset, total_steps=1000):
    set_seed(seed)
    print(f"\n--- Training Bifurcation-Aware FiLM (Seed {seed}) ---")
    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    
    # Warm start from existing checkpoint if available
    ckpt_path = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
    if not os.path.exists(ckpt_path) and seed == 42:
        ckpt_path = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Warm-started from {ckpt_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [5, 15, 25]))
        state_b, params_b, future_b = sample_batch(dataset, K, batch_size=64)
        c_tot0 = state_b[:, 0:1] + state_b[:, 1:2] + state_b[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b, target_ctot=c_tot0)
            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            loss = loss + state_l + 0.05 * deriv_l
        loss = loss / K

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == total_steps:
            print(f"Seed {seed:4d} | Step {step:4d}/{total_steps} | Loss: {loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    out_name = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{seed}.pt")
    torch.save(model.state_dict(), out_name)
    print(f"Saved: {out_name}")
    return model

def main():
    print("=" * 80)
    print("TRAINING BIFURCATION-AWARE MODELS ON GROUND RK4 (ALL 7 SEEDS)")
    print("=" * 80)
    dataset = build_bifurcation_dataset()
    for s in SEEDS:
        train_seed(s, dataset, total_steps=1000)
    print("\nAll 7 seeds successfully trained!")

if __name__ == "__main__":
    main()
