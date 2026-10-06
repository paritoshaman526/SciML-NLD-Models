"""
==============================================================================
UNIFIED TRAINING SUITE: ALL 4 BIOCHEMICAL SCIML MODELS
==============================================================================
This script trains and evaluates all 4 models compared in the publication benchmark:
  1. Model 1: Baseline Generic MLP (Unconstrained Black-Box Neural ODE)
  2. Model 2: Divergence-Regularized MLP (Ablation: Phase-Space Volume Matching)
  3. Model 3: Proposed FiLM Physics-Constrained SciML (Exact Conservation + Positivity)
  4. Model 4: FiLM + Divergence Combined (Champion Model: Invariants + Volume Matching)

Usage:
  python train_all_biochemical_models.py --model all
  python train_all_biochemical_models.py --model baseline
  python train_all_biochemical_models.py --model div
  python train_all_biochemical_models.py --model film
  python train_all_biochemical_models.py --model film_div
==============================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP,
    build_dataset_with_ic_splits,
    simulate_trajectory,
    rk4_step,
    DEFAULT_GUESS,
    DT
)

# 8 Parameter configurations spanning the oscillatory regime in 3D parameter space
TRAINING_PARAMS = [
    (2.0, 3000.0, 0.6, 1.0),
    (2.0, 4500.0, 1.0, 1.0),
    (4.0, 3000.0, 1.0, 1.0),
    (4.0, 4500.0, 0.6, 1.0),
    (6.0, 3000.0, 0.6, 1.0),
    (6.0, 4208.0, 1.0, 1.0),  # Case A Nominal Slide
    (6.0, 4500.0, 1.0, 1.0),
    (8.0, 4500.0, 0.6, 1.0),
]

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_true_divergence(state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Computes the analytical trace of the Jacobian (phase-space divergence)."""
    ka = params[:, 0]
    kd2 = params[:, 1]
    kd3 = params[:, 2]
    kp = params[:, 3]

    m1 = state[:, 2]
    m0_safe = torch.clamp(state[:, 3], min=1e-5)
    denom = torch.clamp(m1 - 2.0 * m0_safe, min=1e-3)

    d_dd_d = -ka
    d_da1_da1 = -2.0 * kp * m0_safe
    d_dm1_dm1 = torch.zeros_like(ka)

    dg_dm0 = 9.2 * kd2 * (m0_safe ** 2) * (3.0 * m1 - 2.0 * m0_safe) / (denom ** 3)
    d_dm0_dm0 = -6.0 * kd3 - 2.0 * kp * m0_safe - dg_dm0

    return d_dd_d + d_da1_da1 + d_dm1_dm1 + d_dm0_dm0

def compute_unconstrained_divergence(model: UnconstrainedBaselineMLP, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Computes div(f_theta) for the unconstrained baseline via PyTorch autograd."""
    s_grad = states.clone().detach().requires_grad_(True)
    s_norm = s_grad / model.state_scales
    p_norm = params / model.param_scales
    x = torch.cat([s_norm, p_norm], dim=1)
    rates = model.net(x) * model.state_scales
    div = torch.zeros(states.shape[0], device=states.device)
    for i in range(4):
        g = torch.zeros_like(rates)
        g[:, i] = 1.0
        grad_i = torch.autograd.grad(rates, s_grad, grad_outputs=g, create_graph=True, retain_graph=True)[0]
        div = div + grad_i[:, i]
    return div

class FiLMWithRates(ProposedFiLMPhysicsConstrainedMLP):
    """Extends ProposedFiLM to expose instantaneous reaction flux rates for divergence penalty."""
    def get_rates(self, state, params):
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)
        h = self.block1(s_norm, p_embed)
        h = self.block2(h, p_embed)
        h = self.block3(h, p_embed)
        out = self.head(h)

        J1 = torch.nn.functional.softplus(out[:, 0:1]) * self.rate_scales[0]
        J2 = torch.nn.functional.softplus(out[:, 1:2]) * self.rate_scales[1]
        J3 = torch.nn.functional.softplus(out[:, 2:3]) * self.rate_scales[2]
        rm0 = out[:, 3:4] * self.rate_scales[3]

        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2
        return torch.cat([rd, ra1, rm1, rm0], dim=1)

def compute_film_divergence(model: FiLMWithRates, states: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Computes div(f_theta) for FiLM model via PyTorch autograd."""
    s_grad = states.clone().detach().requires_grad_(True)
    rates = model.get_rates(s_grad, params)
    div = torch.zeros(states.shape[0], device=states.device)
    for i in range(4):
        g = torch.zeros_like(rates)
        g[:, i] = 1.0
        grad_i = torch.autograd.grad(rates, s_grad, grad_outputs=g, create_graph=True, retain_graph=True)[0]
        div = div + grad_i[:, i]
    return div

def sample_batch(ds, K, batch_size=64):
    states, params_list, future = [], [], []
    indices = np.random.randint(0, len(ds), size=batch_size)
    for idx_sample in indices:
        item = ds[idx_sample]
        tr = item["traj"]
        p = item["params"]
        idx = np.random.randint(0, len(tr) - K - 1)
        states.append(tr[idx])
        params_list.append(p)
        future.append(tr[idx + 1 : idx + K + 1])
    return torch.stack(states), torch.stack(params_list), torch.stack(future)

def evaluate_on_case_a(model, name="Model", is_constrained=True, n_rollout=20000):
    model.eval()
    p = (6.0, 4208.0, 1.0, 1.0)
    p_arr = np.array(p, dtype=np.float64)
    y = DEFAULT_GUESS.copy()
    for _ in range(int(20.0 / DT)):
        y = rk4_step(y, DT, p_arr)

    s = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
    c0 = s[:, 0:1] + s[:, 1:2] + s[:, 2:3] if is_constrained else None

    traj = [y.copy()]
    with torch.no_grad():
        for _ in range(n_rollout):
            if is_constrained:
                s = model(s, p_t, target_ctot=c0)
            else:
                s = model(s, p_t)
            traj.append(s.squeeze(0).cpu().numpy())
    traj = np.array(traj)

    peaks, _ = find_peaks(traj[:, 2], distance=100)
    troughs, _ = find_peaks(-traj[:, 2], distance=100)
    n_pairs = min(len(peaks), len(troughs))
    ptp_cycles = [traj[peaks[i], 2] - traj[troughs[i], 2] for i in range(n_pairs)] if n_pairs > 0 else [0.0]
    cycle_ret = (ptp_cycles[-1] / max(ptp_cycles[0], 1e-6) * 100.0) if len(ptp_cycles) > 1 else 0.0

    quarters = np.array_split(traj[:, 2], 4)
    amps = [np.ptp(q) for q in quarters]
    q_ret = amps[-1] / max(amps[0], 1e-6) * 100.0

    c_tot = traj[:, 0] + traj[:, 1] + traj[:, 2]
    drift = np.ptp(c_tot)

    print(f"[{name}] Case A Evaluation:")
    print(f"  Cycle Retention: {cycle_ret:.1f}% (PTP: {ptp_cycles[0]:.1f} -> {ptp_cycles[-1]:.1f})")
    print(f"  Quarter Retention (Q4/Q1): {q_ret:.1f}% (Q1={amps[0]:.1f} -> Q4={amps[-1]:.1f})")
    print(f"  Stoichiometric Mass Drift: {drift:.3f} mM (Initial c_tot = 10,002 mM)")
    return {"cycle_ret": cycle_ret, "q_ret": q_ret, "drift": drift}

# ==============================================================================
# 1. TRAIN UNCONSTRAINED BASELINE MLP
# ==============================================================================
def train_baseline(train_ds, total_steps=3000, lr=2e-3, k_max=25, seed=42, out_name=None):
    if out_name is None:
        out_name = f"baseline_unconstrained_mlp_s{seed}.pt"
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL 1: Unconstrained Baseline MLP (Seed={seed}, k_max={k_max})")
    print("=" * 80)
    set_seed(seed)
    model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [5, 15, k_max]))
        state_b, params_b, future_b = sample_batch(train_ds, K)
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b)
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
            print(f"Step {step:4d}/{total_steps} | K={K:2d} | Loss: {loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    out_path = os.path.join(CURRENT_DIR, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    evaluate_on_case_a(model, name=f"Baseline MLP (s{seed})", is_constrained=False)
    return model

# ==============================================================================
# 2. TRAIN DIVERGENCE-REGULARIZED BASELINE MLP (ABLATION)
# ==============================================================================
def train_divergence_baseline(train_ds, total_steps=2000, lr=1e-3, alpha_div=1e-4, seed=42, base_ckpt=None, out_name=None):
    if out_name is None:
        out_name = f"divergence_regularized_mlp_s{seed}.pt"
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL 2: Divergence-Regularized MLP (Seed={seed}, alpha_div={alpha_div:.1e})")
    print("=" * 80)
    set_seed(seed)
    model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    
    # Warm-start strictly from same-seed baseline checkpoint
    if base_ckpt is None:
        same_seed_base = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
        default_base = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
        if os.path.exists(same_seed_base):
            base_ckpt = same_seed_base
        elif seed == 42 and os.path.exists(default_base):
            base_ckpt = default_base
        else:
            raise FileNotFoundError(
                f"Cannot warm-start Divergence model for seed {seed}: "
                f"Expected same-seed baseline checkpoint at '{same_seed_base}' not found! "
                f"Please run train_baseline(..., seed={seed}) first."
            )

    model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
    print(f"Warm-started from same-seed baseline: {base_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [10, 20, 30]))
        state_b, params_b, future_b = sample_batch(train_ds, K)
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b)
            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            loss = loss + state_l + 0.05 * deriv_l
        loss = loss / K

        # Subsample 16 states for divergence regularization
        sub_s = state_b[:16]
        sub_p = params_b[:16]
        true_div = compute_true_divergence(sub_s, sub_p)
        pred_div = compute_unconstrained_divergence(model, sub_s, sub_p)
        div_loss = torch.mean(((pred_div - true_div) / 100.0) ** 2)

        total_loss = loss + alpha_div * div_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % 400 == 0 or step == total_steps:
            print(f"Step {step:4d}/{total_steps} | K={K:2d} | Loss: {loss.item():.4e} | DivLoss: {div_loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    out_path = os.path.join(CURRENT_DIR, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    evaluate_on_case_a(model, name=f"Divergence MLP (s{seed})", is_constrained=False)
    return model

# ==============================================================================
# 3. TRAIN PROPOSED FiLM PHYSICS-CONSTRAINED SCIML
# ==============================================================================
def train_proposed_film(train_ds, total_steps=3000, lr=2e-3, seed=42, m0_floor=0.5, out_name=None):
    if out_name is None:
        out_name = f"best_proposed_biochemical_mlp_s{seed}.pt"
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL 3: Proposed FiLM Physics-Constrained SciML (Seed={seed}, Floor={m0_floor})")
    print("=" * 80)
    set_seed(seed)
    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=m0_floor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [5, 15, 25]))
        state_b, params_b, future_b = sample_batch(train_ds, K)
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
            print(f"Step {step:4d}/{total_steps} | K={K:2d} | Loss: {loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    out_path = os.path.join(CURRENT_DIR, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    evaluate_on_case_a(model, name=f"Proposed FiLM (s{seed})", is_constrained=True)
    return model

# ==============================================================================
# 4. TRAIN FiLM + DIVERGENCE COMBINED (CHAMPION MODEL)
# ==============================================================================
def train_film_divergence_combined(train_ds, total_steps=600, lr=1.5e-4, alpha_div=1e-4, seed=42, m0_floor=0.5, base_ckpt=None, out_name=None):
    if out_name is None:
        out_name = f"film_divergence_regularized_mlp_s{seed}.pt"
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL 4: FiLM + Divergence Combined (Seed={seed}, Floor={m0_floor}, alpha_div={alpha_div:.1e})")
    print("=" * 80)
    set_seed(seed)
    model = FiLMWithRates(hidden_dim=128, dt=DT, m0_floor=m0_floor)

    # Initialize strictly from same-seed proposed FiLM checkpoint
    if base_ckpt is None:
        same_seed_film = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
        default_film = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
        if os.path.exists(same_seed_film):
            base_ckpt = same_seed_film
        elif seed == 42 and os.path.exists(default_film):
            base_ckpt = default_film
        else:
            raise FileNotFoundError(
                f"Cannot warm-start FiLM+Div model for seed {seed}: "
                f"Expected same-seed FiLM checkpoint at '{same_seed_film}' not found! "
                f"Please run train_proposed_film(..., seed={seed}) first."
            )

    model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
    print(f"Initialized from same-seed FiLM base: {base_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [10, 20, 30]))
        state_b, params_b, future_b = sample_batch(train_ds, K)
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

        # Subsample 16 states for divergence penalty
        sub_s = state_b[:16]
        sub_p = params_b[:16]
        true_div = compute_true_divergence(sub_s, sub_p)
        model_div = compute_film_divergence(model, sub_s, sub_p)
        div_loss = torch.mean(((model_div - true_div) / 100.0) ** 2)

        total_loss = loss + alpha_div * div_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == total_steps:
            print(f"Step {step:4d}/{total_steps} | K={K:2d} | Loss: {loss.item():.4e} | DivLoss: {div_loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    out_path = os.path.join(CURRENT_DIR, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    evaluate_on_case_a(model, name=f"FiLM + Divergence (s{seed})", is_constrained=True)
    return model

# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Training Suite for Biochemical SciML Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "baseline", "baseline_improved", "div", "film", "film_div"],
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic initialization and data sampling"
    )
    parser.add_argument(
        "--m0_floor",
        type=float,
        default=0.5,
        help="Positivity protection floor for pacemaker m0 (default: 0.5)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"PREPARING BIOCHEMICAL DYNAMICAL DATASETS (Seed={args.seed})...")
    print("=" * 80)
    set_seed(args.seed)
    train_ds, val_ds = build_dataset_with_ic_splits(
        TRAINING_PARAMS, traj_duration=1.2, dt=DT, num_train_ics=4, num_val_ics=2
    )

    s = args.seed
    fl = args.m0_floor
    if args.model in ["all", "baseline"]:
        train_baseline(train_ds, total_steps=3000, lr=2e-3, k_max=25, seed=s)

    if args.model in ["baseline_improved"]:
        train_baseline(train_ds, total_steps=3000, lr=1e-3, k_max=50, seed=s, out_name=f"improved_baseline_unconstrained_mlp_s{s}.pt")

    if args.model in ["all", "div"]:
        train_divergence_baseline(train_ds, total_steps=2000, lr=1e-3, alpha_div=1e-4, seed=s)

    if args.model in ["all", "film"]:
        train_proposed_film(train_ds, total_steps=3000, lr=2e-3, seed=s, m0_floor=fl)

    if args.model in ["all", "film_div"]:
        train_film_divergence_combined(train_ds, total_steps=600, lr=1.5e-4, alpha_div=1e-4, seed=s, m0_floor=fl)

    print("\n" + "=" * 80)
    print(f"ALL REQUESTED MODELS FOR SEED {args.seed} TRAINED AND SAVED SUCCESSFULLY.")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("ALL REQUESTED MODELS TRAINED AND SAVED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
