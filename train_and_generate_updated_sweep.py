import os
import sys
import time
import json
import shutil
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\4fd50b55-98fd-4b91-92cf-1a428e462f37"
OLD_BRAIN = r"C:\Users\satya\.gemini\antigravity-ide\brain\fac6f262-a7ea-43cd-9cf8-6ed5750f1c06"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP,
    system,
    simulate_trajectory,
    rk4_step,
    DEFAULT_GUESS,
    DT,
    set_seed
)
from train_all_biochemical_models import FiLMWithRates
from run_complete_evaluation_suite import analyze_signal
from run_mega_parameter_sweep import SWEEP_CASES, rollout_any, find_steady_state, SEEDS
from train_all_bifurcation_aware_models import build_bifurcation_dataset, sample_batch

def train_seed_with_regime_penalty(seed, dataset, total_steps=300):
    set_seed(seed)
    print(f"\n========================================================")
    print(f"Training FiLM with Targeted Regime Penalty (Seed {seed})")
    print(f"========================================================")

    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    
    # Warm start from existing checkpoint
    ckpt_path = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{seed}.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"  Warm-started from {os.path.basename(ckpt_path)}")

    # Target points for regime classification:
    # Stable regime: X1..X6, with 3x targeted focus on X2 and X3
    ss_cases = [c for c in SWEEP_CASES if c['is_steady_state']]
    ss_tensors = []
    for c in ss_cases:
        p_t = torch.tensor(c['params'], dtype=torch.float32)
        y_ss = torch.tensor(find_steady_state(c['params']), dtype=torch.float32)
        weight = 3.5 if c['id'] in ['X2', 'X3'] else 1.0
        ss_tensors.append((p_t, y_ss, weight))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = 15

        # 1. Trajectory loss across dual-regime dataset
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

        # 2. Targeted Equilibrium Vanishing Velocity Penalty at y_ss (specifically X2 & X3)
        regime_penalty = 0.0
        for p_t, y_ss, w in ss_tensors:
            s_in = y_ss.unsqueeze(0)
            p_in = p_t.unsqueeze(0)
            c_t0 = s_in[:, 0:1] + s_in[:, 1:2] + s_in[:, 2:3]
            s_next = model(s_in, p_in, target_ctot=c_t0)
            v_ss = (s_next - s_in) / model.dt
            regime_penalty = regime_penalty + w * torch.mean((v_ss / model.state_scales) ** 2)
        regime_penalty = regime_penalty / len(ss_tensors)

        total_loss = loss_traj + 0.05 * regime_penalty

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == total_steps:
            print(f"  Seed {seed:4d} | Step {step:3d}/{total_steps} | TrajLoss: {loss_traj.item():.4e} | RegimePenalty: {regime_penalty.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    out_name = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{seed}.pt")
    torch.save(model.state_dict(), out_name)
    print(f"  Saved updated checkpoint to {os.path.basename(out_name)}")
    return model

def generate_figure():
    print("\n" + "=" * 65)
    print("Generating Updated 12-Regime Publication Figure (Seed 42)...")
    print("=" * 65)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=220)
    plt.subplots_adjust(hspace=0.42, wspace=0.22)

    # 1. Updated Bifurcation-Aware Model (Seed 42)
    m_bif = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_bif.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_bifurcation_aware_s42.pt"), map_location="cpu"))
    m_bif.eval()

    # 2. Original FiLM Pure (Osc-Only)
    m_film = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_film.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt"), map_location="cpu"))
    m_film.eval()

    # 3. Projected Baseline (Simplex projection)
    m_base = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    m_base.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt"), map_location="cpu"))
    m_base.eval()

    for idx, c_info in enumerate(SWEEP_CASES):
        r_i = idx // 3
        c_i = idx % 3
        ax = axes[r_i, c_i]

        p_arr = np.array(c_info["params"], dtype=np.float64)
        is_ss = c_info.get("is_steady_state", False)

        if is_ss:
            ic = find_steady_state(c_info["params"])
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c_info["warmup"] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()

        steps = int(c_info["duration"] / DT)
        n_pts = min(steps, 15000)
        t_arr = np.linspace(0, n_pts * DT, n_pts)

        gt_tr = simulate_trajectory(ic, steps, DT, p_arr)
        pred_bif, _, _ = rollout_any(m_bif, ic, c_info["params"], steps, c_info["prominence"], c_info["min_dist"], mode="film")
        pred_f, _, _ = rollout_any(m_film, ic, c_info["params"], steps, c_info["prominence"], c_info["min_dist"], mode="film")
        pred_proj, _, _ = rollout_any(m_base, ic, c_info["params"], steps, c_info["prominence"], c_info["min_dist"], mode="projected_baseline")
        pred_van, _, _ = rollout_any(m_base, ic, c_info["params"], steps, c_info["prominence"], c_info["min_dist"], mode="baseline")

        ax.plot(t_arr, gt_tr[:n_pts, 2], 'k-', lw=2.4, label="Ground Truth RK4")
        ax.plot(t_arr, pred_bif[:n_pts, 2], color='#b300b3', linestyle='-', lw=2.0, label="FiLM Bifurcation-Aware (Ours)")
        ax.plot(t_arr, pred_f[:n_pts, 2], 'g--', lw=1.3, alpha=0.75, label="FiLM Pure (Original)")
        ax.plot(t_arr, pred_proj[:n_pts, 2], color='#4374b3', linestyle='-.', lw=1.3, alpha=0.85, label="Projected Baseline")
        ax.plot(t_arr, pred_van[:n_pts, 2], color='#f17c8f', linestyle=':', lw=1.1, alpha=0.75, label="Vanilla Baseline")

        reg_badge = "Steady State (Straight Line)" if is_ss else "Sustained Limit Cycle"
        ax.set_title(f"{c_info['id']}: $k_a={c_info['params'][0]}$, $k_{{d2}}={c_info['params'][1]:.0f}$, $k_{{d3}}={c_info['params'][2]}$\n[{reg_badge}]", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(r"Polymer $m_1$ (mM)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10.5, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Comprehensive 12-Regime Generalization Sweep (Seed 42: Deep $k_a > 10$ & Diverse $k_{d2}, k_{d3}$)", fontsize=14, fontweight="bold", y=1.01)

    out_img = os.path.join(CURRENT_DIR, "mega_parameter_sweep_comprehensive.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=220, bbox_inches="tight")
    plt.close()

    # Copy to brain artifact directories
    if os.path.exists(BRAIN_DIR):
        shutil.copyfile(out_img, os.path.join(BRAIN_DIR, "mega_parameter_sweep_comprehensive.png"))
    if os.path.exists(OLD_BRAIN):
        shutil.copyfile(out_img, os.path.join(OLD_BRAIN, "mega_parameter_sweep_comprehensive.png"))

    print(f"Successfully saved updated figure to:\n  - {out_img}\n  - {os.path.join(BRAIN_DIR, 'mega_parameter_sweep_comprehensive.png')}")

def main():
    print("=" * 70)
    print("BIFURCATION REGIME-CLASSIFICATION FINE-TUNING & FIGURE GENERATION")
    print("=" * 70)
    
    dataset = build_bifurcation_dataset()
    
    # Train all 7 seeds
    for s in SEEDS:
        train_seed_with_regime_penalty(s, dataset, total_steps=300)
        
    generate_figure()
    print("\nALL SEEDS UPDATED AND 12-REGIME FIGURE GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
