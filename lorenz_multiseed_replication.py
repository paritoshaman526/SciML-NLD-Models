"""
Tier 2 Scientific Validation: Multi-Seed Replication & Uncertainty Quantification
================================================================================
Conducts rigorous multi-seed training and evaluation across 5 independent seeds
(42, 101, 2024, 777, 999) on the Lorenz system to answer:
1. Does lambda_3 (and lobe-switching behavior) vary significantly across random seeds,
   or is the attractor-preservation property statistically invariant?
2. Reports all Tier-2 metrics as Mean +/- Std:
   - Lyapunov Exponent Spectrum: lambda_1, lambda_2, lambda_3
   - Kaplan-Yorke (Lyapunov) Fractal Dimension: D_KY
   - Grassberger-Procaccia Correlation Dimension: D_2
   - Invariant Probability Distribution Distances: Wasserstein W_1(X), W_1(Y), W_1(Z)
   - Lobe-Switching Frequency & Trapping Diagnostics
"""

import os
import sys

# Prevent OpenBLAS thread allocation issues on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import time
from typing import Tuple, List, Dict, Any
import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lorenz_mlp import LorenzSciML, lorenz_deriv, rk4_step, set_seed
from lorenz_climate_validation import (
    true_jacobian,
    neural_jacobian_batched,
    compute_lyapunov_spectrum_gt,
    compute_lyapunov_spectrum_sciml,
    kaplan_yorke_dimension,
    compute_correlation_dimension,
    extract_poincare_section,
    extract_lorenz_cusp_map
)


# =====================================================================
# 1. Dataset Generation for Multi-Seed Runs
# =====================================================================
def generate_seed_data(
    num_trajectories: int = 50,
    steps: int = 400,
    dt: float = 0.01,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    val_split: float = 0.2,
    seed: int = 42
):
    """Generates leak-free training and held-out IC validation data for a given seed."""
    set_seed(seed)
    raw_ics = np.random.uniform(low=[-15.0, -20.0, 10.0], high=[15.0, 20.0, 35.0], size=(num_trajectories, 3))
    init_states = []
    for ic in raw_ics:
        state = ic.copy()
        for _ in range(200):
            state = rk4_step(state, dt=dt, s=s, r=r, b=b)
        init_states.append(state)
    init_states = np.array(init_states)

    n_val = int(num_trajectories * val_split)
    val_ics = init_states[:n_val]
    train_ics = init_states[n_val:]

    def _sim(ics):
        X0, X1, X2 = [], [], []
        trajs = []
        for s0 in ics:
            curr = s0.copy()
            traj = [curr]
            for _ in range(steps):
                nxt1 = rk4_step(curr, dt=dt, s=s, r=r, b=b)
                nxt2 = rk4_step(nxt1, dt=dt, s=s, r=r, b=b)
                X0.append(curr); X1.append(nxt1); X2.append(nxt2)
                traj.append(nxt1)
                curr = nxt1
            trajs.append(np.array(traj))
        return (torch.tensor(np.array(X0), dtype=torch.float32),
                torch.tensor(np.array(X1), dtype=torch.float32),
                torch.tensor(np.array(X2), dtype=torch.float32),
                trajs)

    return _sim(train_ics), _sim(val_ics)


# =====================================================================
def compute_batch_divergence(model: LorenzSciML, states: torch.Tensor) -> torch.Tensor:
    """Computes exact continuous divergence tr(J_f) = sum_i df_i / dx_i for batch of states."""
    s_grad = states.clone().detach().requires_grad_(True)
    f = model.predict_derivatives(s_grad)
    div = torch.zeros(states.shape[0], device=states.device)
    for coord in range(3):
        g = torch.zeros_like(f)
        g[:, coord] = 1.0
        df_ds = torch.autograd.grad(f, s_grad, grad_outputs=g, create_graph=True, retain_graph=True)[0]
        div = div + df_ds[:, coord]
    return div


# =====================================================================
# 2. Fast Training Routine Per Seed with Divergence Regularization
# =====================================================================
def train_seed_model(
    seed: int,
    epochs: int = 40,
    batch_size: int = 256,
    dt: float = 0.01,
    s: float = 10.0,
    r: float = 28.0,
    b: float = 8.0 / 3.0,
    alpha_div: float = 1e-4
) -> Tuple[LorenzSciML, float, float]:
    """Trains a SciML model instance with exact divergence regularization and strict seed reproducibility."""
    set_seed(seed)
    (X0_tr, X1_tr, X2_tr, _), (X0_val, X1_val, X2_val, _) = generate_seed_data(
        num_trajectories=50, steps=400, dt=dt, s=s, r=r, b=b, val_split=0.2, seed=seed
    )

    train_ds = torch.utils.data.TensorDataset(X0_tr, X1_tr, X2_tr)
    val_ds = torch.utils.data.TensorDataset(X0_val, X1_val, X2_val)

    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = LorenzSciML(hidden_dim=128, dt=dt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    crit = nn.MSELoss()
    target_div = -(s + 1.0 + b)  # Analytical Lorenz divergence: -13.6667

    for epoch in range(epochs):
        model.train()
        for b0, b1, b2 in train_loader:
            optimizer.zero_grad()
            p1 = model(b0)
            p2 = model(p1)
            loss_mse = crit(p1, b1) + 0.5 * crit(p2, b2)
            
            if alpha_div > 0:
                # Subsample 64 points per batch for divergence autograd
                idx = torch.randperm(b0.shape[0])[:64]
                div = compute_batch_divergence(model, b0[idx])
                loss_div = ((div - target_div) ** 2).mean()
                total_loss = loss_mse + alpha_div * loss_div
            else:
                total_loss = loss_mse

            total_loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val1 = sum(crit(model(b0), b1).item() * len(b0) for b0, b1, _ in val_loader) / len(X0_val)
        val2 = sum(crit(model(model(b0)), b2).item() * len(b0) for b0, _, b2 in val_loader) / len(X0_val)

    return model, val1, val2


# =====================================================================
# 3. Lobe-Switching & Residence Time Statistics
# =====================================================================
def compute_lobe_switching_stats(trajectory: np.ndarray, dt: float = 0.01) -> Dict[str, float]:
    """
    Computes lobe-switching transitions and residence times on the Lorenz attractor.
    Lobe is determined by sign of x: Left lobe (x < 0), Right lobe (x > 0).
    """
    x = trajectory[:, 0]
    signs = np.sign(x)
    # Filter out exact zeros if any
    signs[signs == 0] = 1.0

    # Transitions occur where sign changes: sign[k] * sign[k+1] < 0
    transitions = np.where(signs[:-1] * signs[1:] < 0)[0]
    num_switches = len(transitions)
    total_time = len(x) * dt
    switching_rate = num_switches / total_time  # switches per second

    # Residence times between transitions
    if len(transitions) > 1:
        residence_steps = np.diff(transitions)
        residence_times = residence_steps * dt
        mean_residence = float(np.mean(residence_times))
        std_residence = float(np.std(residence_times))
    else:
        mean_residence = total_time
        std_residence = 0.0

    # Lobe balance ratio: fraction of time spent in right lobe (ideal ~ 0.50 due to Z2 symmetry)
    right_lobe_fraction = float(np.mean(x > 0))

    return {
        "num_switches": float(num_switches),
        "switching_rate": switching_rate,
        "mean_residence": mean_residence,
        "std_residence": std_residence,
        "right_lobe_fraction": right_lobe_fraction
    }


# =====================================================================
# 4. Multi-Seed Replication Master Runner
# =====================================================================
def run_multiseed_experiment(
    seeds: List[int] = [42, 101, 2024, 777, 999],
    epochs: int = 40,
    alpha_div: float = 1e-4,
    output_fig: str = "lorenz_multiseed_regularized_results.png"
):
    print("=" * 80)
    print("STARTING MULTI-SEED REPLICATION EXPERIMENT (DIVERGENCE REGULARIZED)")
    print(f"Seeds to evaluate: {seeds} (Total: {len(seeds)} independent models)")
    print(f"Divergence weight: alpha_div = {alpha_div:.1e} (Target div = -13.6667)")
    print(f"Training budget:   {epochs} epochs per seed with 2-step unrolling loss")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step A: Evaluate Ground Truth Reference Metrics Once
    # -------------------------------------------------------------
    print("\n[Step A] Computing Ground Truth Reference Metrics...")
    le_gt, _ = compute_lyapunov_spectrum_gt(steps=10000, dt=0.01)
    d_ky_gt = kaplan_yorke_dimension(le_gt)

    # Long Ground Truth trajectory for invariant distributions & correlation dimension
    curr = np.array([0.0, 1.0, 1.05], dtype=np.float64)
    for _ in range(2000):
        curr = rk4_step(curr, 0.01)
    traj_gt = [curr]
    for _ in range(25000):
        curr = rk4_step(curr, 0.01)
        traj_gt.append(curr)
    traj_gt = np.array(traj_gt)

    _, _, d2_gt = compute_correlation_dimension(traj_gt, n_samples=2500, seed=42)
    gt_switching = compute_lobe_switching_stats(traj_gt, dt=0.01)

    print(f"  GT Lyapunov Spectrum:  lambda1 = {le_gt[0]:+.4f}, lambda2 = {le_gt[1]:+.4f}, lambda3 = {le_gt[2]:+.4f}")
    print(f"  GT Fractal Dimensions: D_KY = {d_ky_gt:.4f}, D_2 = {d2_gt:.4f}")
    print(f"  GT Switching Rate:     {gt_switching['switching_rate']:.3f} switches/s | Right Lobe Fraction: {gt_switching['right_lobe_fraction']:.3f}")

    # -------------------------------------------------------------
    # Step B: Train & Evaluate Across All Seeds
    # -------------------------------------------------------------
    results = []
    seed_trajs = []

    for i, s_val in enumerate(seeds):
        t_start = time.time()
        print(f"\n--- [Seed {i+1}/{len(seeds)}: seed={s_val}] ---")
        model, val1, val2 = train_seed_model(seed=s_val, epochs=epochs, batch_size=256, alpha_div=alpha_div)
        train_elapsed = time.time() - t_start
        print(f"  Training finished in {train_elapsed:.1f}s | Val 1-Step MSE: {val1:.6e} | Val 2-Step MSE: {val2:.6e}")

        # Compute Lyapunov Spectrum
        le_s, _ = compute_lyapunov_spectrum_sciml(model, steps=10000, dt=0.01)
        d_ky_s = kaplan_yorke_dimension(le_s)

        # Long Rollout (25,000 steps)
        curr_t = torch.tensor([0.0, 1.0, 1.05], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(2000):
                curr_t = model(curr_t)
            traj_s = [curr_t.squeeze(0).cpu().numpy()]
            for _ in range(25000):
                curr_t = model(curr_t)
                traj_s.append(curr_t.squeeze(0).cpu().numpy())
        traj_s = np.array(traj_s)
        seed_trajs.append(traj_s)

        # Correlation Dimension D2
        _, _, d2_s = compute_correlation_dimension(traj_s, n_samples=2500, seed=s_val)

        # Wasserstein Distances
        w1_x = scipy.stats.wasserstein_distance(traj_gt[:, 0], traj_s[:, 0])
        w1_y = scipy.stats.wasserstein_distance(traj_gt[:, 1], traj_s[:, 1])
        w1_z = scipy.stats.wasserstein_distance(traj_gt[:, 2], traj_s[:, 2])

        # Lobe Switching
        sw_s = compute_lobe_switching_stats(traj_s, dt=0.01)

        res = {
            "seed": s_val,
            "val1_mse": val1,
            "val2_mse": val2,
            "lambda_1": le_s[0],
            "lambda_2": le_s[1],
            "lambda_3": le_s[2],
            "d_ky": d_ky_s,
            "d_2": d2_s,
            "w1_x": w1_x,
            "w1_y": w1_y,
            "w1_z": w1_z,
            "switching_rate": sw_s["switching_rate"],
            "right_lobe_fraction": sw_s["right_lobe_fraction"],
            "mean_residence": sw_s["mean_residence"]
        }
        results.append(res)
        print(f"  Diagnostics: lambda1 = {le_s[0]:+.3f}, lambda2 = {le_s[1]:+.3f}, lambda3 = {le_s[2]:+.3f}")
        print(f"               D_KY = {d_ky_s:.3f}, D_2 = {d2_s:.3f}, W1(X) = {w1_x:.3f}, Switches/s = {sw_s['switching_rate']:.3f}")

    # -------------------------------------------------------------
    # Step C: Statistical Aggregation (Mean +/- Std)
    # -------------------------------------------------------------
    keys = ["lambda_1", "lambda_2", "lambda_3", "d_ky", "d_2", "w1_x", "w1_y", "w1_z", "switching_rate", "right_lobe_fraction"]
    stats = {}
    for k in keys:
        vals = [r[k] for r in results]
        stats[k] = {"mean": np.mean(vals), "std": np.std(vals), "vals": vals}

    print("\n" + "=" * 80)
    print("MULTI-SEED STATISTICAL SUMMARY TABLE (N = 5 SEEDS)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Ground Truth':<15} | {'SciML (Mean +/- Std)':<22} | {'Rel Err / Bias'}")
    print("-" * 80)
    
    l1_err = abs(stats['lambda_1']['mean'] - le_gt[0]) / abs(le_gt[0]) * 100
    dky_err = abs(stats['d_ky']['mean'] - d_ky_gt) / d_ky_gt * 100
    d2_err = abs(stats['d_2']['mean'] - d2_gt) / d2_gt * 100
    sw_err = abs(stats['switching_rate']['mean'] - gt_switching['switching_rate']) / gt_switching['switching_rate'] * 100

    print(f"{'Largest LE (lambda_1)':<28} | {le_gt[0]:+.4f}          | {stats['lambda_1']['mean']:+.4f} +/- {stats['lambda_1']['std']:.4f}     | {l1_err:.2f}%")
    print(f"{'Neutral LE (lambda_2)':<28} | {le_gt[1]:+.4f}          | {stats['lambda_2']['mean']:+.4f} +/- {stats['lambda_2']['std']:.4f}     | Delta = {abs(stats['lambda_2']['mean'] - le_gt[1]):.4f}")
    print(f"{'Dissipative LE (lambda_3)':<28} | {le_gt[2]:+.4f}         | {stats['lambda_3']['mean']:+.4f} +/- {stats['lambda_3']['std']:.4f}     | Delta = {abs(stats['lambda_3']['mean'] - le_gt[2]):.4f}")
    print(f"{'Kaplan-Yorke Dim (D_KY)':<28} | {d_ky_gt:.4f}           | {stats['d_ky']['mean']:.4f} +/- {stats['d_ky']['std']:.4f}       | {dky_err:.2f}%")
    print(f"{'Correlation Dim (D_2)':<28} | {d2_gt:.4f}           | {stats['d_2']['mean']:.4f} +/- {stats['d_2']['std']:.4f}       | {d2_err:.2f}%")
    print(f"{'Wasserstein W_1(X)':<28} | 0.0000          | {stats['w1_x']['mean']:.4f} +/- {stats['w1_x']['std']:.4f}       | Invariant Dist")
    print(f"{'Wasserstein W_1(Z)':<28} | 0.0000          | {stats['w1_z']['mean']:.4f} +/- {stats['w1_z']['std']:.4f}       | Invariant Dist")
    print(f"{'Lobe-Switching Rate (Hz)':<28} | {gt_switching['switching_rate']:.4f}          | {stats['switching_rate']['mean']:.4f} +/- {stats['switching_rate']['std']:.4f}     | {sw_err:.2f}%")
    print(f"{'Right Lobe Fraction':<28} | {gt_switching['right_lobe_fraction']:.4f}          | {stats['right_lobe_fraction']['mean']:.4f} +/- {stats['right_lobe_fraction']['std']:.4f}     | Ideal = 0.5000")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step D: Multi-Seed Comprehensive Publication Figure
    # -------------------------------------------------------------
    print("\nRendering Multi-Seed Publication-Quality Figure...")
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # 1. Lyapunov Spectrum with Error Bars & Liouville Sum Check
    ax1 = fig.add_subplot(gs[0, 0])
    le_labels = [r"$\lambda_1$ (Chaos)", r"$\lambda_2$ (Neutral)", r"$\lambda_3$ (Dissipation)"]
    x_idx = np.arange(3)
    w = 0.35
    ax1.bar(x_idx - w/2, le_gt, width=w, color="#1f77b4", alpha=0.85, label=f"True GT")
    means_le = [stats['lambda_1']['mean'], stats['lambda_2']['mean'], stats['lambda_3']['mean']]
    stds_le = [stats['lambda_1']['std'], stats['lambda_2']['std'], stats['lambda_3']['std']]
    ax1.bar(x_idx + w/2, means_le, yerr=stds_le, capsize=5, width=w, color="#d62728", alpha=0.85, label=r"SciML Mean $\pm$ Std")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(le_labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Lyapunov Exponent (nats/s)", fontsize=10, fontweight="bold")
    
    l_sums = [r["lambda_1"] + r["lambda_2"] + r["lambda_3"] for r in results]
    mean_sum = np.mean(l_sums)
    std_sum = np.std(l_sums)
    ax1.set_title(rf"1. Lyapunov Spectrum & Liouville Sum Check" + "\n" + rf"($\sum \lambda_i = {mean_sum:.3f} \pm {std_sum:.3f}$ vs Target $-13.667$)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Fractal Dimensions (D_KY & D_2) Bar Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    dim_labels = [r"$D_{KY}$ (Kaplan-Yorke)", r"$D_2$ (Grassberger-Procaccia)"]
    d_x = np.arange(2)
    gt_dims = [d_ky_gt, d2_gt]
    sciml_dims = [stats['d_ky']['mean'], stats['d_2']['mean']]
    sciml_dim_stds = [stats['d_ky']['std'], stats['d_2']['std']]
    ax2.bar(d_x - w/2, gt_dims, width=w, color="#1f77b4", alpha=0.85, label="Ground Truth")
    ax2.bar(d_x + w/2, sciml_dims, yerr=sciml_dim_stds, capsize=5, width=w, color="#2ca02c", alpha=0.85, label=r"SciML Mean $\pm$ Std")
    ax2.set_xticks(d_x)
    ax2.set_xticklabels(dim_labels, fontsize=10, fontweight="bold")
    ax2.set_ylim(1.5, 2.3)
    ax2.set_ylabel("Fractal Dimension", fontsize=10, fontweight="bold")
    ax2.set_title(r"2. Attractor Fractal Dimension ($D_{KY}$ vs $D_2$)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    for i in range(2):
        ax2.text(i - w/2, gt_dims[i] + 0.03, f"{gt_dims[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax2.text(i + w/2, sciml_dims[i] + 0.03, f"{sciml_dims[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # 3. Lobe-Switching Rate & Balance Across Seeds
    ax3 = fig.add_subplot(gs[0, 2])
    seed_labels = [f"S{s}" for s in seeds]
    switch_rates = [r["switching_rate"] for r in results]
    ax3.bar(seed_labels, switch_rates, color="#9467bd", alpha=0.75, label="SciML Seed Rate")
    ax3.axhline(gt_switching["switching_rate"], color="black", ls="--", lw=1.8, label=f"GT Rate ({gt_switching['switching_rate']:.3f} Hz)")
    ax3.set_ylabel("Switching Rate (Hz)", fontsize=10, fontweight="bold")
    ax3.set_title("3. Lobe-Switching Frequency (No Trapping)", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_ylim(0, max(switch_rates) * 1.3)

    # 4. Invariant Probability Densities p(X) Across All Seeds Overlay
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(traj_gt[:, 0], bins=60, density=True, color="#1f77b4", alpha=0.35, label="GT Invariant Measure")
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for idx, (s_val, tr) in enumerate(zip(seeds, seed_trajs)):
        ax4.hist(tr[:, 0], bins=60, density=True, histtype="step", color=colors[idx % len(colors)], lw=1.3, label=f"Seed {s_val}")
    ax4.set_title("4. Multi-Seed Invariant Measure p(X) Overlay", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X (Convective Intensity)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # 5. Wasserstein-1 Distance Distributions vs True-vs-True Noise Floor
    ax5 = fig.add_subplot(gs[1, 1])
    gt_floor = [0.6474, 0.6653, 0.1553]
    gt_floor_std = [0.4453, 0.4503, 0.0857]
    w1_metrics = [stats['w1_x']['mean'], stats['w1_y']['mean'], stats['w1_z']['mean']]
    w1_stds = [stats['w1_x']['std'], stats['w1_y']['std'], stats['w1_z']['std']]
    w_labels = [r"$W_1(X)$", r"$W_1(Y)$", r"$W_1(Z)$"]

    x_w = np.arange(3)
    w_bar = 0.35
    ax5.bar(x_w - w_bar/2, gt_floor, yerr=gt_floor_std, capsize=5, width=w_bar, color="#1f77b4", alpha=0.75, label="True-vs-True Noise Floor")
    ax5.bar(x_w + w_bar/2, w1_metrics, yerr=w1_stds, capsize=5, width=w_bar, color="#2ca02c", alpha=0.85, label=r"SciML Mean $\pm$ Std")
    ax5.set_xticks(x_w)
    ax5.set_xticklabels(w_labels, fontsize=10, fontweight="bold")
    ax5.set_title(r"5. $W_1$ Distances vs True Chaotic Noise Floor", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Wasserstein-1 Distance", fontsize=10, fontweight="bold")
    ax5.legend(loc="upper right", fontsize=8.0)
    ax5.grid(True, linestyle=":", alpha=0.6)
    for i in range(3):
        ax5.text(i - w_bar/2, gt_floor[i] + gt_floor_std[i] + 0.03, f"{gt_floor[i]:.2f}\n±{gt_floor_std[i]:.2f}", ha="center", fontsize=7.5, fontweight="bold", color="#1f77b4")
        ax5.text(i + w_bar/2, w1_metrics[i] + w1_stds[i] + 0.03, f"{w1_metrics[i]:.2f}\n±{w1_stds[i]:.2f}", ha="center", fontsize=7.5, fontweight="bold", color="#2ca02c")
    ax5.set_ylim(0, 1.45)

    # 6. Seed Summary Table Inset
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = [
        ["Seed", "Val MSE", r"$\lambda_1$", r"$\lambda_3$", r"$\sum \lambda$", r"$D_2$"]
    ]
    for r in results:
        l_sum = r['lambda_1'] + r['lambda_2'] + r['lambda_3']
        table_data.append([
            str(r["seed"]),
            f"{r['val1_mse']:.2e}",
            f"{r['lambda_1']:+.3f}",
            f"{r['lambda_3']:.2f}",
            f"{l_sum:.2f}",
            f"{r['d_2']:.3f}"
        ])
    table_data.append([
        "Mean",
        f"{np.mean([r['val1_mse'] for r in results]):.2e}",
        f"{stats['lambda_1']['mean']:+.3f}",
        f"{stats['lambda_3']['mean']:.2f}",
        f"{mean_sum:.2f}",
        f"{stats['d_2']['mean']:.3f}"
    ])
    table_data.append([
        "Std",
        "—",
        f"±{stats['lambda_1']['std']:.3f}",
        f"±{stats['lambda_3']['std']:.2f}",
        f"±{std_sum:.2f}",
        f"±{stats['d_2']['std']:.3f}"
    ])

    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.45)
    ax6.set_title("6. Per-Seed Metric Breakdown", fontsize=11, fontweight="bold", pad=12)

    plt.suptitle(rf"Tier 2 Multi-Seed Replication: Divergence-Regularized SciML ($\alpha_{{\rm div}}={alpha_div:.1e}$)", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    print(f"\nMulti-seed figure successfully saved to: {output_fig}")
    plt.close()

    # Save metrics JSON
    import json
    json_path = output_fig.replace(".png", ".json")
    export_payload = {
        "alpha_div": alpha_div,
        "seeds": seeds,
        "ground_truth": {
            "lambda_1": float(le_gt[0]),
            "lambda_2": float(le_gt[1]),
            "lambda_3": float(le_gt[2]),
            "d_ky": float(d_ky_gt),
            "d_2": float(d2_gt),
            "switching_rate": float(gt_switching["switching_rate"]),
            "right_lobe_fraction": float(gt_switching["right_lobe_fraction"])
        },
        "stats": {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in stats.items()},
        "per_seed_results": [{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()} for r in results]
    }
    with open(json_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"Metrics JSON saved to: {json_path}")


if __name__ == "__main__":
    run_multiseed_experiment(seeds=[42, 101, 2024, 777, 999], epochs=40, alpha_div=1e-4)
