"""
================================================================================
BIFURCATION HORIZON & CURVATURE DEVIATION ANALYSIS: FITZHUGH-NAGUMO
================================================================================
Investigates the geometric tracking accuracy of the 7-seed SciML ensemble
against the ground-truth RK4 bifurcation manifold:
  1. Dense parameter sweep across recovery parameter b in [0.55, 1.25]
     (at a = 0.70, eps = 0.08).
  2. Vectorized batched boundary extraction for Analytical Hopf, RK4 Dynamic
     Limit Cycle, and all 7 SciML seeds.
  3. Overlay of all 7 individual seeds to verify attractor basin consensus.
  4. Disjoint regional analysis correlating boundary deviation Delta I
     with boundary slope s(b) and curvature kappa(b).
  5. Publication-grade 3-panel figure:
     Panel A: Phase-plane attractor (v, w) and waveform comparison.
     Panel B: 7-seed bifurcation boundary overlay vs. Ground Truth RK4.
     Panel C: Pointwise deviation Delta I vs. local curvature kappa(b) and slope.
================================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fhn_global_sciml import (
    ProposedFiLMFHNMLP,
    UnconstrainedBaselineFHNMLP,
    simulate_trajectory,
    rk4_step,
    analytical_fixed_point,
    analytical_hopf_currents,
    DT
)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]


from numba import njit

@njit(fastmath=True)
def fast_check_rk4(I, a, b, eps, dt, warmup, measure):
    v = -1.0
    w = -0.5
    for _ in range(warmup):
        k1v = v - (v * v * v) / 3.0 - w + I
        k1w = eps * (v + a - b * w)
        v2 = v + 0.5 * dt * k1v
        w2 = w + 0.5 * dt * k1w
        k2v = v2 - (v2 * v2 * v2) / 3.0 - w2 + I
        k2w = eps * (v2 + a - b * w2)
        v3 = v + 0.5 * dt * k2v
        w3 = w + 0.5 * dt * k2w
        k3v = v3 - (v3 * v3 * v3) / 3.0 - w3 + I
        k3w = eps * (v3 + a - b * w3)
        v4 = v + dt * k3v
        w4 = w + dt * k3w
        k4v = v4 - (v4 * v4 * v4) / 3.0 - w4 + I
        k4w = eps * (v4 + a - b * w4)
        v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)

    v_min = v
    v_max = v
    for _ in range(measure):
        k1v = v - (v * v * v) / 3.0 - w + I
        k1w = eps * (v + a - b * w)
        v2 = v + 0.5 * dt * k1v
        w2 = w + 0.5 * dt * k1w
        k2v = v2 - (v2 * v2 * v2) / 3.0 - w2 + I
        k2w = eps * (v2 + a - b * w2)
        v3 = v + 0.5 * dt * k2v
        w3 = w + 0.5 * dt * k2w
        k3v = v3 - (v3 * v3 * v3) / 3.0 - w3 + I
        k3w = eps * (v3 + a - b * w3)
        v4 = v + dt * k3v
        w4 = w + dt * k3w
        k4v = v4 - (v4 * v4 * v4) / 3.0 - w4 + I
        k4w = eps * (v4 + a - b * w4)
        v += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        w += (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        if v < v_min: v_min = v
        if v > v_max: v_max = v
    return (v_max - v_min) > 1.0


def check_oscillatory_rk4(I, a, b, eps, dt=DT, warmup_steps=2000, measure_steps=2000):
    return fast_check_rk4(float(I), float(a), float(b), float(eps), float(dt), int(warmup_steps), int(measure_steps))



def bisect_rk4_boundary(a, b, eps, low=0.05, high=0.65, iters=10):
    for _ in range(iters):
        mid = 0.5 * (low + high)
        if check_oscillatory_rk4(mid, a, b, eps):
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def find_model_boundary_batched(model, a, b, eps, low=0.05, high=0.65, n_candidates=61, warmup=1500, measure=1500):
    """
    Vectorized batched rollout in PyTorch:
    Evaluates all candidate I values simultaneously in parallel!
    """
    I_vals = torch.linspace(low, high, n_candidates)
    p_batch = torch.zeros(n_candidates, 4, dtype=torch.float32)
    p_batch[:, 0] = I_vals
    p_batch[:, 1] = float(a)
    p_batch[:, 2] = float(b)
    p_batch[:, 3] = float(eps)

    states = torch.tensor([[-1.0, -0.5]], dtype=torch.float32).repeat(n_candidates, 1)

    model.eval()
    step_fn = model.rk4_step_forward if hasattr(model, 'rk4_step_forward') else model
    with torch.no_grad():
        for _ in range(warmup):
            states = step_fn(states, p_batch)
        v_min = states[:, 0].clone()
        v_max = states[:, 0].clone()
        for _ in range(measure):
            states = step_fn(states, p_batch)
            v_min = torch.min(v_min, states[:, 0])
            v_max = torch.max(v_max, states[:, 0])


    amps = (v_max - v_min).cpu().numpy()
    cross_idx = np.where(amps > 1.0)[0]
    if len(cross_idx) == 0:
        return float(high)
    if cross_idx[0] == 0:
        return float(low)
    
    # Sub-grid linear interpolation for high precision
    idx = cross_idx[0]
    i0 = idx - 1
    i1 = idx
    amp0 = amps[i0]
    amp1 = amps[i1]
    frac = (1.0 - amp0) / max(amp1 - amp0, 1e-6)
    i_est = float(I_vals[i0] + frac * (I_vals[i1] - I_vals[i0]))
    return i_est


def main():
    print("=" * 110, flush=True)
    print("FITZHUGH-NAGUMO BIFURCATION HORIZON & CURVATURE DEVIATION ANALYSIS", flush=True)
    print("=" * 110, flush=True)

    scratch_dir = os.path.join(CURRENT_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    # 1. Load trained SciML models across all 7 seeds
    models = {}
    for seed in SEEDS:
        ckpt_path = os.path.join(scratch_dir, f"fhn_sciml_s{seed}.pt")
        if os.path.exists(ckpt_path):
            m = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
            m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            m.eval()
            models[seed] = m
            print(f"Loaded Proposed SciML Seed {seed}")

    # Load baseline model
    base_ckpt = os.path.join(scratch_dir, "fhn_baseline_s42.pt")
    base_model = None
    if os.path.exists(base_ckpt):
        base_model = UnconstrainedBaselineFHNMLP(hidden_dim=128, dt=DT)
        base_model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
        base_model.eval()
        print("Loaded Unconstrained Baseline SciML Seed 42")

    # 2. Dense parameter slice in b in [0.55, 1.25]
    a_fixed = 0.70
    eps_fixed = 0.08
    b_grid = np.linspace(0.55, 1.25, 15)

    print(f"\nEvaluating Dense Grid ({len(b_grid)} points) across b in [{b_grid[0]:.2f}, {b_grid[-1]:.2f}]...", flush=True)

    b_vals = []
    I_analytical = []
    I_rk4_list = []
    I_seeds_dict = {seed: [] for seed in models.keys()}
    I_baseline_list = []

    t0 = time.time()
    for idx, b in enumerate(b_grid):
        b_vals.append(b)
        
        # Analytical Hopf
        I1_a, _ = analytical_hopf_currents(a_fixed, b, eps_fixed)
        I_analytical.append(I1_a)

        # Dynamic RK4 boundary
        I_rk4 = bisect_rk4_boundary(a_fixed, b, eps_fixed, low=0.05, high=0.65)
        I_rk4_list.append(I_rk4)

        # SciML seeds (batched parallel rollout)
        for seed, m in models.items():
            I_s = find_model_boundary_batched(m, a_fixed, b, eps_fixed, low=0.05, high=0.65, n_candidates=61)
            I_seeds_dict[seed].append(I_s)

        # Baseline
        if base_model is not None:
            I_b = find_model_boundary_batched(base_model, a_fixed, b, eps_fixed, low=0.05, high=0.65, n_candidates=61)
            I_baseline_list.append(I_b)

        print(f"  [{idx+1:2d}/{len(b_grid)}] b={b:.3f} | Analytical: {I1_a:.4f} | RK4: {I_rk4:.4f} | Seeds Avg: {np.mean([I_seeds_dict[s][-1] for s in models.keys()]):.4f}", flush=True)

    print(f"Dense grid evaluation completed in {time.time() - t0:.2f} s.\n", flush=True)

    b_arr = np.array(b_vals)
    I_rk4_arr = np.array(I_rk4_list)
    all_seed_arrays = np.array([I_seeds_dict[s] for s in models.keys()])
    I_consensus_arr = np.mean(all_seed_arrays, axis=0)
    I_std_arr = np.std(all_seed_arrays, axis=0)

    # 3. Deviation, Slope and Curvature Analysis
    delta_I = I_consensus_arr - I_rk4_arr
    rmse = np.sqrt(np.mean(delta_I**2))
    max_dev = np.max(np.abs(delta_I))

    # Fit spline to RK4 boundary to get smooth 1st and 2nd derivatives
    spline = UnivariateSpline(b_arr, I_rk4_arr, s=1e-5)
    dI_db = spline.derivative(1)(b_arr)
    d2I_db2 = spline.derivative(2)(b_arr)
    curvature = np.abs(d2I_db2) / ((1.0 + dI_db**2)**1.5)

    print("=" * 95)
    print("BIFURCATION DEVIATION & CURVATURE SUMMARY")
    print("=" * 95)
    print(f"Overall Ensemble RMSE: {rmse:.4f}")
    print(f"Overall Max |Deviation|: {max_dev:.4f}")
    print(f"{'b':<6} | {'True RK4':<9} | {'SciML Mean':<10} | {'Std':<7} | {'Delta I':<9} | {'Slope':<8} | {'Curvature kappa':<15}")
    print("-" * 95)
    for i in range(len(b_arr)):
        print(f"{b_arr[i]:<6.3f} | {I_rk4_arr[i]:<9.4f} | {I_consensus_arr[i]:<10.4f} | {I_std_arr[i]:<7.4f} | {delta_I[i]:<+9.4f} | {dI_db[i]:<8.4f} | {curvature[i]:<15.4f}")
    print("=" * 95)

    # Disjoint Region Metrics:
    # Region 1: Lower b (b in [0.55, 0.85]) -> Steepest boundary slope & changing curvature
    # Region 2: Upper b (b in [0.90, 1.25]) -> Flattening slope approaching cusp
    mask_r1 = (b_arr <= 0.85)
    mask_r2 = (b_arr > 0.85)

    r1_metrics = {
        "b_range": "[0.55, 0.85]",
        "mean_slope": float(np.mean(dI_db[mask_r1])),
        "peak_curvature": float(np.max(curvature[mask_r1])),
        "mean_dev": float(np.mean(delta_I[mask_r1])),
        "rmse": float(np.sqrt(np.mean(delta_I[mask_r1]**2)))
    }
    r2_metrics = {
        "b_range": "[0.90, 1.25]",
        "mean_slope": float(np.mean(dI_db[mask_r2])),
        "peak_curvature": float(np.max(curvature[mask_r2])),
        "mean_dev": float(np.mean(delta_I[mask_r2])),
        "rmse": float(np.sqrt(np.mean(delta_I[mask_r2]**2)))
    }

    print("\nDISJOINT REGIONAL COMPARISON:")
    print(f"  Region 1 {r1_metrics['b_range']}: Mean Slope = {r1_metrics['mean_slope']:.4f} | Peak Curvature = {r1_metrics['peak_curvature']:.4f} | Mean Dev = {r1_metrics['mean_dev']:+.4f} | RMSE = {r1_metrics['rmse']:.4f}")
    print(f"  Region 2 {r2_metrics['b_range']}: Mean Slope = {r2_metrics['mean_slope']:.4f} | Peak Curvature = {r2_metrics['peak_curvature']:.4f} | Mean Dev = {r2_metrics['mean_dev']:+.4f} | RMSE = {r2_metrics['rmse']:.4f}\n")

    # 4. Generate Publication-Quality 3-Panel Figure (White Academic Background)
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300, facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.30, left=0.06, right=0.95, top=0.88, bottom=0.14)

    # Professional academic colors
    col_rk4 = "#1d4ed8"       # Deep Royal Blue
    col_sciml = "#059669"     # Forest Green
    col_baseline = "#dc2626"  # Crimson Red
    col_hopf = "#64748b"      # Slate Grey
    col_dev = "#d97706"       # Deep Amber
    col_curv = "#7c3aed"      # Royal Purple
    col_grid = "#e2e8f0"      # Light grey
    text_dark = "#0f172a"     # Slate black

    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", color=col_grid, alpha=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.0)
        ax.tick_params(colors=text_dark, labelsize=10.5)

    # --- PANEL A: Attractor Phase Plane & Trajectory Waveform ---
    axA = axes[0]

    # Reference limit cycle trajectory at benchmark (I=0.8, a=0.7, b=0.8)
    bench_p = np.array([0.8, 0.7, 0.8, 0.08])
    bench_traj_rk4 = simulate_trajectory(np.array([-1.0, -0.5]), 3000, DT, bench_p)
    axA.plot(bench_traj_rk4[1000:, 0], bench_traj_rk4[1000:, 1], color=col_rk4, lw=2.4, zorder=3, label="True RK4 Attractor")

    # Plot sample SciML seed trajectories
    if 42 in models:
        m_traj = []
        c = torch.tensor([[-1.0, -0.5]], dtype=torch.float32)
        pt = torch.tensor([bench_p], dtype=torch.float32)
        for _ in range(3000):
            c = models[42](c, pt)
            m_traj.append(c.squeeze(0).detach().numpy())
        m_traj = np.array(m_traj)
        axA.plot(m_traj[1000:, 0], m_traj[1000:, 1], color=col_sciml, lw=2.0, ls="--", zorder=4, label="SciML Seed 42")

    # Mark fixed point
    v_fp, w_fp = analytical_fixed_point(0.8, 0.7, 0.8, 0.08)
    axA.plot(v_fp, w_fp, "o", color=col_baseline, markersize=7, zorder=5, label=f"Unstable FP ({v_fp:.2f}, {w_fp:.2f})")

    axA.set_xlabel("Membrane Potential $v$", fontsize=12, fontweight="bold", color=text_dark)
    axA.set_ylabel("Recovery Variable $w$", fontsize=12, fontweight="bold", color=text_dark)
    axA.set_title("Panel A: Phase-Plane Limit Cycle ($I=0.8$)", fontsize=13, fontweight="bold", color=text_dark, pad=10)
    axA.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    # --- PANEL B: 7-Seed Bifurcation Boundary Overlay ---
    axB = axes[1]

    axB.plot(b_arr, I_analytical, color=col_hopf, lw=1.8, ls=":", zorder=2, label="Analytical Hopf $\\mathrm{Tr}(J)=0$")
    axB.plot(b_arr, I_rk4_arr, color=col_rk4, lw=2.6, marker="o", markersize=6, zorder=4, label="Ground Truth RK4 Dynamic")

    # Overlay all 7 seeds individually
    for s_idx, (seed, s_vals) in enumerate(I_seeds_dict.items()):
        lbl = "Individual Seeds (7 runs)" if s_idx == 0 else None
        axB.plot(b_arr, s_vals, color="#94a3b8", lw=1.0, alpha=0.5, marker="x", markersize=4, zorder=3, label=lbl)

    # Plot Ensemble Consensus
    axB.plot(b_arr, I_consensus_arr, color=col_sciml, lw=2.2, marker="s", markersize=5, zorder=5, label="Ensemble Mean (7 Seeds)")
    axB.fill_between(b_arr, I_consensus_arr - I_std_arr, I_consensus_arr + I_std_arr, color=col_sciml, alpha=0.15, zorder=2)

    # Plot Unconstrained Baseline
    if I_baseline_list:
        axB.plot(b_arr, I_baseline_list, color=col_baseline, lw=1.8, ls="--", zorder=3, label="Baseline Generic MLP")

    axB.set_xlabel("Recovery Parameter $b$", fontsize=12, fontweight="bold", color=text_dark)
    axB.set_ylabel("Critical Injected Current $I_{\\mathrm{Hopf}, 1}$", fontsize=12, fontweight="bold", color=text_dark)
    axB.set_title("Panel B: 7-Seed Bifurcation Manifold Overlay", fontsize=13, fontweight="bold", color=text_dark, pad=10)
    axB.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

    # --- PANEL C: Deviation vs. Curvature & Slope Mechanism ---
    axC = axes[2]

    # Primary axis: Deviation Delta I
    axC.axhline(0.0, color="#64748b", lw=1.2, ls="--", zorder=1)
    # Grid resolution band: e.g. delta b or bisection precision (+/- 0.01)
    axC.axhspan(-0.01, 0.01, color="#cbd5e1", alpha=0.45, zorder=1, label="Grid Spacing Band (±0.01)")
    line1 = axC.plot(b_arr, delta_I, color=col_dev, lw=2.4, marker="o", markersize=6, zorder=4, label="Deviation $\\Delta I = I_{\\mathrm{SciML}} - I_{\\mathrm{True}}$")
    axC.fill_between(b_arr, delta_I - I_std_arr, delta_I + I_std_arr, color=col_dev, alpha=0.15, zorder=2)
    axC.set_ylabel("Boundary Deviation $\\Delta I$", fontsize=12, fontweight="bold", color=col_dev)
    axC.tick_params(axis='y', labelcolor=col_dev)
    axC.set_xlabel("Recovery Parameter $b$", fontsize=12, fontweight="bold", color=text_dark)

    # Secondary axis: Curvature kappa(b)
    axC2 = axC.twinx()
    line2 = axC2.plot(b_arr, curvature, color=col_curv, lw=2.2, ls="-.", marker="^", markersize=5, zorder=4, label="Boundary Curvature $\\kappa(b)$")
    axC2.set_ylabel("Manifold Curvature $\\kappa(b)$", fontsize=12, fontweight="bold", color=col_curv)
    axC2.tick_params(axis='y', labelcolor=col_curv, labelsize=10.5)
    for spine in axC2.spines.values():
        spine.set_color("#334155")

    axC.set_title("Panel C: Capacity Saturation & Curvature Lag", fontsize=13, fontweight="bold", color=text_dark, pad=10)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axC.legend(lines, labels, loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    fig_path = os.path.join(scratch_dir, "fhn_dense_grid_deviation_mechanism.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Publication-grade 3-panel figure saved to: {fig_path}", flush=True)

    # Save metrics JSON
    analysis_json = os.path.join(scratch_dir, "fhn_bifurcation_analysis.json")
    with open(analysis_json, "w") as f:
        json.dump({
            "b_grid": list(b_arr),
            "analytical_hopf": list(I_analytical),
            "rk4_dynamic": list(I_rk4_arr),
            "sciml_consensus": list(I_consensus_arr),
            "sciml_std": list(I_std_arr),
            "delta_I": list(delta_I),
            "slope": list(dI_db),
            "curvature": list(curvature),
            "overall_rmse": float(rmse),
            "overall_max_dev": float(max_dev),
            "region_1": r1_metrics,
            "region_2": r2_metrics
        }, f, indent=2)
    print(f"Detailed analysis data saved to: {analysis_json}\n", flush=True)


if __name__ == "__main__":
    main()
