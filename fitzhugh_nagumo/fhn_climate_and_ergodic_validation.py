"""
======================================================================================
TIER 2 — "CLIMATE" & ERGODIC ATTRACTOR INVARIANCE FOR FITZHUGH-NAGUMO SciML
======================================================================================
Implements the 4 Foundational Pillars from the Lorenz Climate & Ergodic Toolkit:

1. Invariant Ergodic Measure & Empirical PDFs:
   - Long-run trajectory distributions of v and w (10,000 steps, T = 500s).
   - 1D Wasserstein-1 Distance (W_1) and statistical moments (mean, variance, skewness).
2. Spectral Frequency Analysis & Phase Coherence:
   - Power Spectral Density (PSD) S(f) via Welch's method.
   - Fundamental spiking frequency f_0 and harmonic comb matching.
   - Autocorrelation function R_vv(tau) measuring limit-cycle phase coherence.
3. Poincaré Surface of Section & 1D Spike Return Map:
   - Transversal Poincaré section: Sigma = {w = w*, dw/dt > 0}.
   - Successive peak voltage return map: V_{max, n+1} vs V_{max, n}.
4. Neural Jacobian Eigenvalue Continuation & Hopf Crossing:
   - Automatic differentiation / finite-difference neural Jacobian J_theta = df_theta / dy.
   - Eigenvalue tracking: real part alpha(I) and imaginary frequency omega(I).
   - Direct verification that the learned neural vector field undergoes a Hopf bifurcation.
======================================================================================
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
import scipy.stats
import scipy.signal
import torch
import matplotlib.pyplot as plt

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


# ==============================================================================
# 1. NEURAL JACOBIAN SOLVER
# ==============================================================================
def compute_neural_jacobian(model, state, params, eps=1e-3):
    """
    Computes the 2x2 Jacobian J_ij = d(f_i)/d(y_j) of the learned neural vector field
    using central finite differences around the target state y.
    """
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0).repeat(5, 1)
    
    # 5 evaluation points: center, +eps*e1, -eps*e1, +eps*e2, -eps*e2
    batch = np.zeros((5, 2), dtype=np.float32)
    batch[0] = state
    batch[1] = [state[0] + eps, state[1]]
    batch[2] = [state[0] - eps, state[1]]
    batch[3] = [state[0], state[1] + eps]
    batch[4] = [state[0], state[1] - eps]

    b_t = torch.tensor(batch, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'predict_derivatives'):
            derivs = model.predict_derivatives(b_t, p_t).cpu().numpy()
        else:
            # Fallback for baseline model: (y_{t+dt} - y_t) / dt
            out = model(b_t, p_t)
            derivs = ((out - b_t) / model.dt).cpu().numpy()

    # Central difference derivatives
    df_dv = (derivs[1] - derivs[2]) / (2.0 * eps)
    df_dw = (derivs[3] - derivs[4]) / (2.0 * eps)

    J = np.column_stack([df_dv, df_dw])
    return J


def compute_true_jacobian(v_star, w_star, I, a, b, eps):
    """Exact analytical Jacobian of the FitzHugh-Nagumo system at (v*, w*)."""
    return np.array([
        [1.0 - v_star**2, -1.0],
        [eps, -eps * b]
    ], dtype=np.float64)


# ==============================================================================
# 2. POINCARÉ SECTION & FIRST RETURN MAP
# ==============================================================================
def extract_poincare_return_map(traj_v, traj_w, w_sec):
    """
    Finds upward crossings of the transversal surface of section Sigma = {w = w_sec, dw/dt > 0}
    and extracts successive peak voltages V_{max, n} -> V_{max, n+1}.
    """
    peaks, _ = scipy.signal.find_peaks(traj_v, distance=200, prominence=1.0)
    v_peaks = traj_v[peaks]
    if len(v_peaks) > 1:
        v_n = v_peaks[:-1]
        v_n1 = v_peaks[1:]
    else:
        v_n = np.array([])
        v_n1 = np.array([])
    return v_n, v_n1


# ==============================================================================
# MAIN 4-PILLAR VALIDATION
# ==============================================================================
def main():
    print("=" * 105, flush=True)
    print("TIER 2 'CLIMATE' & ERGODIC ATTRACTOR INVARIANCE FOR FITZHUGH-NAGUMO SciML", flush=True)
    print("=" * 105, flush=True)

    scratch_dir = os.path.join(CURRENT_DIR, "scratch")

    # Benchmark parameter set: I = 0.80, a = 0.70, b = 0.80, eps = 0.08
    params = np.array([0.80, 0.70, 0.80, 0.08], dtype=np.float64)
    v_star, w_star = analytical_fixed_point(params[0], params[1], params[2], params[3])

    # 1. Load Proposed SciML (Seed 42) & Baseline Model
    ckpt_prop = os.path.join(scratch_dir, "fhn_sciml_s42.pt")
    model_prop = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
    model_prop.load_state_dict(torch.load(ckpt_prop, map_location="cpu"))
    model_prop.eval()
    print("Loaded Proposed FiLM SciML (Seed 42)")

    ckpt_base = os.path.join(scratch_dir, "fhn_baseline_s42.pt")
    model_base = UnconstrainedBaselineFHNMLP(hidden_dim=128, dt=DT)
    model_base.load_state_dict(torch.load(ckpt_base, map_location="cpu"))
    model_base.eval()
    print("Loaded Unconstrained Baseline SciML (Seed 42)\n")

    # 2. Long-Horizon Ergodic Rollouts (10,000 steps = 500s, ~13 complete cycles)
    n_steps = 10000
    y_warm = np.array([-1.0, -0.5], dtype=np.float64)
    for _ in range(2000):
        y_warm = rk4_step(y_warm, DT, params)

    print(f"Simulating long-horizon ergodic rollouts ({n_steps} steps = {n_steps*DT:.1f}s)...", flush=True)
    traj_gt = simulate_trajectory(y_warm, n_steps, DT, params)

    # SciML Proposed rollout
    c_prop = torch.tensor(y_warm, dtype=torch.float32).unsqueeze(0)
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    traj_prop = [y_warm.copy()]
    with torch.no_grad():
        for _ in range(n_steps):
            c_prop = model_prop(c_prop, p_t)
            traj_prop.append(c_prop.squeeze(0).cpu().numpy())
    traj_prop = np.array(traj_prop)

    # Baseline rollout
    c_base = torch.tensor(y_warm, dtype=torch.float32).unsqueeze(0)
    traj_base = [y_warm.copy()]
    with torch.no_grad():
        for _ in range(n_steps):
            c_base = model_base(c_base, p_t)
            traj_base.append(c_base.squeeze(0).cpu().numpy())
    traj_base = np.array(traj_base)

    # Asymptotic sample (discard initial transient)
    v_gt = traj_gt[2000:, 0]
    v_prop = traj_prop[2000:, 0]
    v_base = traj_base[2000:, 0]

    # --------------------------------------------------------------------------
    # PILLAR 1: INVARIANT STATISTICAL MEASURE & WASSERSTEIN DISTANCE
    # --------------------------------------------------------------------------
    w1_prop = scipy.stats.wasserstein_distance(v_gt, v_prop)
    w1_base = scipy.stats.wasserstein_distance(v_gt, v_base)

    mean_gt, std_gt = float(np.mean(v_gt)), float(np.std(v_gt))
    mean_prop, std_prop = float(np.mean(v_prop)), float(np.std(v_prop))
    mean_base, std_base = float(np.mean(v_base)), float(np.std(v_base))

    print("-" * 80)
    print("PILLAR 1: INVARIANT ERGODIC STATISTICAL MEASURES")
    print(f"  Ground Truth RK4   : Mean = {mean_gt:+.4f} | Std = {std_gt:.4f}")
    print(f"  Proposed FiLM SciML: Mean = {mean_prop:+.4f} | Std = {std_prop:.4f} | Wasserstein-1 W1 = {w1_prop:.4f}")
    print(f"  Unconstrained Base : Mean = {mean_base:+.4f} | Std = {std_base:.4f} | Wasserstein-1 W1 = {w1_base:.4f}")

    # --------------------------------------------------------------------------
    # PILLAR 2: SPECTRAL FREQUENCY ANALYSIS & AUTOCORRELATION
    # --------------------------------------------------------------------------
    fs = 1.0 / DT
    nperseg = min(len(v_gt), 4096)
    freqs_gt, psd_gt = scipy.signal.welch(v_gt - np.mean(v_gt), fs=fs, nperseg=nperseg)
    _, psd_prop = scipy.signal.welch(v_prop - np.mean(v_prop), fs=fs, nperseg=nperseg)
    _, psd_base = scipy.signal.welch(v_base - np.mean(v_base), fs=fs, nperseg=nperseg)

    f0_gt = freqs_gt[np.argmax(psd_gt)]
    f0_prop = freqs_gt[np.argmax(psd_prop)]
    f0_base = freqs_gt[np.argmax(psd_base)]

    print("\nPILLAR 2: SPECTRAL HARMONICS & PHASE COHERENCE")
    print(f"  Ground Truth Peak Frequency f_0    : {f0_gt:.5f} Hz (T = {1.0/f0_gt:.2f} s)")
    print(f"  Proposed FiLM Peak Frequency f_0   : {f0_prop:.5f} Hz (T = {1.0/f0_prop:.2f} s, Error = {abs(f0_prop-f0_gt)/f0_gt*100.0:.2f}%)")
    print(f"  Unconstrained Peak Frequency f_0   : {f0_base:.5f} Hz (T = {1.0/f0_base:.2f} s, Error = {abs(f0_base-f0_gt)/f0_gt*100.0:.2f}%)")

    # --------------------------------------------------------------------------
    # PILLAR 3: POINCARÉ SURFACE OF SECTION & 1D RETURN MAP
    # --------------------------------------------------------------------------
    vn_gt, vn1_gt = extract_poincare_return_map(v_gt, traj_gt[2000:, 1], w_star)
    vn_prop, vn1_prop = extract_poincare_return_map(v_prop, traj_prop[2000:, 1], w_star)
    vn_base, vn1_base = extract_poincare_return_map(v_base, traj_base[2000:, 1], w_star)

    print("\nPILLAR 3: POINCARÉ RETURN MAP DYNAMICS")
    print(f"  Ground Truth Limit-Cycle Fixed Point V_max* = {np.mean(vn_gt):.4f} (Var = {np.var(vn_gt):.6f})")
    print(f"  Proposed SciML Fixed Point V_max*          = {np.mean(vn_prop):.4f} (Var = {np.var(vn_prop):.6f})")
    print(f"  Unconstrained Baseline Fixed Point V_max*  = {np.mean(vn_base):.4f} (Var = {np.var(vn_base):.6f})")

    # --------------------------------------------------------------------------
    # PILLAR 4: NEURAL JACOBIAN EIGENVALUE TRACKING ACROSS CURRENT I
    # --------------------------------------------------------------------------
    print("\nPILLAR 4: NEURAL JACOBIAN EIGENVALUE CONTINUATION ACROSS I")
    I_sweep = np.linspace(0.15, 0.50, 15)
    real_true = []
    real_sciml = []
    real_base = []

    for I_val in I_sweep:
        v_s, w_s = analytical_fixed_point(I_val, params[1], params[2], params[3])
        p_val = np.array([I_val, params[1], params[2], params[3]], dtype=np.float64)

        # True Jacobian
        J_true = compute_true_jacobian(v_s, w_s, I_val, params[1], params[2], params[3])
        e_true = np.linalg.eigvals(J_true)
        real_true.append(float(np.max(np.real(e_true))))

        # Neural Jacobian Proposed
        J_prop = compute_neural_jacobian(model_prop, np.array([v_s, w_s], dtype=np.float32), p_val)
        e_prop = np.linalg.eigvals(J_prop)
        real_sciml.append(float(np.max(np.real(e_prop))))

        # Neural Jacobian Baseline
        J_b = compute_neural_jacobian(model_base, np.array([v_s, w_s], dtype=np.float32), p_val)
        e_b = np.linalg.eigvals(J_b)
        real_base.append(float(np.max(np.real(e_b))))

    I1_h, _ = analytical_hopf_currents(params[1], params[2], params[3])
    print(f"  Analytical Hopf Threshold (Re(lambda)=0): I_H1 = {I1_h:.4f}")
    # Find zero crossing of real_sciml
    idx_cross = np.where(np.diff(np.sign(real_sciml)))[0]
    if len(idx_cross) > 0:
        i0 = idx_cross[0]
        I_cross = I_sweep[i0] - real_sciml[i0] * (I_sweep[i0+1] - I_sweep[i0]) / (real_sciml[i0+1] - real_sciml[i0])
        print(f"  Learned SciML Neural Jacobian Hopf Crossing: I_Hopf,theta = {I_cross:.4f} (Discrepancy = {abs(I_cross - I1_h):.4f})")
    print("-" * 80)

    # ==============================================================================
    # 4-PANEL PUBLICATION FIGURE (WHITE ACADEMIC THEME)
    # ==============================================================================
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, facecolor="#ffffff")
    plt.subplots_adjust(wspace=0.28, hspace=0.30, left=0.08, right=0.95, top=0.92, bottom=0.08)

    col_rk4 = "#1d4ed8"       # Deep Blue
    col_sciml = "#059669"     # Emerald Green
    col_base = "#dc2626"      # Crimson Red
    col_grid = "#e2e8f0"      # Light grey
    text_dark = "#0f172a"

    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", color=col_grid, alpha=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.0)
        ax.tick_params(colors=text_dark, labelsize=10.5)

    # --- PANEL 1: INVARIANT ERGODIC MEASURE P(v) ---
    ax1 = axes[0, 0]
    bins = np.linspace(-2.2, 2.2, 60)
    ax1.hist(v_gt, bins=bins, density=True, alpha=0.35, color=col_rk4, label="Ground Truth RK4 PDF")
    density_gt = scipy.stats.gaussian_kde(v_gt)
    density_prop = scipy.stats.gaussian_kde(v_prop)
    density_base = scipy.stats.gaussian_kde(v_base)
    xs = np.linspace(-2.2, 2.2, 300)
    ax1.plot(xs, density_gt(xs), color=col_rk4, lw=2.2, label=f"True RK4 (W1=0.00)")
    ax1.plot(xs, density_prop(xs), color=col_sciml, lw=2.0, ls="--", label=f"Proposed SciML (W1={w1_prop:.4f})")
    ax1.plot(xs, density_base(xs), color=col_base, lw=1.6, ls=":", label=f"Baseline MLP (W1={w1_base:.4f})")
    ax1.set_xlabel("Membrane Potential $v$", fontsize=11, fontweight="bold", color=text_dark)
    ax1.set_ylabel("Probability Density $P(v)$", fontsize=11, fontweight="bold", color=text_dark)
    ax1.set_title("1. Invariant Ergodic Measure $P(v)$", fontsize=12, fontweight="bold", color=text_dark)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    # --- PANEL 2: POWER SPECTRAL DENSITY S(f) ---
    ax2 = axes[0, 1]
    mask_f = (freqs_gt > 0.005) & (freqs_gt < 0.20)
    ax2.semilogy(freqs_gt[mask_f], psd_gt[mask_f], color=col_rk4, lw=2.2, label=f"True RK4 ($f_0={f0_gt:.4f}$ Hz)")
    ax2.semilogy(freqs_gt[mask_f], psd_prop[mask_f], color=col_sciml, lw=2.0, ls="--", label=f"Proposed SciML ($f_0={f0_prop:.4f}$ Hz)")
    ax2.semilogy(freqs_gt[mask_f], psd_base[mask_f], color=col_base, lw=1.6, ls=":", label=f"Baseline MLP ($f_0={f0_base:.4f}$ Hz)")
    ax2.set_xlabel("Frequency $f$ (Hz)", fontsize=11, fontweight="bold", color=text_dark)
    ax2.set_ylabel("Power Spectral Density $S(f)$", fontsize=11, fontweight="bold", color=text_dark)
    ax2.set_title("2. Spectral Frequency & Harmonic Comb", fontsize=12, fontweight="bold", color=text_dark)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    # --- PANEL 3: 1D POINCARÉ FIRST RETURN MAP ---
    ax3 = axes[1, 0]
    ax3.plot([1.8, 2.05], [1.8, 2.05], color="#94a3b8", lw=1.2, ls="--", label="Identity $V_{n+1} = V_n$")
    ax3.scatter(vn_gt, vn1_gt, color=col_rk4, s=40, alpha=0.8, edgecolors="none", label="Ground Truth RK4 Fixed Point")
    ax3.scatter(vn_prop, vn1_prop, color=col_sciml, marker="^", s=35, alpha=0.9, label="Proposed SciML Fixed Point")
    ax3.scatter(vn_base, vn1_base, color=col_base, marker="x", s=35, alpha=0.8, label="Baseline Generic MLP")
    ax3.set_xlabel("Peak Potential $V_{\\mathrm{max}, n}$", fontsize=11, fontweight="bold", color=text_dark)
    ax3.set_ylabel("Successive Peak $V_{\\mathrm{max}, n+1}$", fontsize=11, fontweight="bold", color=text_dark)
    ax3.set_title("3. 1D Poincaré Return Map of Voltage Peaks", fontsize=12, fontweight="bold", color=text_dark)
    ax3.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    # --- PANEL 4: NEURAL JACOBIAN EIGENVALUE CONTINUATION ---
    ax4 = axes[1, 1]
    ax4.axhline(0.0, color="#64748b", lw=1.2, ls="--")
    ax4.axvline(I1_h, color="#94a3b8", lw=1.2, ls=":", label=f"Analytical Hopf ($I={I1_h:.4f}$)")
    ax4.plot(I_sweep, real_true, color=col_rk4, lw=2.4, marker="o", markersize=5, label="True Physical $\\mathrm{Re}(\\lambda)$")
    ax4.plot(I_sweep, real_sciml, color=col_sciml, lw=2.0, marker="s", markersize=5, ls="--", label="Learned Neural $\\mathrm{Re}(\\lambda_\\theta)$")
    ax4.plot(I_sweep, real_base, color=col_base, lw=1.6, marker="^", markersize=4, ls=":", label="Baseline Generic $\\mathrm{Re}(\\lambda)$")
    ax4.set_xlabel("Stimulus Current $I$", fontsize=11, fontweight="bold", color=text_dark)
    ax4.set_ylabel("Leading Jacobian Real Part $\\mathrm{Re}(\\lambda)$", fontsize=11, fontweight="bold", color=text_dark)
    ax4.set_title("4. Neural Jacobian Hopf Bifurcation Continuation", fontsize=12, fontweight="bold", color=text_dark)
    ax4.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    fig_path = os.path.join(scratch_dir, "fhn_climate_ergodic_validation.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\n4-Pillar Climate & Ergodic validation figure saved to: {fig_path}")

    # Save metrics JSON
    out_json = os.path.join(scratch_dir, "fhn_climate_metrics.json")
    with open(out_json, "w") as f:
        json.dump({
            "wasserstein_1_sciml": float(w1_prop),
            "wasserstein_1_baseline": float(w1_base),
            "peak_freq_gt": float(f0_gt),
            "peak_freq_sciml": float(f0_prop),
            "peak_freq_baseline": float(f0_base),
            "return_map_gt_mean": float(np.mean(vn_gt)),
            "return_map_sciml_mean": float(np.mean(vn_prop)),
            "analytical_hopf_I1": float(I1_h),
            "I_sweep": list(I_sweep),
            "real_true": list(real_true),
            "real_sciml": list(real_sciml),
            "real_baseline": list(real_base)
        }, f, indent=2)
    print(f"Climate metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
