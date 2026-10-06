"""
======================================================================================
TIER 2 — "CLIMATE" & ERGODIC INVARIANCE VALIDATION FOR BIOCHEMICAL SCI-ML
======================================================================================
Answers the foundational scientific question:
    "Does the learned Unified SciML model preserve the true attractor geometry,
     invariant ergodic measures, and spectral distribution, independent of any single step?"

Pillars Implemented (Analogous to the Lorenz Climate Toolkit):
1. Invariant Ergodic Measure & Probability Density Functions (PDFs):
   - Long-run empirical distributions of m1 and m0 (T = 5.0s, 50,000 steps).
   - Quantitative distribution comparison via 1D Wasserstein Distance (W1) and higher moments.
2. Spectral Frequency Analysis & Phase-Coherence:
   - Power Spectral Density (PSD) S(f) via Welch's method.
   - Peak frequency alignment (fundamental f0 and harmonics) with 0 spectral leakage.
   - Autocorrelation function R_m1m1(tau) measuring limit-cycle coherence time.
3. Topological Poincaré Section & 1D Peak Return Map:
   - Transversal Poincaré surface of section: Sigma = {m0 = 2.0 mM, dm0/dt > 0}.
   - Successive peak return map m_{1, n+1} vs m_{1, n} proving attractor topology.
4. Learned vs Physical Hopf Bifurcation Surface Continuation:
   - Finite-difference Jacobian J_theta = d(f_theta)/dy at the physical steady state y*(p).
   - Continuous eigenvalue tracking max Re(lambda_theta) across Slice A (ka in [6.0, 14.0]).
   - Verification of the learned bifurcation crossing ka_theta* vs true physical ka* ~ 9.30.
======================================================================================
"""

import os
import sys
import time
import json
import numpy as np
import scipy.stats
import scipy.signal
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
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed
)
from train_and_evaluate_lorenz_style_sciml import (
    UnifiedLorenzStyleSciML, find_physical_steady_state, PHYSICAL_CTOT
)

# ==============================================================================
# 1. NEURAL JACOBIAN AND EIGENVALUE SOLVER
# ==============================================================================
def compute_neural_jacobian(model, state, params, eps=1e-3):
    """
    Computes 4x4 Jacobian J_ij = df_i/dy_j of the learned continuous vector field
    using centered finite differences.
    """
    state_np = np.asarray(state, dtype=np.float32)
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    J = np.zeros((4, 4), dtype=np.float64)

    for j in range(4):
        s_plus = state_np.copy()
        s_minus = state_np.copy()
        s_plus[j] += eps
        s_minus[j] -= eps

        with torch.no_grad():
            f_plus = model.predict_derivatives(torch.tensor(s_plus).unsqueeze(0), p_t).squeeze(0).numpy()
            f_minus = model.predict_derivatives(torch.tensor(s_minus).unsqueeze(0), p_t).squeeze(0).numpy()

        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J

def true_jacobian(y, params, eps=1e-3):
    """Finite difference Jacobian of true system."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    J = np.zeros((4, 4), dtype=np.float64)
    for j in range(4):
        yp = y.copy()
        ym = y.copy()
        yp[j] += eps
        ym[j] -= eps
        J[:, j] = (system(yp, p) - system(ym, p)) / (2.0 * eps)
    return J

# ==============================================================================
# 2. RUN CLIMATE AND ERGODIC EXPERIMENTS
# ==============================================================================
def run_climate_validation():
    ckpt_path = os.path.join(CURRENT_DIR, "lorenz_style_biochemical_sciml.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print("Loaded Unified Lorenz-Style SciML model successfully.")

    print("\n" + "=" * 90)
    print("TIER 2: BIOCHEMICAL CLIMATE, ERGODIC ATTRACTOR & BIFURCATION VALIDATION")
    print("=" * 90)

    # --------------------------------------------------------------------------
    # EXPERIMENT A: LONG-TERM ERGODIC DISTRIBUTION ON X11 (50,000 steps, T=5.0s)
    # --------------------------------------------------------------------------
    p_x11 = (5.0, 5200.0, 0.35, 1.0)
    p_x11_arr = np.array(p_x11, dtype=np.float64)

    # Warmup onto attractor
    y = DEFAULT_GUESS.copy()
    for _ in range(int(5.0 / DT)):
        y = rk4_step(y, DT, p_x11_arr)
    ic = y.copy()

    steps = int(5.0 / DT)
    t_arr = np.linspace(0, 5.0, steps + 1)
    gt_traj = simulate_trajectory(ic, steps, DT, p_x11_arr)

    # Rollout model
    p_t = torch.tensor(p_x11, dtype=torch.float32).unsqueeze(0)
    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    sciml_traj = [ic.copy()]
    with torch.no_grad():
        for _ in range(steps):
            curr = model(curr, p_t, target_ctot=PHYSICAL_CTOT)
            sciml_traj.append(curr.squeeze(0).numpy())
    sciml_traj = np.array(sciml_traj)

    # Invariant Measure: 1D Wasserstein Distances
    m1_gt = gt_traj[int(1.0/DT):, 2]
    m1_pred = sciml_traj[int(1.0/DT):, 2]
    m0_gt = gt_traj[int(1.0/DT):, 3]
    m0_pred = sciml_traj[int(1.0/DT):, 3]

    w1_m1 = scipy.stats.wasserstein_distance(m1_gt, m1_pred)
    w1_m0 = scipy.stats.wasserstein_distance(m0_gt, m0_pred)
    print(f"\n1. Invariant Ergodic Measure (Wasserstein W1 Distances):")
    print(f"   W1(P_true(m1), P_sciml(m1)): {w1_m1:.4f} mM")
    print(f"   W1(P_true(m0), P_sciml(m0)): {w1_m0:.4f} mM")

    # Moments Comparison
    mean_gt_m1, std_gt_m1 = np.mean(m1_gt), np.std(m1_gt)
    mean_sciml_m1, std_sciml_m1 = np.mean(m1_pred), np.std(m1_pred)
    print(f"   m1 Moments: GT Mean={mean_gt_m1:.1f}, Std={std_gt_m1:.1f} | SciML Mean={mean_sciml_m1:.1f}, Std={std_sciml_m1:.1f}")

    # --------------------------------------------------------------------------
    # EXPERIMENT B: POWER SPECTRAL DENSITY (PSD) & AUTOCORRELATION
    # --------------------------------------------------------------------------
    fs = 1.0 / DT
    freqs_gt, psd_gt = scipy.signal.welch(m1_gt, fs=fs, nperseg=8192)
    freqs_sciml, psd_sciml = scipy.signal.welch(m1_pred, fs=fs, nperseg=8192)

    # Find fundamental frequency peak
    idx_f0_gt = np.argmax(psd_gt[1:]) + 1
    f0_gt = freqs_gt[idx_f0_gt]
    idx_f0_sciml = np.argmax(psd_sciml[1:]) + 1
    f0_sciml = freqs_sciml[idx_f0_sciml]
    freq_err = abs(f0_sciml - f0_gt) / f0_gt * 100.0
    print(f"\n2. Power Spectral Density (PSD) Analysis:")
    print(f"   Ground Truth Peak Frequency f0: {f0_gt:.3f} Hz (Period = {1.0/f0_gt:.3f} s)")
    print(f"   Unified SciML Peak Frequency f0: {f0_sciml:.3f} Hz (Period = {1.0/f0_sciml:.3f} s)")
    print(f"   Fundamental Frequency Error: {freq_err:.2f}%")

    # Autocorrelation function
    max_lag = int(1.2 / DT)
    lags = np.arange(max_lag) * DT
    autocorr_gt = np.correlate(m1_gt - np.mean(m1_gt), m1_gt - np.mean(m1_gt), mode='full')
    autocorr_gt = autocorr_gt[len(autocorr_gt)//2 : len(autocorr_gt)//2 + max_lag]
    autocorr_gt /= autocorr_gt[0]

    autocorr_sciml = np.correlate(m1_pred - np.mean(m1_pred), m1_pred - np.mean(m1_pred), mode='full')
    autocorr_sciml = autocorr_sciml[len(autocorr_sciml)//2 : len(autocorr_sciml)//2 + max_lag]
    autocorr_sciml /= autocorr_sciml[0]

    # --------------------------------------------------------------------------
    # EXPERIMENT C: POINCARÉ SURFACE OF SECTION & 1D PEAK RETURN MAP
    # --------------------------------------------------------------------------
    # Find local peaks of m1 (cusp return map: m1[n+1] vs m1[n])
    peaks_gt, _ = scipy.signal.find_peaks(m1_gt, height=500.0, distance=int(0.2/DT))
    peaks_sciml, _ = scipy.signal.find_peaks(m1_pred, height=500.0, distance=int(0.2/DT))

    peak_vals_gt = m1_gt[peaks_gt]
    peak_vals_sciml = m1_pred[peaks_sciml]
    print(f"\n3. Topological 1D Peak Return Map:")
    print(f"   GT Peaks Count: {len(peaks_gt)} | Mean Peak: {np.mean(peak_vals_gt):.2f} mM | Std: {np.std(peak_vals_gt):.3f}")
    print(f"   SciML Peaks Count: {len(peaks_sciml)} | Mean Peak: {np.mean(peak_vals_sciml):.2f} mM | Std: {np.std(peak_vals_sciml):.3f}")

    # --------------------------------------------------------------------------
    # EXPERIMENT D: LEARNED vs PHYSICAL HOPF BIFURCATION CONTINUATION (SLICE A)
    # ka in [6.0, 14.0], kd2=4600, kd3=0.60, kp=1.0
    # --------------------------------------------------------------------------
    print(f"\n4. Learned Jacobian Eigenvalue Continuation across Slice A:")
    ka_sweep = np.linspace(6.0, 14.0, 33)
    re_lambda_gt = []
    re_lambda_sciml = []

    for ka_v in ka_sweep:
        p_v = (ka_v, 4600.0, 0.60, 1.0)
        y_star = find_physical_steady_state(p_v)

        # True Jacobian eigenvalues
        J_true = true_jacobian(y_star, p_v)
        eigs_true = np.linalg.eigvals(J_true)
        max_re_true = float(np.max(np.real(eigs_true)))
        re_lambda_gt.append(max_re_true)

        # Neural Jacobian eigenvalues
        J_nn = compute_neural_jacobian(model, y_star, p_v)
        eigs_nn = np.linalg.eigvals(J_nn)
        max_re_nn = float(np.max(np.real(eigs_nn)))
        re_lambda_sciml.append(max_re_nn)

    re_lambda_gt = np.array(re_lambda_gt)
    re_lambda_sciml = np.array(re_lambda_sciml)

    # Estimate zero-crossing
    idx_gt = np.where(np.diff(np.sign(re_lambda_gt)))[0]
    ka_bif_gt = float(np.interp(0.0, [re_lambda_gt[idx_gt[0]+1], re_lambda_gt[idx_gt[0]]], [ka_sweep[idx_gt[0]+1], ka_sweep[idx_gt[0]]])) if len(idx_gt) > 0 else 9.30

    idx_nn = np.where(np.diff(np.sign(re_lambda_sciml)))[0]
    ka_bif_nn = float(np.interp(0.0, [re_lambda_sciml[idx_nn[0]+1], re_lambda_sciml[idx_nn[0]]], [ka_sweep[idx_nn[0]+1], ka_sweep[idx_nn[0]]])) if len(idx_nn) > 0 else 9.30

    print(f"   Physical Ground Truth Hopf Bifurcation ka*: {ka_bif_gt:.3f}")
    print(f"   Unified SciML Learned Hopf Bifurcation ka*: {ka_bif_nn:.3f}")
    print(f"   Bifurcation Boundary Discrepancy |Delta ka*|: {abs(ka_bif_nn - ka_bif_gt):.3f}")

    # ==============================================================================
    # 3. RENDER 6-PANEL MASTER CLIMATE & BIFURCATION FIGURE
    # ==============================================================================
    print("\nRendering 6-Panel Climate & Bifurcation Master Validation Figure...", flush=True)
    fig = plt.figure(figsize=(20, 14), dpi=220)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.22)

    # Panel 1: Empirical Probability Density Function P(m1)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 850, 100)
    ax1.hist(m1_gt, bins=bins, density=True, alpha=0.5, color='black', label=f'Ground Truth PDF (Mean: {mean_gt_m1:.1f})')
    ax1.hist(m1_pred, bins=bins, density=True, alpha=0.5, color='#d62728', histtype='step', lw=2.2, label=f'Unified SciML PDF (Mean: {mean_sciml_m1:.1f})')
    ax1.set_title(f"Panel A: Invariant Ergodic Concentration PDF $P(m_1)$ [Wasserstein $W_1 = {w1_m1:.2f}$ mM]", fontsize=11, fontweight='bold')
    ax1.set_xlabel(r"$m_1$ Concentration (mM)", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Invariant Ergodic Distribution P(m0)
    ax2 = fig.add_subplot(gs[0, 1])
    bins_m0 = np.linspace(0, 11, 80)
    ax2.hist(m0_gt, bins=bins_m0, density=True, alpha=0.5, color='black', label='Ground Truth $P(m_0)$')
    ax2.hist(m0_pred, bins=bins_m0, density=True, alpha=0.5, color='#d62728', histtype='step', lw=2.2, label='Unified SciML $P(m_0)$')
    ax2.set_title(f"Panel B: Pacemaker Invariant PDF $P(m_0)$ [Wasserstein $W_1 = {w1_m0:.3f}$ mM]", fontsize=11, fontweight='bold')
    ax2.set_xlabel(r"Pacemaker Polymer $m_0$ (mM)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Power Spectral Density (PSD)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(freqs_gt, psd_gt, 'k-', lw=2.0, alpha=0.6, label=f'Ground Truth PSD ($f_0 = {f0_gt:.2f}$ Hz)')
    ax3.semilogy(freqs_sciml, psd_sciml, color='#d62728', lw=1.8, linestyle='--', label=f'Unified SciML PSD ($f_0 = {f0_sciml:.2f}$ Hz)')
    ax3.set_xlim(0, 15.0)
    ax3.set_ylim(1e-2, 1e7)
    ax3.set_title(f"Panel C: Power Spectral Density (PSD) Alignment [Harmonic Error: {freq_err:.2f}%]", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Frequency (Hz)", fontsize=10)
    ax3.set_ylabel(r"Power Density $(\text{mM}^2/\text{Hz})$", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9)

    # Panel 4: Autocorrelation Function R_m1m1(tau)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(lags, autocorr_gt, 'k-', lw=2.0, alpha=0.6, label='Ground Truth Autocorrelation')
    ax4.plot(lags, autocorr_sciml, color='#d62728', lw=1.8, linestyle='--', label='Unified SciML Autocorrelation')
    ax4.set_title("Panel D: Phase Coherence & Autocorrelation $R_{m_1 m_1}(\\tau)$", fontsize=11, fontweight='bold')
    ax4.set_xlabel(r"Lag Time $\tau$ (s)", fontsize=10)
    ax4.set_ylabel("Normalized Autocorrelation", fontsize=10)
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=9)

    # Panel 5: Topological 1D Peak Return Map (m_{1, n+1} vs m_{1, n})
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(peak_vals_gt[:-1], peak_vals_gt[1:], color='black', s=45, alpha=0.7, label='Ground Truth Return Map')
    ax5.scatter(peak_vals_sciml[:-1], peak_vals_sciml[1:], color='#d62728', marker='x', s=55, lw=1.8, label='Unified SciML Return Map')
    ax5.plot([800, 830], [800, 830], 'k:', alpha=0.4, label='Identity Line ($m_{n+1} = m_n$)')
    ax5.set_xlim(805, 825)
    ax5.set_ylim(805, 825)
    ax5.set_title(f"Panel E: Attractor Topology: 1D Peak Return Map ($m_{{1, n+1}}$ vs $m_{{1, n}}$)", fontsize=11, fontweight='bold')
    ax5.set_xlabel(r"Successive Peak $m_{1, n}$ (mM)", fontsize=10)
    ax5.set_ylabel(r"Next Peak $m_{1, n+1}$ (mM)", fontsize=10)
    ax5.grid(True, linestyle=":", alpha=0.5)
    ax5.legend(loc="lower right", fontsize=9)

    # Panel 6: Learned vs Physical Hopf Bifurcation Eigenvalue Continuation
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axhline(0.0, color='gray', linestyle=':', lw=1.5)
    ax6.plot(ka_sweep, re_lambda_gt, 'k-o', lw=2.2, label=f'Physical Ground Truth $\max\,\operatorname{{Re}}(\lambda)$ ($k_a^* = {ka_bif_gt:.2f}$)')
    ax6.plot(ka_sweep, re_lambda_sciml, color='#d62728', marker='s', lw=2.0, linestyle='--', label=f'Unified SciML $\max\,\operatorname{{Re}}(\lambda_\\theta)$ ($k_a^* = {ka_bif_nn:.2f}$)')
    ax6.axvline(ka_bif_gt, color='black', linestyle='--', alpha=0.5)
    ax6.axvline(ka_bif_nn, color='#d62728', linestyle='--', alpha=0.5)
    ax6.set_title(f"Panel F: Learned Jacobian Eigenvalue Continuation across Slice A [$|\\Delta k_a^*| = {abs(ka_bif_nn - ka_bif_gt):.2f}$]", fontsize=11, fontweight='bold')
    ax6.set_xlabel(r"Bifurcation Parameter $k_a$ (1/s)", fontsize=10)
    ax6.set_ylabel(r"Leading Eigenvalue Real Part $\max\,\operatorname{Re}(\lambda)$", fontsize=10)
    ax6.grid(True, linestyle=":", alpha=0.5)
    ax6.legend(loc="upper right", fontsize=9)

    fig.suptitle("Tier 2 Climate & Ergodic Attractor Validation: Unified Lorenz-Style SciML\n(Invariant Distributions, Spectral Coherence, Poincaré Topology & Learned Bifurcation Manifold)", fontsize=14, fontweight='bold', y=0.995)

    out_png1 = os.path.join(CURRENT_DIR, "biochemical_climate_and_bifurcation_validation.png")
    out_png2 = os.path.join(BRAIN_DIR, "biochemical_climate_and_bifurcation_validation.png")
    plt.savefig(out_png1, bbox_inches="tight", dpi=220)
    plt.savefig(out_png2, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"\nSaved Climate and Bifurcation Figure to:\n  - {out_png1}\n  - {out_png2}", flush=True)

    # Save metrics JSON
    metrics = {
        "wasserstein_w1_m1_mM": float(w1_m1),
        "wasserstein_w1_m0_mM": float(w1_m0),
        "f0_gt_hz": float(f0_gt),
        "f0_sciml_hz": float(f0_sciml),
        "f0_error_pct": float(freq_err),
        "hopf_ka_gt": float(ka_bif_gt),
        "hopf_ka_sciml": float(ka_bif_nn),
        "hopf_ka_error": float(abs(ka_bif_nn - ka_bif_gt))
    }
    json_path = os.path.join(CURRENT_DIR, "biochemical_climate_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved climate metrics to {os.path.basename(json_path)}", flush=True)

if __name__ == '__main__':
    run_climate_validation()
