"""
======================================================================================
VAN DER POL BIFURCATION HORIZON & TIER-2 CLIMATE INVARIANCE VALIDATION
======================================================================================
1. Dense Grid Evaluation (15 points) across mu in [0.10, 2.50]:
   - Tracks transition from harmonic oscillation (T ~ 2pi) to relaxation oscillation (T ~ 1.61mu)
   - Evaluates true slope s(mu), curvature kappa(mu), and neural capacity saturation
2. 4-Pillar Climate & Ergodic Attractor Invariance:
   - Pillar 1: Invariant Ergodic Measure P(x) & Wasserstein-1 (W1)
   - Pillar 2: Power Spectral Density S(f) & Harmonic Comb Alignment
   - Pillar 3: Poincare Surface of Section & 1D Peak Return Map (X_{n+1} vs X_n)
   - Pillar 4: Neural Jacobian Eigenvalue Continuation across mu in [-0.5, 0.5]
3. Standard Academic White Themes for all PNG Figures
======================================================================================
"""

import os
import sys
import json
import numpy as np
import scipy.stats
import scipy.signal
import torch
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import (
    ProposedFiLMVDPMLP, UnconstrainedBaselineVDPMLP,
    simulate_trajectory, measure_period_and_amplitude,
    analytical_hopf_threshold, rk4_step, DT
)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"

def apply_white_theme(ax):
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
        spine.set_linewidth(1.2)
    ax.tick_params(colors="#1e293b", labelsize=9.5, width=1.1)
    ax.grid(True, linestyle="--", alpha=0.35, color="#64748b")

def run_bifurcation_analysis():
    print("\n" + "=" * 100)
    print("1. EVALUATING DENSE GRID BIFURCATION & RELAXATION MANIFOLD (15 POINTS)")
    print("=" * 100)

    # Load 7 seeds of Model A (From-Scratch)
    models_a = []
    for s in SEEDS:
        ckpt = os.path.join(SCRATCH_DIR, f"vdp_sciml_fromscratch_s{s}.pt")
        m = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        m.eval()
        models_a.append((s, m))

    # Load Model B (Physics-Residual)
    m_b = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=True)
    m_b.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_sciml_residual_s42.pt"), map_location="cpu", weights_only=True))
    m_b.eval()

    # 15 Dense Grid Points in mu in [0.10, 2.50]
    mu_grid = np.linspace(0.10, 2.50, 15)
    bench_data = []

    print(f"Evaluating {len(mu_grid)} slice points across mu in [0.10, 2.50]...")
    for idx, mu in enumerate(mu_grid):
        # Warmup and simulate Ground Truth RK4
        y_w = np.array([2.0, 0.0], dtype=np.float64)
        for _ in range(int(30.0 / DT)):
            y_w = rk4_step(y_w, DT, mu)
        steps = int(40.0 / DT)
        gt_traj = simulate_trajectory(y_w, steps, DT, mu)
        gt_per, gt_amp = measure_period_and_amplitude(gt_traj[:, 0], dt=DT)

        # Model A 7-seed evaluation
        p_t = torch.tensor([[mu]], dtype=torch.float32)
        seed_pers = []
        seed_amps = []
        for s, m in models_a:
            curr = torch.tensor([y_w], dtype=torch.float32)
            pred_x = [y_w[0]]
            with torch.no_grad():
                for _ in range(steps):
                    curr = m.rk4_step_forward(curr, p_t)
                    pred_x.append(curr[0, 0].item())
            per, amp = measure_period_and_amplitude(np.array(pred_x), dt=DT)
            seed_pers.append(per)
            seed_amps.append(amp)

        # Model B evaluation
        curr_b = torch.tensor([y_w], dtype=torch.float32)
        pred_b_x = [y_w[0]]
        with torch.no_grad():
            for _ in range(steps):
                curr_b = m_b.rk4_step_forward(curr_b, p_t)
                pred_b_x.append(curr_b[0, 0].item())
        per_b, amp_b = measure_period_and_amplitude(np.array(pred_b_x), dt=DT)

        mean_per_a = float(np.mean(seed_pers))
        std_per_a = float(np.std(seed_pers))
        delta_per_a = mean_per_a - gt_per
        delta_per_b = per_b - gt_per

        bench_data.append({
            "mu": float(mu),
            "gt_period": float(gt_per),
            "gt_amp": float(gt_amp),
            "model_a_per_mean": mean_per_a,
            "model_a_per_std": std_per_a,
            "model_a_seed_pers": seed_pers,
            "delta_per_a": delta_per_a,
            "model_b_per": float(per_b),
            "delta_per_b": delta_per_b
        })
        print(f"  [{idx+1:2d}/15] mu={mu:.3f} | GT Period: {gt_per:6.3f}s | Model A: {mean_per_a:6.3f}s (Delta: {delta_per_a:+6.3f}s) | Model B: {per_b:6.3f}s (Delta: {delta_per_b:+6.3f}s)")

    # Compute Manifold Derivatives: Slope s(mu) and Curvature kappa(mu)
    mus = np.array([d["mu"] for d in bench_data])
    gt_T = np.array([d["gt_period"] for d in bench_data])
    dmu = mus[1] - mus[0]

    # First and second numerical derivatives
    dydx = np.gradient(gt_T, dmu)
    d2ydx2 = np.gradient(dydx, dmu)
    kappa = np.abs(d2ydx2) / (1.0 + dydx**2)**1.5

    for i, d in enumerate(bench_data):
        d["slope"] = float(dydx[i])
        d["curvature"] = float(kappa[i])

    # Disjoint Regional Comparison
    # Region 1: Weakly nonlinear quasi-harmonic [0.10, 0.90] (mus <= 0.90)
    # Region 2: Strongly nonlinear relaxation transition [1.00, 2.50] (mus >= 1.00)
    reg1 = [d for d in bench_data if d["mu"] <= 0.90]
    reg2 = [d for d in bench_data if d["mu"] >= 1.00]

    rmse1_a = np.sqrt(np.mean([d["delta_per_a"]**2 for d in reg1]))
    rmse2_a = np.sqrt(np.mean([d["delta_per_a"]**2 for d in reg2]))
    rmse1_b = np.sqrt(np.mean([d["delta_per_b"]**2 for d in reg1]))
    rmse2_b = np.sqrt(np.mean([d["delta_per_b"]**2 for d in reg2]))

    print("\n" + "=" * 80)
    print("DISJOINT REGIONAL COMPARISON:")
    print(f"  Region 1 (Quasi-Harmonic mu <= 0.90): Peak Curvature = {max(d['curvature'] for d in reg1):.4f} | Model A RMSE = {rmse1_a:.4f}s | Model B RMSE = {rmse1_b:.4f}s")
    print(f"  Region 2 (Relaxation Transition mu >= 1.00): Mean Slope = {np.mean([d['slope'] for d in reg2]):.4f} | Model A RMSE = {rmse2_a:.4f}s | Model B RMSE = {rmse2_b:.4f}s")
    print("=" * 80)

    # Generate Publication-Grade 3-Panel Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), facecolor="#ffffff", dpi=300)
    
    # Panel A: Phase Portraits at mu = 1.0
    ax = axes[0]
    apply_white_theme(ax)
    y_test = np.array([2.0, 0.0], dtype=np.float64)
    traj_gt = simulate_trajectory(y_test, int(20.0/DT), DT, 1.0)
    ax.plot(traj_gt[-int(10.0/DT):, 0], traj_gt[-int(10.0/DT):, 1], color="#1e3a8a", lw=2.4, label="Ground Truth RK4")
    
    # Model A Seed 42 rollout
    m42 = models_a[0][1]
    curr = torch.tensor([y_test], dtype=torch.float32)
    p1 = torch.tensor([[1.0]], dtype=torch.float32)
    pred_42 = [y_test]
    with torch.no_grad():
        for _ in range(int(20.0/DT)):
            curr = m42.rk4_step_forward(curr, p1)
            pred_42.append(curr[0].cpu().numpy())
    pred_42 = np.array(pred_42)
    ax.plot(pred_42[-int(10.0/DT):, 0], pred_42[-int(10.0/DT):, 1], color="#059669", lw=2.0, ls="--", label="Model A (From-Scratch s=42)")
    ax.plot(0, 0, marker="o", color="#dc2626", markersize=7, label="Unstable Focus (0,0)")
    ax.set_title("A. Limit-Cycle Attractor Phase Portrait ($\mu = 1.0$)", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.set_xlabel("State $x(t)$", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Velocity $v(t) = \dot{x}$", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    # Panel B: Period Manifold & Multi-Seed Overlay
    ax = axes[1]
    apply_white_theme(ax)
    ax.plot(mus, gt_T, color="#1e3a8a", lw=2.5, marker="o", markersize=6, label="Ground Truth RK4 $T(\mu)$")
    for s, m in models_a:
        s_pers = [d["model_a_seed_pers"][SEEDS.index(s)] for d in bench_data]
        ax.plot(mus, s_pers, color="#94a3b8", lw=0.9, alpha=0.6, ls=":")
    means_a = [d["model_a_per_mean"] for d in bench_data]
    stds_a = [d["model_a_per_std"] for d in bench_data]
    ax.errorbar(mus, means_a, yerr=stds_a, fmt="s", color="#059669", markersize=5.5, capsize=4, lw=1.8, label="Model A 7-Seed Consensus")
    pers_b = [d["model_b_per"] for d in bench_data]
    ax.plot(mus, pers_b, color="#7c3aed", lw=2.0, ls="-.", marker="^", markersize=5, label="Model B (Physics-Residual)")
    ax.set_title("B. Limit-Cycle Period Manifold $T(\mu)$ across 7 Seeds", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Damping Parameter $\mu$", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Limit-Cycle Period $T$ (s)", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    # Panel C: Pointwise Deviation vs True Curvature
    ax = axes[2]
    apply_white_theme(ax)
    devs_a = [d["delta_per_a"] for d in bench_data]
    devs_b = [d["delta_per_b"] for d in bench_data]
    ax.plot(mus, devs_a, color="#d97706", lw=2.2, marker="o", markersize=6, label=r"Model A Deviation $\Delta T_{\mathrm{scratch}}(\mu)$")
    ax.plot(mus, devs_b, color="#7c3aed", lw=2.0, marker="^", markersize=5, ls="-.", label=r"Model B Deviation $\Delta T_{\mathrm{residual}}(\mu)$")
    ax.axhline(0.0, color="#64748b", lw=1.2, ls="--")
    
    # Twin axis for curvature
    ax_twin = ax.twinx()
    ax_twin.plot(mus, kappa, color="#db2777", lw=1.8, ls=":", label=r"Manifold Curvature $\kappa(\mu)$")
    ax_twin.set_ylabel(r"Curvature $\kappa(\mu)$", fontsize=10.5, fontweight="bold", color="#db2777")
    ax_twin.tick_params(colors="#db2777")
    
    ax.set_title(r"C. Period Deviation $\Delta T(\mu)$ vs. Manifold Curvature $\kappa$", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Damping Parameter $\mu$", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel(r"Period Deviation $\Delta T$ (s)", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)
    ax_twin.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    plt.tight_layout()
    fig_path = os.path.join(SCRATCH_DIR, "vdp_dense_grid_deviation_mechanism.png")
    plt.savefig(fig_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"\n3-panel bifurcation figure saved to: {fig_path}")

    # Save JSON metrics
    json_path = os.path.join(SCRATCH_DIR, "vdp_bifurcation_analysis.json")
    with open(json_path, "w") as f:
        json.dump({
            "dense_grid": bench_data,
            "regional_summary": {
                "region1_rmse_model_a": float(rmse1_a),
                "region2_rmse_model_a": float(rmse2_a),
                "region1_rmse_model_b": float(rmse1_b),
                "region2_rmse_model_b": float(rmse2_b),
            }
        }, f, indent=2)

    return bench_data


def run_climate_and_ergodic_validation():
    print("\n" + "=" * 100)
    print("2. EXECUTING TIER-2 4-PILLAR CLIMATE & ERGODIC ATTRACTOR INVARIANCE SUITE")
    print("=" * 100)

    # Load Model A (From-Scratch Seed 42), Model B (Physics-Residual Seed 42), and Baseline
    m_a = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
    m_a.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_sciml_fromscratch_s42.pt"), map_location="cpu", weights_only=True))
    m_a.eval()

    m_b = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=True)
    m_b.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_sciml_residual_s42.pt"), map_location="cpu", weights_only=True))
    m_b.eval()

    m_base = UnconstrainedBaselineVDPMLP(hidden_dim=64, dt=DT)
    m_base.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_baseline_s42.pt"), map_location="cpu", weights_only=True))
    m_base.eval()

    mu_test = 1.0
    p_t = torch.tensor([[mu_test]], dtype=torch.float32)
    y0 = np.array([2.0, 0.0], dtype=np.float64)
    steps = 10000  # 500.0s (over 75 complete cycles)

    print("Simulating long-horizon trajectories (10,000 steps = 500.0s)...")
    gt_traj = simulate_trajectory(y0, steps, DT, mu_test)
    gt_x = gt_traj[:, 0]

    def rollout_model(m, is_base=False):
        curr = torch.tensor([y0], dtype=torch.float32)
        x_out = [y0[0]]
        with torch.no_grad():
            for _ in range(steps):
                curr = m.rk4_step_forward(curr, p_t)
                x_out.append(curr[0, 0].item())
        return np.array(x_out)

    pred_a_x = rollout_model(m_a)
    pred_b_x = rollout_model(m_b)
    pred_base_x = rollout_model(m_base)

    # Discard warmup transient (first 2,000 steps)
    asympt_gt = gt_x[2000:]
    asympt_a = pred_a_x[2000:]
    asympt_b = pred_b_x[2000:]
    asympt_base = pred_base_x[2000:]

    # PILLAR 1: Invariant Ergodic Measure & Wasserstein-1 Distance
    w1_a = float(scipy.stats.wasserstein_distance(asympt_gt, asympt_a))
    w1_b = float(scipy.stats.wasserstein_distance(asympt_gt, asympt_b))
    w1_base = float(scipy.stats.wasserstein_distance(asympt_gt, asympt_base))
    print("\nPILLAR 1: INVARIANT ERGODIC STATISTICAL MEASURES")
    print(f"  Ground Truth RK4   : Mean = {np.mean(asympt_gt):.4f} | Std = {np.std(asympt_gt):.4f}")
    print(f"  Model A (From-Scratch): Mean = {np.mean(asympt_a):.4f} | Std = {np.std(asympt_a):.4f} | Wasserstein-1 W1 = {w1_a:.4f}")
    print(f"  Model B (Residual)    : Mean = {np.mean(asympt_b):.4f} | Std = {np.std(asympt_b):.4f} | Wasserstein-1 W1 = {w1_b:.4f}")
    print(f"  Baseline Unconstrained: Mean = {np.mean(asympt_base):.4f} | Std = {np.std(asympt_base):.4f} | Wasserstein-1 W1 = {w1_base:.4f}")

    # PILLAR 2: Spectral Harmonics & Power Spectral Density
    fs = 1.0 / DT
    f_gt, psd_gt = scipy.signal.welch(asympt_gt, fs=fs, nperseg=2048)
    f_a, psd_a = scipy.signal.welch(asympt_a, fs=fs, nperseg=2048)
    f_b, psd_b = scipy.signal.welch(asympt_b, fs=fs, nperseg=2048)
    f_base, psd_base = scipy.signal.welch(asympt_base, fs=fs, nperseg=2048)

    f0_gt = float(f_gt[np.argmax(psd_gt)])
    f0_a = float(f_a[np.argmax(psd_a)])
    f0_b = float(f_b[np.argmax(psd_b)])
    f0_base = float(f_base[np.argmax(psd_base)])
    print("\nPILLAR 2: SPECTRAL HARMONICS & PHASE COHERENCE")
    print(f"  Ground Truth Peak f_0 : {f0_gt:.4f} Hz (T = {1.0/f0_gt:.2f} s)")
    print(f"  Model A Peak f_0      : {f0_a:.4f} Hz (T = {1.0/f0_a:.2f} s, Error = {abs(f0_a-f0_gt)/f0_gt*100:.2f}%)")
    print(f"  Model B Peak f_0      : {f0_b:.4f} Hz (T = {1.0/f0_b:.2f} s, Error = {abs(f0_b-f0_gt)/f0_gt*100:.2f}%)")
    print(f"  Baseline Peak f_0     : {f0_base:.4f} Hz (T = {1.0/f0_base:.2f} s, Error = {abs(f0_base-f0_gt)/f0_gt*100:.2f}%)")

    # PILLAR 3: Poincare Return Map (Transversal section: peak x return map)
    def extract_peaks(sig):
        pks, _ = scipy.signal.find_peaks(sig, distance=int(3.0/DT), prominence=0.5)
        return sig[pks]

    pks_gt = extract_peaks(asympt_gt)
    pks_a = extract_peaks(asympt_a)
    pks_b = extract_peaks(asympt_b)
    pks_base = extract_peaks(asympt_base)
    print("\nPILLAR 3: POINCARE RETURN MAP DYNAMICS")
    print(f"  Ground Truth Fixed Point X* : {np.mean(pks_gt):.4f} (Var = {np.var(pks_gt):.6f})")
    print(f"  Model A Fixed Point X*      : {np.mean(pks_a):.4f} (Var = {np.var(pks_a):.6f})")
    print(f"  Model B Fixed Point X*      : {np.mean(pks_b):.4f} (Var = {np.var(pks_b):.6f})")
    print(f"  Baseline Fixed Point X*     : {np.mean(pks_base):.4f} (Var = {np.var(pks_base):.6f})")

    # PILLAR 4: Neural Jacobian Continuation across mu in [-0.5, 0.5]
    print("\nPILLAR 4: NEURAL JACOBIAN CONTINUATION ACROSS mu IN [-0.5, 0.5]")
    mu_sweep = np.linspace(-0.5, 0.5, 21)
    real_true = []
    real_a = []
    real_b = []
    real_base = []

    for mu_val in mu_sweep:
        # Analytical Jacobian at (0, 0): Tr(J) = mu => Re(lambda) = mu / 2
        real_true.append(mu_val / 2.0)

        # Finite-difference Jacobian of neural networks at (0, 0)
        h_eps = 1e-4
        pt = torch.tensor([[mu_val]], dtype=torch.float32)

        def get_neural_jacobian_real_part(model):
            # Evaluate J_theta at origin
            s0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
            f0 = model.predict_derivatives(s0, pt)[0].detach().cpu().numpy()

            # perturb x
            sx = torch.tensor([[h_eps, 0.0]], dtype=torch.float32)
            fx = model.predict_derivatives(sx, pt)[0].detach().cpu().numpy()
            col1 = (fx - f0) / h_eps

            # perturb v
            sv = torch.tensor([[0.0, h_eps]], dtype=torch.float32)
            fv = model.predict_derivatives(sv, pt)[0].detach().cpu().numpy()
            col2 = (fv - f0) / h_eps

            J_num = np.column_stack([col1, col2])
            eigs = np.linalg.eigvals(J_num)
            return float(np.max(np.real(eigs)))

        real_a.append(get_neural_jacobian_real_part(m_a))
        real_b.append(get_neural_jacobian_real_part(m_b))

        # Baseline Jacobian
        def get_base_jacobian(model):
            s0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
            f0 = (model.forward(s0, pt) - s0)[0].detach().cpu().numpy() / DT
            sx = torch.tensor([[h_eps, 0.0]], dtype=torch.float32)
            fx = (model.forward(sx, pt) - sx)[0].detach().cpu().numpy() / DT
            col1 = (fx - f0) / h_eps
            sv = torch.tensor([[0.0, h_eps]], dtype=torch.float32)
            fv = (model.forward(sv, pt) - sv)[0].detach().cpu().numpy() / DT
            col2 = (fv - f0) / h_eps
            J_num = np.column_stack([col1, col2])
            eigs = np.linalg.eigvals(J_num)
            return float(np.max(np.real(eigs)))
        real_base.append(get_base_jacobian(m_base))

    # Detect zero crossings
    idx_a = np.where(np.diff(np.sign(real_a)))[0]
    mu_hopf_a = float(mu_sweep[idx_a[0]]) if len(idx_a) > 0 else np.nan
    idx_b = np.where(np.diff(np.sign(real_b)))[0]
    mu_hopf_b = float(mu_sweep[idx_b[0]]) if len(idx_b) > 0 else np.nan
    print(f"  Analytical Hopf Threshold (mu_c = 0): mu = 0.0000")
    print(f"  Model A Neural Jacobian Crossing    : mu = {mu_hopf_a:.4f} (Discrepancy = {abs(mu_hopf_a):.4f})")
    print(f"  Model B Neural Jacobian Crossing    : mu = {mu_hopf_b:.4f} (Discrepancy = {abs(mu_hopf_b):.4f})")

    # 4-Pillar Climate Figure (Standard Academic White Theme)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor="#ffffff", dpi=300)
    text_dark = "#0f172a"

    # Panel 1: Invariant Ergodic Density P(x)
    ax1 = axes[0, 0]
    apply_white_theme(ax1)
    ax1.hist(asympt_gt, bins=50, density=True, alpha=0.35, color="#1e3a8a", label="Ground Truth RK4")
    ax1.hist(asympt_a, bins=50, density=True, histtype="step", color="#059669", lw=2.2, label=f"Model A (From-Scratch, $W_1={w1_a:.4f}$)")
    ax1.hist(asympt_b, bins=50, density=True, histtype="step", color="#7c3aed", lw=2.0, ls="-.", label=f"Model B (Residual, $W_1={w1_b:.4f}$)")
    ax1.hist(asympt_base, bins=50, density=True, histtype="step", color="#dc2626", lw=1.6, ls=":", label=f"Baseline Generic ($W_1={w1_base:.4f}$)")
    ax1.set_xlabel("State $x(t)$", fontsize=11, fontweight="bold", color=text_dark)
    ax1.set_ylabel("Ergodic Probability Density $P(x)$", fontsize=11, fontweight="bold", color=text_dark)
    ax1.set_title("1. Invariant Ergodic Statistical Density & Wasserstein-1", fontsize=12, fontweight="bold", color=text_dark)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.8)

    # Panel 2: Power Spectral Density S(f)
    ax2 = axes[0, 1]
    apply_white_theme(ax2)
    ax2.semilogy(f_gt, psd_gt, color="#1e3a8a", lw=2.2, label="Ground Truth RK4")
    ax2.semilogy(f_a, psd_a, color="#059669", lw=1.8, ls="--", label="Model A (From-Scratch)")
    ax2.semilogy(f_b, psd_b, color="#7c3aed", lw=1.8, ls="-.", label="Model B (Residual)")
    ax2.semilogy(f_base, psd_base, color="#dc2626", lw=1.4, ls=":", label="Baseline Generic")
    ax2.set_xlim(0.0, 1.5)
    ax2.set_xlabel("Frequency $f$ (Hz)", fontsize=11, fontweight="bold", color=text_dark)
    ax2.set_ylabel("Power Spectral Density $S(f)$", fontsize=11, fontweight="bold", color=text_dark)
    ax2.set_title("2. Spectral Frequency & Harmonic Comb Alignment", fontsize=12, fontweight="bold", color=text_dark)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.8)

    # Panel 3: Poincare Return Map
    ax3 = axes[1, 0]
    apply_white_theme(ax3)
    def plot_return_map(pks, color, marker, label):
        if len(pks) > 2:
            ax3.scatter(pks[:-1], pks[1:], color=color, marker=marker, s=35, alpha=0.85, label=label)
    plot_return_map(pks_gt, "#1e3a8a", "o", "Ground Truth RK4")
    plot_return_map(pks_a, "#059669", "s", "Model A (From-Scratch)")
    plot_return_map(pks_b, "#7c3aed", "^", "Model B (Residual)")
    plot_return_map(pks_base, "#dc2626", "x", "Baseline Generic")
    lims = [1.5, 2.5]
    ax3.plot(lims, lims, color="#94a3b8", lw=1.2, ls="--", label="Identity $X_{n+1} = X_n$")
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_xlabel(r"Return Peak $X_n$", fontsize=11, fontweight="bold", color=text_dark)
    ax3.set_ylabel(r"Next Peak $X_{n+1}$", fontsize=11, fontweight="bold", color=text_dark)
    ax3.set_title(r"3. Poincaré Surface of Section Peak Return Map ($X_{n+1}$ vs $X_n$)", fontsize=12, fontweight="bold", color=text_dark)
    ax3.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.8)

    # Panel 4: Neural Jacobian Continuation across mu
    ax4 = axes[1, 1]
    apply_white_theme(ax4)
    ax4.axhline(0.0, color="#64748b", lw=1.2, ls="--")
    ax4.axvline(0.0, color="#94a3b8", lw=1.2, ls=":", label="Analytical Hopf ($\mu_c = 0.0$)")
    ax4.plot(mu_sweep, real_true, color="#1e3a8a", lw=2.4, marker="o", markersize=5, label=r"True Physical $\mathrm{Re}(\lambda) = \mu/2$")
    ax4.plot(mu_sweep, real_a, color="#059669", lw=2.0, marker="s", markersize=5, ls="--", label=r"Model A Learned $\mathrm{Re}(\lambda_\theta)$")
    ax4.plot(mu_sweep, real_b, color="#7c3aed", lw=2.0, marker="^", markersize=5, ls="-.", label=r"Model B Residual $\mathrm{Re}(\lambda_\theta)$")
    ax4.plot(mu_sweep, real_base, color="#dc2626", lw=1.6, marker="x", markersize=5, ls=":", label=r"Baseline Generic $\mathrm{Re}(\lambda)$")
    ax4.set_xlabel(r"Damping Parameter $\mu$", fontsize=11, fontweight="bold", color=text_dark)
    ax4.set_ylabel(r"Leading Jacobian Real Part $\mathrm{Re}(\lambda)$", fontsize=11, fontweight="bold", color=text_dark)
    ax4.set_title(r"4. Neural Jacobian Hopf Bifurcation Continuation across $\mu$", fontsize=12, fontweight="bold", color=text_dark)
    ax4.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.8)

    plt.tight_layout()
    climate_fig_path = os.path.join(SCRATCH_DIR, "vdp_climate_ergodic_validation.png")
    plt.savefig(climate_fig_path, dpi=300, facecolor="#ffffff")
    plt.close()
    print(f"4-pillar climate figure saved to: {climate_fig_path}")

    # Save climate JSON
    climate_json = os.path.join(SCRATCH_DIR, "vdp_climate_metrics.json")
    with open(climate_json, "w") as f:
        json.dump({
            "wasserstein_1": {"model_a": w1_a, "model_b": w1_b, "baseline": w1_base},
            "peak_frequency": {"gt": f0_gt, "model_a": f0_a, "model_b": f0_b, "baseline": f0_base},
            "poincare_fixed_point": {"gt": float(np.mean(pks_gt)), "model_a": float(np.mean(pks_a)), "model_b": float(np.mean(pks_b)), "baseline": float(np.mean(pks_base))},
            "hopf_crossing": {"analytical": 0.0, "model_a": mu_hopf_a, "model_b": mu_hopf_b}
        }, f, indent=2)

if __name__ == "__main__":
    run_bifurcation_analysis()
    run_climate_and_ergodic_validation()
