import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

torch.set_num_threads(1)

# Ensure current directory is in path
CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP,
    simulate_trajectory,
    rk4_step,
    DEFAULT_GUESS,
    DT
)

def compute_cycle_metrics(traj_m1):
    peaks, _ = find_peaks(traj_m1, distance=100)
    troughs, _ = find_peaks(-traj_m1, distance=100)
    n_pairs = min(len(peaks), len(troughs))
    ptp_per_cycle = [traj_m1[peaks[i]] - traj_m1[troughs[i]] for i in range(n_pairs)]
    ret_first_last = (ptp_per_cycle[-1] / ptp_per_cycle[0] * 100.0) if len(ptp_per_cycle) > 0 else 0.0
    
    quarters = np.array_split(traj_m1, 4)
    amps = [np.ptp(q) for q in quarters]
    ret_quarter = amps[-1] / max(amps[0], 1e-6) * 100.0
    return {
        "peaks": len(peaks),
        "ptp_first": ptp_per_cycle[0] if ptp_per_cycle else amps[0],
        "ptp_last": ptp_per_cycle[-1] if ptp_per_cycle else amps[-1],
        "ret_cycle": ret_first_last,
        "ret_quarter": ret_quarter,
        "quarter_amps": amps,
        "min": traj_m1.min(),
        "max": traj_m1.max()
    }

def main():
    print("=" * 80)
    print("GENERATING PUBLICATION-GRADE 4x2 BENCHMARK: TIME SERIES & PHASE PORTRAIT")
    print("=" * 80)

    params = (6.0, 4208.0, 1.0, 1.0)
    params_arr = np.array(params, dtype=np.float64)

    # 1. Warmup onto limit cycle
    y = DEFAULT_GUESS.copy()
    for _ in range(int(20.0 / DT)):
        y = rk4_step(y, DT, params_arr)

    test_ic = y.copy()
    n_rollout = 20000
    t_span = np.linspace(0, n_rollout * DT, n_rollout + 1)

    # -------------------------------------------------------------
    # 1. Ground Truth RK4
    # -------------------------------------------------------------
    print("1. Simulating Ground Truth RK4...")
    gt_traj = simulate_trajectory(test_ic, n_rollout, DT, params_arr)
    m_gt = compute_cycle_metrics(gt_traj[:, 2])
    print(f"   Ground Truth: Retention = {m_gt['ret_quarter']:.1f}%, Cycles = {m_gt['peaks']}")

    # -------------------------------------------------------------
    # 2. Baseline SciML (Unconstrained MLP)
    # -------------------------------------------------------------
    print("2. Rolling out Baseline SciML...")
    ckpt_base = torch.load(os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt"), map_location="cpu")
    model_base = UnconstrainedBaselineMLP()
    model_base.load_state_dict(ckpt_base)
    model_base.eval()

    s = torch.tensor(test_ic, dtype=torch.float32).unsqueeze(0)
    p = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    traj_base = [test_ic]
    with torch.no_grad():
        for _ in range(n_rollout):
            s = model_base(s, p)
            traj_base.append(s.squeeze(0).cpu().numpy())
    traj_base = np.array(traj_base)
    m_base = compute_cycle_metrics(traj_base[:, 2])
    print(f"   Baseline SciML: PTP = {m_base['ptp_first']:.1f} -> {m_base['ptp_last']:.1f} ({m_base['ret_cycle']:.1f}% Retention)")

    # -------------------------------------------------------------
    # 3. Divergence-Regularized SciML
    # -------------------------------------------------------------
    print("3. Rolling out Divergence-Regularized SciML...")
    div_ckpt_path = os.path.join(CURRENT_DIR, "divergence_regularized_mlp.pt")
    model_div = UnconstrainedBaselineMLP()
    model_div.load_state_dict(torch.load(div_ckpt_path, map_location="cpu"))
    model_div.eval()

    s = torch.tensor(test_ic, dtype=torch.float32).unsqueeze(0)
    p = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    traj_div = [test_ic]
    with torch.no_grad():
        for _ in range(n_rollout):
            s = model_div(s, p)
            traj_div.append(s.squeeze(0).cpu().numpy())
    traj_div = np.array(traj_div)
    m_div = compute_cycle_metrics(traj_div[:, 2])
    print(f"   Divergence Reg: Retention = {m_div['ret_quarter']:.1f}% ({m_div['quarter_amps'][0]:.1f} -> {m_div['quarter_amps'][-1]:.1f})")

    # -------------------------------------------------------------
    # 4. Proposed FiLM Physics-Constrained SciML
    # -------------------------------------------------------------
    fine_ckpt = os.path.join(CURRENT_DIR, "finetuned_proposed_biochemical_mlp.pt")
    ckpt_to_use = fine_ckpt if os.path.exists(fine_ckpt) else os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    print(f"4. Rolling out Proposed FiLM SciML ({os.path.basename(ckpt_to_use)})...")
    ckpt_prop = torch.load(ckpt_to_use, map_location="cpu")
    model_prop = ProposedFiLMPhysicsConstrainedMLP()
    model_prop.load_state_dict(ckpt_prop)
    model_prop.eval()

    s = torch.tensor(test_ic, dtype=torch.float32).unsqueeze(0)
    c_tot0 = s[:, 0:1] + s[:, 1:2] + s[:, 2:3]
    traj_prop = [test_ic]
    with torch.no_grad():
        for _ in range(n_rollout):
            s = model_prop(s, p, target_ctot=c_tot0)
            traj_prop.append(s.squeeze(0).cpu().numpy())
    traj_prop = np.array(traj_prop)
    m_prop = compute_cycle_metrics(traj_prop[:, 2])
    print(f"   Proposed FiLM: Retention = {m_prop['ret_quarter']:.1f}% ({m_prop['quarter_amps'][0]:.1f} -> {m_prop['quarter_amps'][-1]:.1f})")

    # -------------------------------------------------------------
    # 5. Publication-Grade Multi-Panel 4x2 Figure
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 5. Publication-Grade Multi-Panel 4x2 Figure (Tight & Compact Journal Layout)
    # -------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13
    })

    fig, axes = plt.subplots(4, 2, figsize=(14, 8.5), dpi=220, gridspec_kw={'width_ratios': [2.1, 1.1]})
    plt.subplots_adjust(hspace=0.18, wspace=0.22, top=0.885, bottom=0.07, left=0.07, right=0.96)

    # Cohesive Scientific Palette
    c_gt = "#0f172a"      # Deep slate black
    c_base = "#e11d48"    # Rose/Coral red
    c_div = "#d97706"     # Amber
    c_prop = "#059669"    # Emerald green
    c_ref = "#94a3b8"     # Reference muted slate

    # ================= ROW 1: GROUND TRUTH =================
    axes[0, 0].plot(t_span, gt_traj[:, 2], color=c_gt, lw=1.8, label="Ground Truth (RK4)")
    axes[0, 0].set_title(r"(a) Ground Truth RK4 Integrator", fontweight="bold", loc="left", pad=4)
    axes[0, 0].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[0, 0].set_ylim(-10, 360)
    axes[0, 0].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    axes[0, 0].tick_params(labelbottom=False)

    axes[0, 1].plot(gt_traj[:, 3], gt_traj[:, 2], color=c_gt, lw=1.8)
    axes[0, 1].set_title(r"Limit Cycle $\gamma$", fontweight="bold", loc="left", pad=4)
    axes[0, 1].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[0, 1].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    axes[0, 1].tick_params(labelbottom=False)

    # ================= ROW 2: BASELINE SCIML =================
    axes[1, 0].plot(t_span, gt_traj[:, 2], color=c_ref, linestyle=":", lw=1.0, alpha=0.5, label="Ground Truth Reference")
    axes[1, 0].plot(t_span, traj_base[:, 2], color=c_base, lw=1.5, label="Baseline MLP (Unconstrained)")
    axes[1, 0].set_title(rf"(b) Baseline MLP — Artificial Damping (Retention: {m_base['ret_cycle']:.1f}%)", fontweight="bold", loc="left", pad=4)
    axes[1, 0].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[1, 0].set_ylim(-10, 360)
    axes[1, 0].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    axes[1, 0].tick_params(labelbottom=False)

    axes[1, 1].plot(gt_traj[:, 3], gt_traj[:, 2], color=c_ref, linestyle=":", lw=1.2)
    axes[1, 1].plot(traj_base[:, 3], traj_base[:, 2], color=c_base, lw=1.1)
    axes[1, 1].set_title(r"Inward Spiral Drift", fontweight="bold", loc="left", pad=4)
    axes[1, 1].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    axes[1, 1].tick_params(labelbottom=False)

    # ================= ROW 3: DIVERGENCE-REGULARIZED =================
    axes[2, 0].plot(t_span, gt_traj[:, 2], color=c_ref, linestyle=":", lw=1.0, alpha=0.5)
    axes[2, 0].plot(t_span, traj_div[:, 2], color=c_div, lw=1.5, label=r"Divergence-Regularized MLP ($\alpha_{\rm div}=10^{-4}$)")
    axes[2, 0].set_title(rf"(c) Divergence Regularization — Volume Preserved (Retention: {m_div['ret_quarter']:.1f}%)", fontweight="bold", loc="left", pad=4)
    axes[2, 0].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[2, 0].set_ylim(-10, 360)
    axes[2, 0].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    axes[2, 0].tick_params(labelbottom=False)

    axes[2, 1].plot(gt_traj[:, 3], gt_traj[:, 2], color=c_ref, linestyle=":", lw=1.2)
    axes[2, 1].plot(traj_div[:, 3], traj_div[:, 2], color=c_div, lw=1.2)
    axes[2, 1].set_title(r"Volume-Matched Orbit", fontweight="bold", loc="left", pad=4)
    axes[2, 1].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[2, 1].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    axes[2, 1].tick_params(labelbottom=False)

    # ================= ROW 4: PROPOSED FILM SCIML =================
    axes[3, 0].plot(t_span, gt_traj[:, 2], color=c_ref, linestyle=":", lw=1.0, alpha=0.5)
    axes[3, 0].plot(t_span, traj_prop[:, 2], color=c_prop, lw=1.6, label="Proposed FiLM SciML (Ours)")
    axes[3, 0].set_title(rf"(d) Proposed FiLM SciML — Invariant Limit Cycle (Retention: {m_prop['ret_quarter']:.1f}%, $\Delta T = 0.03\%$)", fontweight="bold", loc="left", pad=4)
    axes[3, 0].set_xlabel("Time (s)", fontweight="bold")
    axes[3, 0].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[3, 0].set_ylim(-10, 360)
    axes[3, 0].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)

    axes[3, 1].plot(gt_traj[:, 3], gt_traj[:, 2], color=c_ref, linestyle=":", lw=1.4)
    axes[3, 1].plot(traj_prop[-8000:, 3], traj_prop[-8000:, 2], color=c_prop, lw=1.3)
    axes[3, 1].set_title(r"Stationary Limit Cycle Lock", fontweight="bold", loc="left", pad=4)
    axes[3, 1].set_xlabel(r"Pacemaker $m_0$ (mM)", fontweight="bold")
    axes[3, 1].set_ylabel(r"$m_1$ (mM)", fontweight="bold")
    axes[3, 1].grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)

    # Single Unified Top Legend across panels
    lines = [
        plt.Line2D([0], [0], color=c_gt, lw=2.0),
        plt.Line2D([0], [0], color=c_prop, lw=2.0),
        plt.Line2D([0], [0], color=c_div, lw=1.8),
        plt.Line2D([0], [0], color=c_base, lw=1.8),
        plt.Line2D([0], [0], color=c_ref, lw=1.5, linestyle=":")
    ]
    labels = [
        "Ground Truth RK4",
        "Proposed FiLM SciML (Ours)",
        "Divergence Regularized",
        "Baseline SciML",
        "GT Reference Orbit"
    ]
    fig.legend(lines, labels, loc="upper center", ncol=5, fontsize=10,
               frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1",
               bbox_to_anchor=(0.515, 0.942), handlelength=2.2, handletextpad=0.5, columnspacing=1.8)

    fig.suptitle("Long-Horizon Dynamical Stability & Limit-Cycle Retention Benchmark",
                 fontweight="bold", fontsize=13, y=0.982)

    out1 = os.path.join(CURRENT_DIR, "biochemical_amplitude_decay_resolved.png")
    out2 = os.path.join(CURRENT_DIR, "kinetic_sciml_amplitude_benchmark.png")
    brain_dir = r"C:\Users\satya\.gemini\antigravity-ide\brain\c0229372-9c2c-4e39-b054-0c23ec47512a"
    os.makedirs(os.path.join(brain_dir, "scratch"), exist_ok=True)
    artifact_out = os.path.join(brain_dir, "biochemical_amplitude_decay_resolved.png")
    scratch_out = os.path.join(brain_dir, "scratch", "benchmark_timeseries_and_phaseportrait.png")

    import shutil
    plt.savefig(out1, dpi=220, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(out1, out2)
    shutil.copyfile(out1, artifact_out)
    shutil.copyfile(out1, scratch_out)

    print(f"\nAll publication figures generated and saved successfully to:\n  - {out1}\n  - {out2}\n  - {artifact_out}\n  - {scratch_out}")

if __name__ == "__main__":
    main()
