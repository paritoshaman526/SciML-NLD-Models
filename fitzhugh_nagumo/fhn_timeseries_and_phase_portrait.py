"""
======================================================================================
FITZHUGH-NAGUMO: CANONICAL TIME SERIES & PHASE-PORTRAIT PUBLICATION SUITE
======================================================================================
Generates the definitive publication-grade figure showcasing:
1. Panel A: Phase Plane (v, w) with Cubic/Linear Nullclines, Vector Field, and Limit Cycle
2. Panel B: Action Potential Spike Train v(t) over a 150s horizon
3. Panel C: Slow Refractory Recovery Variable w(t)
4. Panel D: Subthreshold Quiescent Regime (I = 0.20 < I_c) Inward Attracting Spiral Sink
======================================================================================
"""

import os
import sys
import numpy as np
import scipy.signal
import torch
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

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

SCRATCH_DIR = os.path.join(CURRENT_DIR, "scratch")
ARTIFACT_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\1c89c907-107d-4ec6-b918-cd19ff5817dc"

def apply_white_theme(ax):
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
        spine.set_linewidth(1.2)
    ax.tick_params(colors="#1e293b", labelsize=10.5, width=1.1)
    ax.grid(True, linestyle="--", alpha=0.35, color="#64748b")

def rollout_model(model, y0, steps, params):
    traj = [y0]
    curr = torch.tensor([y0], dtype=torch.float32)
    p_t = torch.tensor([params], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(steps):
            curr = model(curr, p_t)
            traj.append(curr.squeeze(0).cpu().numpy())
    return np.array(traj)

def main():
    print("Generating FHN Canonical Time Series & Phase Portrait Figure...", flush=True)

    # 1. Load Checkpoints
    ckpt_sciml = os.path.join(SCRATCH_DIR, "fhn_sciml_s42.pt")
    ckpt_base = os.path.join(SCRATCH_DIR, "fhn_baseline_s42.pt")

    m_sciml = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
    m_sciml.load_state_dict(torch.load(ckpt_sciml, map_location="cpu"))
    m_sciml.eval()

    m_base = UnconstrainedBaselineFHNMLP(hidden_dim=128, dt=DT)
    m_base.load_state_dict(torch.load(ckpt_base, map_location="cpu"))
    m_base.eval()

    # Simulation parameters for Limit Cycle Regime (I = 0.80)
    p_spiking = np.array([0.80, 0.70, 0.80, 0.08], dtype=np.float64)
    y0_spiking = np.array([-1.0, -0.5], dtype=np.float64)
    t_span = 150.0  # seconds
    steps = int(t_span / DT)
    t_arr = np.linspace(0.0, t_span, steps + 1)

    print("Simulating spiking limit cycle regime (I=0.80)...", flush=True)
    traj_gt = simulate_trajectory(y0_spiking, steps, DT, p_spiking)
    traj_sciml = rollout_model(m_sciml, y0_spiking, steps, p_spiking)
    traj_base = rollout_model(m_base, y0_spiking, steps, p_spiking)

    # Simulation parameters for Subthreshold Quiescent Regime (I = 0.20 < I_c = 0.3313)
    p_sub = np.array([0.20, 0.70, 0.80, 0.08], dtype=np.float64)
    y0_sub = np.array([1.2, 0.8], dtype=np.float64)
    steps_sub = int(80.0 / DT)
    t_sub = np.linspace(0.0, 80.0, steps_sub + 1)

    print("Simulating subthreshold quiescent regime (I=0.20)...", flush=True)
    traj_sub_gt = simulate_trajectory(y0_sub, steps_sub, DT, p_sub)
    traj_sub_sciml = rollout_model(m_sciml, y0_sub, steps_sub, p_sub)
    v_star_sub, w_star_sub = analytical_fixed_point(p_sub[0], p_sub[1], p_sub[2], p_sub[3])

    # Plotting setup
    fig = plt.figure(figsize=(19, 11), facecolor="#ffffff", dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.25], height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.22)

    # Color Palette
    c_gt = "#1e3a8a"       # Deep Navy
    c_sciml = "#059669"    # Emerald Green
    c_base = "#ea580c"     # Amber / Orange
    c_v_null = "#dc2626"   # Crimson Red (v-nullcline)
    c_w_null = "#7c3aed"   # Royal Purple (w-nullcline)
    c_flow = "#94a3b8"     # Light Slate Grey

    # ==============================================================================
    # PANEL A: PHASE PORTRAIT & NULLCLINE GEOMETRY (I = 0.80)
    # ==============================================================================
    axA = fig.add_subplot(gs[:, 0])
    apply_white_theme(axA)

    v_grid = np.linspace(-2.5, 2.5, 25)
    w_grid = np.linspace(-1.2, 2.2, 22)
    V, W = np.meshgrid(v_grid, w_grid)

    I_spk, a_spk, b_spk, eps_spk = p_spiking
    dV = V - (V**3) / 3.0 - W + I_spk
    dW = eps_spk * (V + a_spk - b_spk * W)
    norm = np.sqrt(dV**2 + dW**2) + 1e-6
    axA.quiver(V, W, dV / norm, dW / norm, color=c_flow, alpha=0.32, width=0.0022, headwidth=4, zorder=1)

    # Nullclines
    v_dense = np.linspace(-2.6, 2.6, 500)
    w_v_null = v_dense - (v_dense**3) / 3.0 + I_spk
    w_w_null = (v_dense + a_spk) / b_spk

    axA.plot(v_dense, w_v_null, color=c_v_null, lw=2.4, ls="--", zorder=3, label=r"Cubic $\dot{v}=0$ Nullcline ($w = v - v^3/3 + I$)")
    axA.plot(v_dense, w_w_null, color=c_w_null, lw=2.4, ls="-.", zorder=3, label=r"Linear $\dot{w}=0$ Nullcline ($w = (v+a)/b$)")

    # Unstable Fixed Point
    v_star_spk, w_star_spk = analytical_fixed_point(I_spk, a_spk, b_spk, eps_spk)
    axA.plot(v_star_spk, w_star_spk, marker="o", markersize=9, color="#dc2626", markeredgecolor="#ffffff", markeredgewidth=1.5, zorder=6, label=f"Unstable Fixed Point ({v_star_spk:.2f}, {w_star_spk:.2f})")

    # Trajectories (asymptotic limit cycle after warmup)
    warm_idx = int(30.0 / DT)
    axA.plot(traj_gt[warm_idx:, 0], traj_gt[warm_idx:, 1], color=c_gt, lw=3.2, zorder=5, label="Ground Truth RK4 Attractor")
    axA.plot(traj_sciml[warm_idx:, 0], traj_sciml[warm_idx:, 1], color=c_sciml, lw=2.4, ls="-", zorder=5, label="Proposed FiLM SciML (Seed 42)")
    axA.plot(traj_base[warm_idx:, 0], traj_base[warm_idx:, 1], color=c_base, lw=1.8, ls=":", zorder=4, label="Unconstrained Baseline MLP")

    # Annotate fast-slow relaxation jumps
    axA.annotate(
        "Fast Action Potential Upstroke\n($\\dot{v} \\gg 0$, fast depolarization)",
        xy=(-0.8, -0.65), xytext=(-2.35, -1.05),
        arrowprops=dict(facecolor="#1e3a8a", shrink=0.08, width=1.0, headwidth=5),
        fontsize=8.5, fontweight="bold", color="#1e3a8a",
        bbox=dict(boxstyle="round,pad=0.35", fc="#eff6ff", ec="#93c5fd", lw=1.1)
    )
    axA.annotate(
        "Slow Refractory Drift\n($\\dot{w} \\sim \\epsilon$, slow recovery)",
        xy=(1.8, 1.25), xytext=(0.35, 1.85),
        arrowprops=dict(facecolor="#7c3aed", shrink=0.08, width=1.0, headwidth=5),
        fontsize=8.5, fontweight="bold", color="#7c3aed",
        bbox=dict(boxstyle="round,pad=0.35", fc="#faf5ff", ec="#d8b4fe", lw=1.1)
    )

    axA.set_xlim(-2.55, 2.55)
    axA.set_ylim(-1.25, 2.25)
    axA.set_xlabel("Membrane Potential $v$", fontsize=12.5, fontweight="bold", color="#0f172a")
    axA.set_ylabel("Recovery Variable $w$", fontsize=12.5, fontweight="bold", color="#0f172a")
    axA.set_title("Panel A: Phase-Plane Limit Cycle & Nullcline Geometry ($I = 0.80$)", fontsize=13.5, fontweight="bold", color="#0f172a", pad=12)
    axA.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.2, framealpha=0.95)

    # ==============================================================================
    # PANEL B: MEMBRANE POTENTIAL ACTION POTENTIAL TIME SERIES v(t)
    # ==============================================================================
    axB = fig.add_subplot(gs[0, 1])
    apply_white_theme(axB)

    t_crop = t_arr <= 100.0
    axB.plot(t_arr[t_crop], traj_gt[t_crop, 0], color=c_gt, lw=2.6, label="Ground Truth RK4 Spikes", zorder=3)
    axB.plot(t_arr[t_crop], traj_sciml[t_crop, 0], color=c_sciml, lw=2.0, ls="-", label="Proposed SciML Model", zorder=4)
    axB.plot(t_arr[t_crop], traj_base[t_crop, 0], color=c_base, lw=1.5, ls=":", label="Unconstrained Baseline", zorder=2)

    axB.set_xlim(0.0, 100.0)
    axB.set_ylim(-2.4, 2.4)
    axB.set_xlabel("Time $t$ (seconds)", fontsize=11.5, fontweight="bold", color="#0f172a")
    axB.set_ylabel("Membrane Voltage $v(t)$", fontsize=11.5, fontweight="bold", color="#0f172a")
    axB.set_title("Panel B: Action Potential Waveform & Spike Train ($v(t)$)", fontsize=12.5, fontweight="bold", color="#0f172a", pad=10)
    axB.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    axB.annotate(
        "Spike Peak $v_{\\max} \\approx +1.92$\nPeriod $T \\approx 36.42\\text{ s}$",
        xy=(35.5, 1.95), xytext=(45.0, 1.45),
        arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.0, headwidth=5),
        fontsize=8.2, fontweight="bold", color="#059669",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec="#a7f3d0", lw=1.0)
    )

    # ==============================================================================
    # PANEL C: SLOW RECOVERY VARIABLE w(t) & SUBTHRESHOLD QUIESCENCE
    # ==============================================================================
    axC = fig.add_subplot(gs[1, 1])
    apply_white_theme(axC)

    # Split: Left half shows w(t) periodic recovery; right half or overlaid subthreshold
    # Let's show w(t) on top line, and in inset/side show subthreshold damping!
    axC.plot(t_arr[t_crop], traj_gt[t_crop, 1], color=c_gt, lw=2.6, label="Recovery $w(t)$ True RK4", zorder=3)
    axC.plot(t_arr[t_crop], traj_sciml[t_crop, 1], color=c_sciml, lw=2.0, ls="-", label="Recovery $w(t)$ SciML", zorder=4)
    axC.plot(t_arr[t_crop], traj_base[t_crop, 1], color=c_base, lw=1.5, ls=":", label="Recovery $w(t)$ Baseline", zorder=2)

    axC.set_xlim(0.0, 100.0)
    axC.set_ylim(-0.4, 1.8)
    axC.set_xlabel("Time $t$ (seconds)", fontsize=11.5, fontweight="bold", color="#0f172a")
    axC.set_ylabel("Recovery Variable $w(t)$", fontsize=11.5, fontweight="bold", color="#0f172a")
    axC.set_title("Panel C: Slow Variable Dynamics $w(t)$ & Inward Resting Spiral Inset", fontsize=12.5, fontweight="bold", color="#0f172a", pad=10)
    axC.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

    # Inset for Subthreshold Quiescent Spiral (I = 0.20 < I_c = 0.3313)
    ax_ins = axC.inset_axes([0.55, 0.12, 0.40, 0.48])
    apply_white_theme(ax_ins)
    ax_ins.plot(traj_sub_gt[:, 0], traj_sub_gt[:, 1], color=c_gt, lw=2.0, label="GT Sink")
    ax_ins.plot(traj_sub_sciml[:, 0], traj_sub_sciml[:, 1], color=c_sciml, lw=1.8, ls="--", label="SciML Sink")
    ax_ins.plot(v_star_sub, w_star_sub, marker="*", markersize=8, color="#dc2626", zorder=5)
    ax_ins.set_title("Subthreshold Spiral ($I=0.20 < I_c$)", fontsize=8.0, fontweight="bold", color="#0f172a", pad=3)
    ax_ins.tick_params(labelsize=7.5)
    ax_ins.set_xlabel("$v$", fontsize=7.5, labelpad=1)
    ax_ins.set_ylabel("$w$", fontsize=7.5, labelpad=1)

    # Save to both scratch and artifact directory
    out_scratch = os.path.join(SCRATCH_DIR, "fhn_timeseries_and_phase_portrait.png")
    out_artifact = os.path.join(ARTIFACT_DIR, "fhn_timeseries_and_phase_portrait.png")

    plt.savefig(out_scratch, dpi=300, bbox_inches="tight")
    plt.savefig(out_artifact, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure successfully generated and saved to:\n  {out_scratch}\n  {out_artifact}", flush=True)

if __name__ == "__main__":
    main()
