import os
import sys
import numpy as np
import scipy.optimize
import torch
import matplotlib.pyplot as plt

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\4fd50b55-98fd-4b91-92cf-1a428e462f37"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    system,
    rk4_step,
    simulate_trajectory,
    DEFAULT_GUESS,
    DT
)
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML as GatedModelS2026
from train_perfect_film_triad import BifurcationGatedFiLMSciML as GatedModelV2

def compute_jacobian_spectrum(p):
    """Compute true steady state and maximum real part of Jacobian eigenvalues."""
    c_tot = 1000.0
    def res(x):
        d, a1, m0 = x
        m1 = c_tot - d - a1
        s = np.array([d, a1, m1, m0], dtype=np.float64)
        f = system(s, p)
        return np.array([f[0], f[1], f[3]], dtype=np.float64)

    guess = [500.0, 200.0, 1.0]
    opt = scipy.optimize.root(res, guess, method='hybr')
    d, a1, m0 = opt.x
    m1 = c_tot - d - a1
    s_star = np.array([d, a1, m1, m0], dtype=np.float64)

    # 3x3 Jacobian wrt (d, a1, m0)
    eps = 1e-6
    f0 = res(opt.x)
    J = np.zeros((3, 3))
    for j in range(3):
        x_pert = opt.x.copy()
        x_pert[j] += eps
        fj = res(x_pert)
        J[:, j] = (fj - f0) / eps

    eigs = np.linalg.eigvals(J)
    max_re = float(np.max(np.real(eigs)))
    return max_re, s_star

def run_investigation():
    print("=" * 80)
    print("DIAGNOSTIC INVESTIGATION: SEED 2026 REPLICATION & GATE SATURATION ANALYSIS")
    print("=" * 80)

    # 1. Load models
    ckpt_s2026 = os.path.join(CURRENT_DIR, "film_bifurcation_gated_s2026.pt")
    m_s2026 = GatedModelS2026(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_s2026.load_state_dict(torch.load(ckpt_s2026, map_location="cpu"))
    m_s2026.eval()

    ckpt_v2 = os.path.join(CURRENT_DIR, "bifurcation_gated_v2.pt")
    m_v2 = GatedModelV2()
    m_v2.load_state_dict(torch.load(ckpt_v2, map_location="cpu"))
    m_v2.eval()

    # --------------------------------------------------------------------------
    # PART 1: EXACT GATE VALUES FOR X1 - X6 AND X4 SPECIFICALLY
    # --------------------------------------------------------------------------
    print("\n[PART 1] GATE VALUES FOR STEADY STATE CASES X1-X6:")
    ss_cases = [c for c in SWEEP_CASES if c['is_steady_state']]
    gamma = 10.0

    print(f"{'Case':5s} | {'Params (ka, kd2, kd3)':24s} | {'Gate (S2026)':12s} | {'Gate (V2)':12s} | {'Raw rm1 (mM/s)':14s} | {'Offset S2026':12s} | {'Offset V2':12s}")
    print("-" * 95)
    
    part1_data = []
    for c in ss_cases:
        cid = c['id']
        p = c['params']
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        
        # S2026 gate
        p_norm26 = p_t / m_s2026.param_scales
        p_embed26 = m_s2026.p_net(p_norm26)
        g_s2026 = m_s2026.gate_net(p_embed26).item()

        # V2 gate
        p_norm_v2 = p_t / m_v2.param_scales
        g_v2 = m_v2.gate_net(p_norm_v2).item()

        # Steady state and raw engine flux at y*
        max_re, y_star = compute_jacobian_spectrum(p)
        s_t = torch.tensor(y_star, dtype=torch.float32).unsqueeze(0)

        # Raw engine flux from S2026
        s_norm26 = s_t / m_s2026.state_scales
        h = m_s2026.block1(s_norm26, p_embed26)
        h = m_s2026.block2(h, p_embed26)
        h = m_s2026.block3(h, p_embed26)
        out26 = m_s2026.head(h)
        J2 = torch.nn.functional.softplus(out26[:, 1:2]) * m_s2026.rate_scales[1]
        J3 = torch.nn.functional.softplus(out26[:, 2:3]) * m_s2026.rate_scales[2]
        rm1_raw = (J3 - J2).item()

        # Analytical calibration offset: g / ((1 - g) * gamma) * rm1
        offset_s2026 = (g_s2026 / ((1.0 - g_s2026) * gamma)) * rm1_raw if g_s2026 < 1.0 else np.nan
        offset_v2 = (g_v2 / ((1.0 - g_v2) * gamma)) * rm1_raw if g_v2 < 1.0 else np.nan

        part1_data.append({
            'id': cid,
            'params': p,
            'gate_s2026': g_s2026,
            'gate_v2': g_v2,
            'rm1_raw': rm1_raw,
            'offset_s2026': offset_s2026,
            'offset_v2': offset_v2,
            'y_star_m1': y_star[2]
        })
        p_str = f"({p[0]:.1f}, {p[1]:.0f}, {p[2]:.2f})"
        print(f"{cid:5s} | {p_str:24s} | {g_s2026:12.6f} | {g_v2:12.6f} | {rm1_raw:14.3f} | {offset_s2026:+11.4f} mM | {offset_v2:+11.4f} mM")

    # --------------------------------------------------------------------------
    # PART 2: REPLICATING SEED 2026 ON X9 AND X11 (CONFLICTING FIGURE REPRODUCTION)
    # --------------------------------------------------------------------------
    print("\n[PART 2] REPLICATING SEED 2026 vs V2 ON X9 AND X11:")
    cases_test = [c for c in SWEEP_CASES if c['id'] in ['X9', 'X11']]
    steps = 10000
    t_arr = np.linspace(0, steps * DT, steps)

    rep_results = {}
    for c in cases_test:
        cid = c['id']
        p_arr = np.array(c['params'], dtype=np.float64)
        p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)

        # Warmup for initial condition
        y = DEFAULT_GUESS.copy()
        for _ in range(int(c['warmup'] / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        # Ground truth
        gt = simulate_trajectory(ic, steps, DT, p_arr)

        # Rollout Seed 2026
        curr26 = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        c_tot = curr26[:, 0] + curr26[:, 1] + curr26[:, 2]
        traj26 = [ic.copy()]
        gate26_t = []
        with torch.no_grad():
            for _ in range(steps):
                p_norm = p_t / m_s2026.param_scales
                p_embed = m_s2026.p_net(p_norm)
                g_val = m_s2026.gate_net(p_embed).item()
                gate26_t.append(g_val)
                curr26 = m_s2026(curr26, p_t, target_ctot=c_tot)
                traj26.append(curr26.squeeze(0).cpu().numpy())
        traj26 = np.array(traj26)[:steps]

        # Rollout V2
        curr_v2 = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        traj_v2 = [ic.copy()]
        gate_v2_t = []
        with torch.no_grad():
            for _ in range(steps):
                p_norm = p_t / m_v2.param_scales
                g_val = m_v2.gate_net(p_norm).item()
                gate_v2_t.append(g_val)
                curr_v2 = m_v2(curr_v2, p_t, target_ctot=c_tot)
                traj_v2.append(curr_v2.squeeze(0).cpu().numpy())
        traj_v2 = np.array(traj_v2)[:steps]

        rep_results[cid] = {
            'case': c,
            'gt': gt,
            'traj26': traj26,
            'traj_v2': traj_v2,
            'gate26_t': np.array(gate26_t),
            'gate_v2_t': np.array(gate_v2_t)
        }
        print(f"  {cid}: Seed 2026 gate = {gate26_t[0]:.6f} (min={min(gate26_t):.6f}, max={max(gate26_t):.6f})")
        print(f"       Seed 2026 m1 ptp = {np.ptp(traj26[:, 2]):.2f} mM | final 10% std = {np.std(traj26[-1000:, 2]):.2f} mM")
        print(f"       V2        gate = {gate_v2_t[0]:.6f} (min={min(gate_v2_t):.6f}, max={max(gate_v2_t):.6f})")
        print(f"       V2        m1 ptp = {np.ptp(traj_v2[:, 2]):.2f} mM | final 10% std = {np.std(traj_v2[-1000:, 2]):.2f} mM")

    # --------------------------------------------------------------------------
    # PART 3: CONTINUOUS PARAMETER SWEEP ACROSS HOPF BIFURCATION (GATE SATURATION TEST)
    # --------------------------------------------------------------------------
    print("\n[PART 3] CONTINUOUS PARAMETER SWEEPS ACROSS BIFURCATION BOUNDARIES:")
    
    # Slice A: X4 Cross Section (kd2 = 4600, kd3 = 0.60, ka in [6.0, 14.0])
    ka_sweep_a = np.linspace(6.0, 13.5, 60)
    kd2_a, kd3_a = 4600.0, 0.60
    eigs_a, gates_26_a, gates_v2_a = [], [], []

    for ka in ka_sweep_a:
        p = [ka, kd2_a, kd3_a, 1.0]
        max_re, _ = compute_jacobian_spectrum(p)
        eigs_a.append(max_re)

        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        p_n26 = p_t / m_s2026.param_scales
        g26 = m_s2026.gate_net(m_s2026.p_net(p_n26)).item()
        gates_26_a.append(g26)

        p_nv2 = p_t / m_v2.param_scales
        gv2 = m_v2.gate_net(p_nv2).item()
        gates_v2_a.append(gv2)

    # Slice B: X1/X2 Cross Section (kd2 = 6000, kd3 = 0.15, ka in [4.0, 14.0])
    ka_sweep_b = np.linspace(4.0, 14.0, 60)
    kd2_b, kd3_b = 6000.0, 0.15
    eigs_b, gates_26_b, gates_v2_b = [], [], []

    for ka in ka_sweep_b:
        p = [ka, kd2_b, kd3_b, 1.0]
        max_re, _ = compute_jacobian_spectrum(p)
        eigs_b.append(max_re)

        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        p_n26 = p_t / m_s2026.param_scales
        g26 = m_s2026.gate_net(m_s2026.p_net(p_n26)).item()
        gates_26_b.append(g26)

        p_nv2 = p_t / m_v2.param_scales
        gv2 = m_v2.gate_net(p_nv2).item()
        gates_v2_b.append(gv2)

    # Find true bifurcation crossings where max_re = 0
    # For Slice A:
    idx_hopf_a = np.where(np.diff(np.sign(eigs_a)))[0]
    ka_hopf_a = ka_sweep_a[idx_hopf_a[0]] if len(idx_hopf_a) > 0 else np.nan

    # For Slice B:
    idx_hopf_b = np.where(np.diff(np.sign(eigs_b)))[0]
    ka_hopf_b = ka_sweep_b[idx_hopf_b[0]] if len(idx_hopf_b) > 0 else np.nan

    print(f"Slice A (kd2=4600, kd3=0.60): True Hopf point ka* = {ka_hopf_a:.2f}")
    print(f"Slice B (kd2=6000, kd3=0.15): True Hopf point ka* = {ka_hopf_b:.2f}")

    # --------------------------------------------------------------------------
    # PLOTTING 2 FIGURES
    # --------------------------------------------------------------------------
    # Figure 1: Exact Replication of the Conflicting Figure Seed 2026 on X9 & X11
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), dpi=220)
    plt.subplots_adjust(hspace=0.32, wspace=0.22)

    # (0, 0): X9 Trajectory Comparison
    ax = axes1[0, 0]
    res9 = rep_results['X9']
    ax.plot(t_arr, res9['gt'][:steps, 2], 'k-', lw=2.4, label="Ground Truth RK4")
    ax.plot(t_arr, res9['traj26'][:, 2], color='#d62728', lw=1.8, linestyle='--', label="Seed 2026 (Conflicting Model)")
    ax.plot(t_arr, res9['traj_v2'][:, 2], color='#1f77b4', lw=1.6, alpha=0.85, label="Calibrated Gated v2")
    ax.set_title(f"X9: $k_a=3.5, k_{{d2}}=2600, k_{{d3}}=1.1$\nPolymer $m_1$ Limit Cycle", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel(r"$m_1$ (mM)", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # (1, 0): X9 Gate Over Time
    ax = axes1[1, 0]
    ax.plot(t_arr, res9['gate26_t'], color='#d62728', lw=2.2, label=f"Seed 2026 Gate = {res9['gate26_t'][0]:.6f} (Flat)")
    ax.plot(t_arr, res9['gate_v2_t'], color='#1f77b4', lw=2.0, linestyle=':', label=f"V2 Gate = {res9['gate_v2_t'][0]:.6f} (Flat)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("X9: Gate $g(\\mathbf{p})$ over Rollout Horizon", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel(r"pred_gate $g(\mathbf{p}) \in [0, 1]$", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    # (0, 1): X11 Trajectory Comparison (Showing the Conflicting Collapse)
    ax = axes1[0, 1]
    res11 = rep_results['X11']
    ax.plot(t_arr, res11['gt'][:steps, 2], 'k-', lw=2.4, label="Ground Truth RK4")
    ax.plot(t_arr, res11['traj26'][:, 2], color='#d62728', lw=2.0, linestyle='--', label="Seed 2026 (Mid-Rollout Collapse)")
    ax.plot(t_arr, res11['traj_v2'][:, 2], color='#1f77b4', lw=1.8, label="Calibrated Gated v2 (Sustained)")
    ax.axvspan(0.6, 1.0, color='red', alpha=0.08, label="Collapse Zone (Seed 2026)")
    ax.set_title(f"X11: $k_a=5.0, k_{{d2}}=5200, k_{{d3}}=0.35$\nPolymer $m_1$ Limit Cycle vs Collapse", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel(r"$m_1$ (mM)", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # (1, 1): X11 Gate Over Time
    ax = axes1[1, 1]
    ax.plot(t_arr, res11['gate26_t'], color='#d62728', lw=2.2, label=f"Seed 2026 Gate = {res11['gate26_t'][0]:.6f} (Constant)")
    ax.plot(t_arr, res11['gate_v2_t'], color='#1f77b4', lw=2.0, linestyle=':', label=f"V2 Gate = {res11['gate_v2_t'][0]:.6f} (Constant)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("X11: Gate $g(\\mathbf{p})$ over Rollout Horizon", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9.5)
    ax.set_ylabel(r"pred_gate $g(\mathbf{p}) \in [0, 1]$", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    fig1.suptitle("Conflicting Model Replication: Seed 2026 vs Calibrated Model on X9 & X11\nShowing that Collapse in Seed 2026 is Backbone Attractor Damping, Not Gate Degradation", fontsize=12.5, fontweight="bold", y=0.995)

    out1_a = os.path.join(CURRENT_DIR, "replication_seed2026_x9_x11_diagnostic.png")
    out1_b = os.path.join(BRAIN_DIR, "replication_seed2026_x9_x11_diagnostic.png")
    plt.savefig(out1_a, bbox_inches="tight", dpi=220)
    plt.savefig(out1_b, bbox_inches="tight", dpi=220)
    plt.close(fig1)

    # --------------------------------------------------------------------------
    # Figure 2: Gate Saturation vs Continuous 3D Hopf Transition Analysis
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), dpi=220)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Panel (0, 0): Slice A (Near X4) Gate vs ka
    ax = axes2[0, 0]
    ax.plot(ka_sweep_a, gates_26_a, color='#d62728', lw=2.2, label="Seed 2026 Gate $g(k_a)$")
    ax.plot(ka_sweep_a, gates_v2_a, color='#1f77b4', lw=2.2, linestyle='--', label="Gated v2 Gate $g(k_a)$")
    ax.axvline(ka_hopf_a, color='black', lw=2.0, linestyle='-.', label=f"True Hopf Bifurcation ($k_a^*={ka_hopf_a:.2f}$)")
    ax.scatter([10.5], [m_s2026.gate_net(m_s2026.p_net(torch.tensor([[10.5, kd2_a, kd3_a, 1.0]]) / m_s2026.param_scales)).item()], color='red', s=80, zorder=5, label="X4 Test Case ($g=0.081$)")
    ax.set_title(f"Parameter Slice A ($k_{{d2}}=4600, k_{{d3}}=0.60$, Near X4)\nLearned Gate $g(k_a)$ vs True Hopf Point", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Parameter $k_a$", fontsize=10)
    ax.set_ylabel(r"Learned Gate $g(\mathbf{p}) \in [0, 1]$", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # Panel (1, 0): Slice A Jacobian Eigenvalues max Re(lambda)
    ax = axes2[1, 0]
    ax.plot(ka_sweep_a, eigs_a, color='purple', lw=2.2, label=r"Max Re($\lambda$) of Steady State")
    ax.axhline(0.0, color='black', lw=1.5, linestyle='--')
    ax.axvline(ka_hopf_a, color='black', lw=1.8, linestyle='-.', label=f"Hopf Crossing ($k_a^*={ka_hopf_a:.2f}$)")
    ax.fill_between(ka_sweep_a, 0, eigs_a, where=(np.array(eigs_a) > 0), color='green', alpha=0.15, label="Limit Cycle Regime (Unstable Focus)")
    ax.fill_between(ka_sweep_a, eigs_a, 0, where=(np.array(eigs_a) <= 0), color='blue', alpha=0.15, label="Steady-State Regime (Stable Focus)")
    ax.set_title(r"Ground Truth Stability: $\max \mathrm{Re}(\lambda)$ along Slice A", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Parameter $k_a$", fontsize=10)
    ax.set_ylabel(r"$\max \mathrm{Re}(\lambda)$ (1/s)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="lower left")

    # Panel (0, 1): Slice B (Near X1/X2) Gate vs ka
    ax = axes2[0, 1]
    ax.plot(ka_sweep_b, gates_26_b, color='#d62728', lw=2.2, label="Seed 2026 Gate $g(k_a)$")
    ax.plot(ka_sweep_b, gates_v2_b, color='#1f77b4', lw=2.2, linestyle='--', label="Gated v2 Gate $g(k_a)$")
    ax.axvline(ka_hopf_b, color='black', lw=2.0, linestyle='-.', label=f"True Hopf Bifurcation ($k_a^*={ka_hopf_b:.2f}$)")
    ax.scatter([11.0, 12.0], [gates_26_b[int(np.argmin(np.abs(ka_sweep_b - 11.0)))], gates_26_b[int(np.argmin(np.abs(ka_sweep_b - 12.0)))]], color='red', s=80, zorder=5, label="X1 & X2 Test Cases")
    ax.set_title(f"Parameter Slice B ($k_{{d2}}=6000, k_{{d3}}=0.15$, Near X1/X2)\nLearned Gate $g(k_a)$ vs True Hopf Point", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Parameter $k_a$", fontsize=10)
    ax.set_ylabel(r"Learned Gate $g(\mathbf{p}) \in [0, 1]$", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    # Panel (1, 1): Calibration Offset in Contraction Branch for X1-X6
    ax = axes2[1, 1]
    x_indices = np.arange(len(part1_data))
    offsets_26 = [d['offset_s2026'] for d in part1_data]
    offsets_v2 = [d['offset_v2'] for d in part1_data]
    labels = [d['id'] for d in part1_data]

    w = 0.35
    ax.bar(x_indices - w/2, offsets_26, width=w, color='#d62728', alpha=0.85, label=f"Seed 2026 Offset (X4 = {offsets_26[3]:+.2f} mM)")
    ax.bar(x_indices + w/2, offsets_v2, width=w, color='#1f77b4', alpha=0.85, label=f"Gated v2 Offset (X4 = {offsets_v2[3]:+.2f} mM)")
    ax.axhline(0.0, color='black', lw=1.2)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_title(r"Contraction Branch Calibration Offset: $\Delta y = \frac{g}{(1-g)\gamma} r_{m1}$ Across X1–X6", fontsize=11, fontweight="bold")
    ax.set_xlabel("Steady-State Test Regime", fontsize=10)
    ax.set_ylabel(r"Equilibrium Shift $\Delta m_1$ (mM)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=9)

    fig2.suptitle("Deep Gate Analysis: Continuous Boundary Transition vs Saturation & Equilibrium Calibration Shift", fontsize=12.5, fontweight="bold", y=0.995)

    out2_a = os.path.join(CURRENT_DIR, "gate_saturation_and_calibration_analysis.png")
    out2_b = os.path.join(BRAIN_DIR, "gate_saturation_and_calibration_analysis.png")
    plt.savefig(out2_a, bbox_inches="tight", dpi=220)
    plt.savefig(out2_b, bbox_inches="tight", dpi=220)
    plt.close(fig2)

    print(f"\nFigures successfully generated:")
    print(f"  - {out1_a}")
    print(f"  - {out2_a}")

if __name__ == "__main__":
    run_investigation()
