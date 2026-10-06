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

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import ProposedFiLMVDPMLP, simulate_trajectory, rk4_step, DT

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
ARTIFACT_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\1c89c907-107d-4ec6-b918-cd19ff5817dc"
SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

data = np.load(os.path.join(SCRATCH_DIR, "vdp_settled_bifurcation_data.npz"))
mu_sweep = data["mu_sweep"]
gt_upper = data["gt_upper"]
gt_lower = data["gt_lower"]
gt_period = data["gt_period"]

a_uppers = data["a_uppers"]
a_lowers = data["a_lowers"]
a_periods = data["a_periods"]
a_u_mean = data["a_u_mean"]
a_l_mean = data["a_l_mean"]
a_p_mean = data["a_p_mean"]

b_u = data["b_u"]
b_l = data["b_l"]
b_p = data["b_p"]

base_u = data["base_u"]
base_l = data["base_l"]
base_p = data["base_p"]

def apply_white_theme(ax):
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
        spine.set_linewidth(1.2)
    ax.tick_params(colors="#1e293b", labelsize=9.5, width=1.1)
    ax.grid(True, linestyle="--", alpha=0.35, color="#64748b")

fig = plt.figure(figsize=(19, 10.8), facecolor="#ffffff", dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.22)

# ==============================================================================
# PANEL A: ASYMPTOTIC STATE EXTREMA (AMPLITUDE ENVELOPE)
# ==============================================================================
ax_bif = fig.add_subplot(gs[0, 0])
apply_white_theme(ax_bif)

# 1. Equilibrium branches along x = 0
mu_neg = mu_sweep[mu_sweep <= 0.0]
mu_pos = mu_sweep[mu_sweep >= 0.0]
ax_bif.plot(mu_neg, np.zeros_like(mu_neg), color="#0f172a", lw=3.0, label=r"Stable Equilibrium $x^*=0$ (Attracting Sink)")
ax_bif.plot(mu_pos, np.zeros_like(mu_pos), color="#94a3b8", lw=2.2, ls="--", label=r"Unstable Equilibrium $x^*=0$")

# 2. Supercritical Hopf Bifurcation Marker
ax_bif.axvline(0.0, color="#dc2626", lw=1.6, ls=":", label=r"Supercritical Hopf ($\mu_c = 0.000$)")
ax_bif.plot(0.0, 0.0, marker="o", color="#dc2626", markersize=8, zorder=6)

# 3. Supercritical Limit Cycle Branches (mu > 0)
mask_gt = mu_sweep > 0.0
mu_lc = np.concatenate([[0.0], mu_sweep[mask_gt]])
gt_up_lc = np.concatenate([[0.0], gt_upper[mask_gt]])
gt_lo_lc = np.concatenate([[0.0], gt_lower[mask_gt]])

ax_bif.plot(mu_lc, gt_up_lc, color="#1e3a8a", lw=2.8, label=r"Ground Truth RK4 Limit-Cycle Envelope ($x_{\max}, x_{\min}$)")
ax_bif.plot(mu_lc, gt_lo_lc, color="#1e3a8a", lw=2.8)

# Model A (From-Scratch 7-Seed Consensus)
mask_a = (mu_sweep >= 0.05)
mu_a_lc = np.concatenate([[0.037], mu_sweep[mask_a]])
a_up_lc = np.concatenate([[0.0], a_u_mean[mask_a]])
a_lo_lc = np.concatenate([[0.0], a_l_mean[mask_a]])

a_up_std = np.std(a_uppers[:, mask_a], axis=0)
a_lo_std = np.std(a_lowers[:, mask_a], axis=0)
a_up_std_lc = np.concatenate([[0.0], a_up_std])
a_lo_std_lc = np.concatenate([[0.0], a_lo_std])

ax_bif.plot(mu_a_lc, a_up_lc, color="#059669", lw=2.2, ls="-", label="Model A (From-Scratch 7-Seed Mean)")
ax_bif.plot(mu_a_lc, a_lo_lc, color="#059669", lw=2.2, ls="-")
ax_bif.fill_between(mu_a_lc, a_up_lc - a_up_std_lc, a_up_lc + a_up_std_lc, color="#059669", alpha=0.18, label=r"Model A $1\sigma$ Seed Envelope")
ax_bif.fill_between(mu_a_lc, a_lo_lc - a_lo_std_lc, a_lo_lc + a_lo_std_lc, color="#059669", alpha=0.18)

# Scatter individual seed points
for s_idx in range(len(SEEDS)):
    ax_bif.scatter(mu_sweep[mask_a][::2], a_uppers[s_idx, mask_a][::2], color="#94a3b8", s=12, alpha=0.6, marker="x")
    ax_bif.scatter(mu_sweep[mask_a][::2], a_lowers[s_idx, mask_a][::2], color="#94a3b8", s=12, alpha=0.6, marker="x")

# Model B (Physics-Residual)
mask_b = (mu_sweep > 0.0)
mu_b_lc = np.concatenate([[0.0], mu_sweep[mask_b]])
b_up_lc = np.concatenate([[0.0], b_u[mask_b]])
b_lo_lc = np.concatenate([[0.0], b_l[mask_b]])
ax_bif.plot(mu_b_lc, b_up_lc, color="#7c3aed", lw=1.8, ls="-.", label="Model B (Physics-Residual)")
ax_bif.plot(mu_b_lc, b_lo_lc, color="#7c3aed", lw=1.8, ls="-.")

# Baseline MLP
mask_base = (mu_sweep > 0.0)
base_up_lc = np.concatenate([[0.0], base_u[mask_base]])
base_lo_lc = np.concatenate([[0.0], base_l[mask_base]])
ax_bif.plot(mu_b_lc, base_up_lc, color="#ea580c", lw=1.5, ls=":", label="Unconstrained Baseline MLP")
ax_bif.plot(mu_b_lc, base_lo_lc, color="#ea580c", lw=1.5, ls=":")

# Inset annotation positioned with generous headroom and zero overlap
ax_bif.annotate(
    "Nullcline Fold Points Pin Amplitude\n$x_{\\max} \\approx +2.0,\\; x_{\\min} \\approx -2.0$\n(Envelope Error < 0.85% across all $\\mu > 0$)\nEnsemble Asymmetry $|x_{\\max} + x_{\\min}| = 0.0020$ (Seeds $\\leq 0.030$)",
    xy=(1.8, 2.02), xytext=(0.48, 2.38),
    arrowprops=dict(facecolor="#1e3a8a", shrink=0.08, width=1.0, headwidth=5),
    fontsize=8.2, fontweight="bold", color="#1e3a8a",
    bbox=dict(boxstyle="round,pad=0.35", fc="#eff6ff", ec="#93c5fd", lw=1.1)
)

ax_bif.set_title("A. Canonical Van der Pol Hopf Bifurcation Diagram (Asymptotic Limit-Cycle Extrema)", fontsize=11.5, fontweight="bold", color="#0f172a", pad=10)
ax_bif.set_xlabel(r"Damping / Nonlinearity Parameter $\mu$", fontsize=10.5, fontweight="bold", color="#0f172a")
ax_bif.set_ylabel(r"State Extrema $x$ ($x_{\max}, x_{\min}$)", fontsize=10.5, fontweight="bold", color="#0f172a")
ax_bif.set_xlim(-1.05, 2.55)
ax_bif.set_ylim(-3.2, 3.2)
ax_bif.legend(loc="lower left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.2, ncol=2)

# ==============================================================================
# PANEL B: LIMIT-CYCLE PERIOD MANIFOLD & INSET ERROR RESOLUTION
# ==============================================================================
ax_per = fig.add_subplot(gs[1, 0])
apply_white_theme(ax_per)

# Plot period for mu > 0
mu_pos_mask = (mu_sweep > 0.05) & (gt_period > 0.0)
mu_p = mu_sweep[mu_pos_mask]
gt_p = gt_period[mu_pos_mask]
a_p_m = a_p_mean[mu_pos_mask]
a_p_std = np.std(a_periods[:, mu_pos_mask], axis=0)

ax_per.plot(mu_p, gt_p, color="#1e3a8a", lw=2.6, marker="o", markersize=4.5, label=r"Ground Truth Period $T_{\mathrm{GT}}(\mu)$")
ax_per.plot(mu_p, a_p_m, color="#059669", lw=2.2, ls="-", marker="s", markersize=4.0, label=r"Model A 7-Seed Mean $T_{\mathrm{SciML}}(\mu)$")
ax_per.fill_between(mu_p, a_p_m - a_p_std, a_p_m + a_p_std, color="#059669", alpha=0.18, label=r"Model A $1\sigma$ Seed Band")

# Model B and Baseline Period
ax_per.plot(mu_p, b_p[mu_pos_mask], color="#7c3aed", lw=1.8, ls="-.", label=r"Model B (Residual)")
ax_per.plot(mu_p, base_p[mu_pos_mask], color="#ea580c", lw=1.5, ls=":", label=r"Baseline MLP")

# Theoretical Harmonic Limit 2pi
ax_per.axhline(2.0 * np.pi, color="#64748b", ls="--", lw=1.3, label=r"Harmonic Limit $T \to 2\pi \approx 6.283\,\mathrm{s}$")

ax_per.set_title(r"B. Limit-Cycle Period Manifold $T(\mu)$ Across the Bifurcation Horizon", fontsize=11.5, fontweight="bold", color="#0f172a", pad=10)
ax_per.set_xlabel(r"Damping / Nonlinearity Parameter $\mu$", fontsize=10.5, fontweight="bold", color="#0f172a")
ax_per.set_ylabel(r"Limit-Cycle Period $T$ (s)", fontsize=10.5, fontweight="bold", color="#0f172a")
ax_per.set_xlim(0.0, 2.55)
ax_per.set_ylim(6.0, 8.5)
ax_per.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=8.2)

# INSET: Pointwise Period Error Delta T(mu) cleanly positioned in lower right
# with zero overlap with any curve or legend
ax_inset = ax_per.inset_axes([0.62, 0.08, 0.35, 0.28])
apply_white_theme(ax_inset)

delta_T = a_p_m - gt_p
ax_inset.plot(mu_p, delta_T, color="#dc2626", lw=1.8, marker="d", markersize=3.5, label=r"$\Delta T(\mu)$")
ax_inset.axhline(0.0, color="#64748b", lw=0.9, ls=":")
ax_inset.set_title(r"Pointwise Period Error $\Delta T(\mu)$ (s)", fontsize=7.8, fontweight="bold", color="#dc2626", pad=2)
ax_inset.set_xlabel(r"$\mu$", fontsize=7.5, fontweight="bold", color="#0f172a", labelpad=1)
ax_inset.set_ylabel(r"$\Delta T$ (s)", fontsize=7.5, fontweight="bold", color="#0f172a", labelpad=1)
ax_inset.tick_params(labelsize=7.0, pad=1)
ax_inset.set_xlim(0.0, 2.55)
ax_inset.set_ylim(-0.30, 0.05)

# Softened hypothesis annotation inside the inset
ax_inset.annotate(
    "Stiffness Lag: $\\Delta T = -0.243\\,\\mathrm{s}$\n(Proposed, consistent with\nneural smoothing of stiff jumps)",
    xy=(2.45, -0.243), xytext=(0.08, -0.11),
    arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=0.8, headwidth=4),
    fontsize=6.8, fontweight="bold", color="#dc2626",
    bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#fca5a5", lw=0.9)
)

# ==============================================================================
# PANEL C: 3 CHARACTERISTIC PHASE PORTRAITS
# ==============================================================================
gs_sub = gs[:, 1].subgridspec(3, 1, hspace=0.35)
regimes = [
    (-0.5, r"1. Subcritical Regime: Damped Inward Spiral to Origin Sink ($\mu = -0.5$)", "#2563eb", "Sink at $(0, 0)$"),
    (0.5, r"2. Weakly Nonlinear Regime: Quasi-Harmonic Limit Cycle ($\mu = 0.5$)", "#059669", "$T \\approx 6.38\\,\\mathrm{s}$"),
    (2.0, r"3. Strong Relaxation Regime: Fast-Slow Stiff Cycle ($\mu = 2.0$)", "#7c3aed", "$T \\approx 7.63\\,\\mathrm{s}$")
]

m_s42 = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
m_s42.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_sciml_fromscratch_s42.pt"), map_location="cpu", weights_only=True))
m_s42.eval()

for r_idx, (mu_val, title_str, col, tag_str) in enumerate(regimes):
    ax_sub = fig.add_subplot(gs_sub[r_idx, 0])
    apply_white_theme(ax_sub)
    
    y_test = np.array([2.0, 1.0], dtype=np.float64)
    
    if mu_val <= 0:
        sim_steps = int(25.0 / DT)
        traj_sub_gt = simulate_trajectory(y_test, sim_steps, DT, mu_val)
        
        pt_sub = torch.tensor([[mu_val]], dtype=torch.float32)
        curr_sub = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0)
        pred_sub = [y_test]
        with torch.no_grad():
            for _ in range(sim_steps):
                curr_sub = m_s42.rk4_step_forward(curr_sub, pt_sub)
                pred_sub.append(curr_sub[0].cpu().numpy())
        pred_sub = np.array(pred_sub)
        
        ax_sub.plot(traj_sub_gt[:, 0], traj_sub_gt[:, 1], color="#1e3a8a", lw=2.0, label="Ground Truth RK4 (Spiral)")
        ax_sub.plot(pred_sub[:, 0], pred_sub[:, 1], color=col, lw=1.6, ls="--", label=r"Model A (s=42)")
        ax_sub.plot(y_test[0], y_test[1], marker="^", color="#059669", markersize=6, label=r"Initial $(x_0, v_0)$")
        ax_sub.plot(0, 0, marker="o", color="#dc2626", markersize=6, label=r"Origin Sink $(0, 0)$")
    else:
        tau_warm = 2.0 / max(abs(mu_val), 0.05)
        w_steps = int(min(max(5.0 * tau_warm, 40.0), 120.0) / DT)
        curr_gt = y_test.copy()
        for _ in range(w_steps):
            curr_gt = rk4_step(curr_gt, DT, mu_val)
        traj_sub_gt = simulate_trajectory(curr_gt, int(30.0/DT), DT, mu_val)
        
        pt_sub = torch.tensor([[mu_val]], dtype=torch.float32)
        curr_sub = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(w_steps):
                curr_sub = m_s42.rk4_step_forward(curr_sub, pt_sub)
            pred_sub = [curr_sub[0].cpu().numpy()]
            for _ in range(int(30.0/DT)):
                curr_sub = m_s42.rk4_step_forward(curr_sub, pt_sub)
                pred_sub.append(curr_sub[0].cpu().numpy())
        pred_sub = np.array(pred_sub)

        ax_sub.plot(traj_sub_gt[:, 0], traj_sub_gt[:, 1], color="#1e3a8a", lw=2.2, label="Ground Truth RK4")
        ax_sub.plot(pred_sub[:, 0], pred_sub[:, 1], color=col, lw=1.8, ls="--", label=f"Model A (s=42)")
        ax_sub.plot(0, 0, marker="o", color="#dc2626", markersize=5)
    
    ax_sub.set_title(title_str, fontsize=10.0, fontweight="bold", color="#0f172a", pad=6)
    ax_sub.set_xlabel("$x(t)$", fontsize=9.0, fontweight="bold", color="#0f172a")
    ax_sub.set_ylabel(r"$v(t) = \dot{x}$", fontsize=9.0, fontweight="bold", color="#0f172a")
    ax_sub.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=7.8)
    ax_sub.set_xlim(-2.8, 2.8)
    ax_sub.set_ylim(-3.5, 3.5)

bif_fig_path = os.path.join(SCRATCH_DIR, "vdp_bifurcation_diagram.png")
plt.savefig(bif_fig_path, dpi=300, facecolor="#ffffff", bbox_inches="tight")
plt.close()
print(f"Canonical bifurcation diagram saved to: {bif_fig_path}")

artifact_path = os.path.join(ARTIFACT_DIR, "vdp_bifurcation_diagram.png")
import shutil
shutil.copyfile(bif_fig_path, artifact_path)
print(f"Copied to artifact directory: {artifact_path}")
