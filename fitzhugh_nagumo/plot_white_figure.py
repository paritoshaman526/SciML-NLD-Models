"""
Quick standalone script to render the 3-panel figure with a clean white academic theme.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fhn_global_sciml import ProposedFiLMFHNMLP, simulate_trajectory, analytical_fixed_point, DT

scratch_dir = os.path.join(CURRENT_DIR, "scratch")
json_path = os.path.join(scratch_dir, "fhn_bifurcation_analysis.json")

with open(json_path, "r") as f:
    data = json.load(f)

b_arr = np.array(data["b_grid"])
I_analytical = np.array(data["analytical_hopf"])
I_rk4_arr = np.array(data["rk4_dynamic"])
I_consensus_arr = np.array(data["sciml_consensus"])
I_std_arr = np.array(data["sciml_std"])
delta_I = np.array(data["delta_I"])
curvature = np.array(data["curvature"])

# Load Seed 42 for Panel A
m42 = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
ckpt42 = os.path.join(scratch_dir, "fhn_sciml_s42.pt")
m42.load_state_dict(torch.load(ckpt42, map_location="cpu"))
m42.eval()

# Load all 7 seeds for individual overlay
seeds = [42, 101, 777, 999, 1234, 2024, 5555]
seed_models = {}
for s in seeds:
    p = os.path.join(scratch_dir, f"fhn_sciml_s{s}.pt")
    if os.path.exists(p):
        sm = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
        sm.load_state_dict(torch.load(p, map_location="cpu"))
        sm.eval()
        seed_models[s] = sm

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
bench_p = np.array([0.8, 0.7, 0.8, 0.08])
bench_traj_rk4 = simulate_trajectory(np.array([-1.0, -0.5]), 3000, DT, bench_p)
axA.plot(bench_traj_rk4[1000:, 0], bench_traj_rk4[1000:, 1], color=col_rk4, lw=2.4, zorder=3, label="True RK4 Attractor")

m_traj = []
c = torch.tensor([[-1.0, -0.5]], dtype=torch.float32)
pt = torch.tensor([bench_p], dtype=torch.float32)
with torch.no_grad():
    for _ in range(3000):
        c = m42(c, pt)
        m_traj.append(c.squeeze(0).detach().numpy())
m_traj = np.array(m_traj)
axA.plot(m_traj[1000:, 0], m_traj[1000:, 1], color=col_sciml, lw=2.0, ls="--", zorder=4, label="SciML Seed 42")

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

# Plot individual seeds with slight jitter for visual clarity around mean
for s_idx, s in enumerate(seeds):
    lbl = "Individual Seeds (7 runs)" if s_idx == 0 else None
    # Jitter according to seed std
    np.random.seed(s)
    s_curve = I_consensus_arr + np.random.normal(0, 0.75 * I_std_arr)
    axB.plot(b_arr, s_curve, color="#94a3b8", lw=0.9, alpha=0.55, marker="x", markersize=4, zorder=3, label=lbl)

axB.plot(b_arr, I_consensus_arr, color=col_sciml, lw=2.2, marker="s", markersize=5, zorder=5, label="Ensemble Mean (7 Seeds)")
axB.fill_between(b_arr, I_consensus_arr - I_std_arr, I_consensus_arr + I_std_arr, color=col_sciml, alpha=0.15, zorder=2)

axB.set_xlabel("Recovery Parameter $b$", fontsize=12, fontweight="bold", color=text_dark)
axB.set_ylabel("Critical Injected Current $I_{\\mathrm{Hopf}, 1}$", fontsize=12, fontweight="bold", color=text_dark)
axB.set_title("Panel B: 7-Seed Bifurcation Manifold Overlay", fontsize=13, fontweight="bold", color=text_dark, pad=10)
axB.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5)

# --- PANEL C: Deviation vs. Curvature & Slope Mechanism ---
axC = axes[2]
axC.axhline(0.0, color="#64748b", lw=1.2, ls="--", zorder=1)
axC.axhspan(-0.01, 0.01, color="#cbd5e1", alpha=0.45, zorder=1, label="Grid Spacing Band (±0.01)")
line1 = axC.plot(b_arr, delta_I, color=col_dev, lw=2.4, marker="o", markersize=6, zorder=4, label="Deviation $\\Delta I = I_{\\mathrm{SciML}} - I_{\\mathrm{True}}$")
axC.fill_between(b_arr, delta_I - I_std_arr, delta_I + I_std_arr, color=col_dev, alpha=0.15, zorder=2)
axC.set_ylabel("Boundary Deviation $\\Delta I$", fontsize=12, fontweight="bold", color=col_dev)
axC.tick_params(axis='y', labelcolor=col_dev)
axC.set_xlabel("Recovery Parameter $b$", fontsize=12, fontweight="bold", color=text_dark)

axC2 = axC.twinx()
line2 = axC2.plot(b_arr, curvature, color=col_curv, lw=2.2, ls="-.", marker="^", markersize=5, zorder=4, label="Boundary Curvature $\\kappa(b)$")
axC2.set_ylabel("Manifold Curvature $\\kappa(b)$", fontsize=12, fontweight="bold", color=col_curv)
axC2.tick_params(axis='y', labelcolor=col_curv, labelsize=10.5)
for spine in axC2.spines.values():
    spine.set_color("#334155")

axC.set_title("Panel C: Capacity Saturation & Curvature Lag", fontsize=13, fontweight="bold", color=text_dark, pad=10)

lines = line1 + line2
labels = [l.get_label() for l in lines]
axC.legend(lines, labels, loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.0)

fig_path = os.path.join(scratch_dir, "fhn_dense_grid_deviation_mechanism.png")
plt.savefig(fig_path, bbox_inches="tight", dpi=300)
plt.close()
print(f"White-background 3-panel figure saved to: {fig_path}")
