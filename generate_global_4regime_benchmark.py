import os
import sys

# Ensure OpenBLAS thread count is 1
if sys.platform.startswith("win") and "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    rk4_step,
    simulate_trajectory,
    DEFAULT_GUESS,
    DT,
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP
)

# Load Models
ckpt_film = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
model_prop = ProposedFiLMPhysicsConstrainedMLP()
model_prop.load_state_dict(torch.load(ckpt_film, map_location="cpu"))
model_prop.eval()

ckpt_comb = os.path.join(CURRENT_DIR, "film_divergence_regularized_mlp.pt")
model_comb = ProposedFiLMPhysicsConstrainedMLP()
model_comb.load_state_dict(torch.load(ckpt_comb, map_location="cpu"))
model_comb.eval()

ckpt_base = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
model_base = UnconstrainedBaselineMLP()
model_base.load_state_dict(torch.load(ckpt_base, map_location="cpu"))
model_base.eval()

ckpt_div = os.path.join(CURRENT_DIR, "divergence_regularized_mlp.pt")
model_div = UnconstrainedBaselineMLP()
model_div.load_state_dict(torch.load(ckpt_div, map_location="cpu"))
model_div.eval()

REGIMES = [
    {
        "id": "Case A",
        "title": r"Case A: Nominal Slide [Interpolation]",
        "subtitle": r"$k_a = 6.0\,\mathrm{s}^{-1},\ k_{d2} = 4208.0\,\mathrm{mM}^{-1}\mathrm{s}^{-1},\ k_{d3} = 1.0\,\mathrm{s}^{-1}$",
        "params": (6.0, 4208.0, 1.0, 1.0),
        "steps": 25000,
        "t_plot": 2.5,
        "ylim_t": (40, 340),
        "xlim_p": (1.2, 7.8),
        "ylim_p": (45, 335),
        "has_inset": False
    },
    {
        "id": "Case B",
        "title": r"Case B: Slower High-Amplitude Regime [Interpolation]",
        "subtitle": r"$k_a = 3.0\,\mathrm{s}^{-1},\ k_{d2} = 4200.0\,\mathrm{mM}^{-1}\mathrm{s}^{-1},\ k_{d3} = 0.9\,\mathrm{s}^{-1}$",
        "params": (3.0, 4200.0, 0.9, 1.0),
        "steps": 35000,
        "t_plot": 2.5,
        "ylim_t": (-20, 720),
        "xlim_p": (-0.5, 13.5),
        "ylim_p": (-20, 680),
        "has_inset": False
    },
    {
        "id": "Case C",
        "title": r"Case C: Giant Amplitude Surge [Extrapolation]",
        "subtitle": r"$k_a = 5.0\,\mathrm{s}^{-1},\ k_{d2} = 4000.0\,\mathrm{mM}^{-1}\mathrm{s}^{-1},\ k_{d3} = 0.5\,\mathrm{s}^{-1}$",
        "params": (5.0, 4000.0, 0.5, 1.0),
        "steps": 30000,
        "t_plot": 2.5,
        "ylim_t": (-30, 1150),
        "xlim_p": (-0.5, 15.5),
        "ylim_p": (-30, 1080),
        "has_inset": True,
        "inset_xlim": (-5, 45),
        "inset_ylim": (-100, 3600)
    },
    {
        "id": "Case D",
        "title": r"Case D: Fast Stiff Cycle [Extrapolation]",
        "subtitle": r"$k_a = 9.0\,\mathrm{s}^{-1},\ k_{d2} = 4500.0\,\mathrm{mM}^{-1}\mathrm{s}^{-1},\ k_{d3} = 0.6\,\mathrm{s}^{-1}$",
        "params": (9.0, 4500.0, 0.6, 1.0),
        "steps": 25000,
        "t_plot": 2.5,
        "ylim_t": (195, 305),
        "xlim_p": (4.75, 5.75),
        "ylim_p": (225, 275),
        "has_inset": True,
        "inset_xlim": (-1, 16),
        "inset_ylim": (-50, 1100)
    }
]

print("Simulating 4 regimes...")
trajs = {}

for regime in REGIMES:
    rid = regime["id"]
    p = regime["params"]
    p_arr = np.array(p, dtype=np.float64)
    steps = regime["steps"]
    
    y_warm = DEFAULT_GUESS.copy()
    for _ in range(int(20.0 / DT)):
        y_warm = rk4_step(y_warm, DT, p_arr)
    
    gt_traj = simulate_trajectory(y_warm, steps, DT, p_arr)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
    c_tot0 = torch.tensor(y_warm[:3].sum(), dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    
    reg_trajs = {"gt": gt_traj}
    models = [("comb", model_comb), ("prop", model_prop), ("div", model_div), ("base", model_base)]
    for key, mod in models:
        curr = torch.tensor(y_warm, dtype=torch.float32).unsqueeze(0)
        c_tot0 = curr[:, 0:1] + curr[:, 1:2] + curr[:, 2:3]
        pred = [y_warm.copy()]
        with torch.no_grad():
            for _ in range(steps):
                curr = mod(curr, p_t, target_ctot=c_tot0)
                pred.append(curr.squeeze(0).cpu().numpy())
        reg_trajs[key] = np.array(pred)
        
    trajs[rid] = reg_trajs

# Clean, uncluttered publication styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

c_gt = "#0f172a"      # Deep Slate Black
c_comb = "#0284c7"    # Electric Sky Blue
c_prop = "#059669"    # Emerald Green
c_div = "#d97706"     # Amber
c_base = "#f43f5e"    # Coral Red

fig, axes = plt.subplots(4, 2, figsize=(16, 18), dpi=220)
plt.subplots_adjust(hspace=0.38, wspace=0.20, top=0.915, bottom=0.04, left=0.07, right=0.96)

for i, regime in enumerate(REGIMES):
    rid = regime["id"]
    reg_trajs = trajs[rid]
    
    t_end = regime["t_plot"]
    n_plot = int(t_end / DT)
    t_span = np.linspace(0, t_end, n_plot)
    
    # -------------------------------------------------------------
    # Left Panel: Time Series m1(t)
    # -------------------------------------------------------------
    ax_t = axes[i, 0]
    
    # Unconstrained models in background
    ax_t.plot(t_span, reg_trajs["base"][:n_plot, 2], color=c_base, lw=1.1, alpha=0.35, linestyle="-", label="Baseline SciML (Unconstrained)")
    ax_t.plot(t_span, reg_trajs["div"][:n_plot, 2], color=c_div, lw=1.2, alpha=0.75, linestyle="-", label="Div MLP (Ablation)")
    # Ground Truth
    ax_t.plot(t_span, reg_trajs["gt"][:n_plot, 2], color=c_gt, lw=2.2, label="Ground Truth RK4")
    # Proposed models
    ax_t.plot(t_span, reg_trajs["comb"][:n_plot, 2], color=c_comb, lw=1.9, linestyle=(0, (4, 1.8)), label="FiLM + Div Combined (Ours)")
    ax_t.plot(t_span, reg_trajs["prop"][:n_plot, 2], color=c_prop, lw=1.6, linestyle=(0, (2.5, 1.5, 1, 1.5)), label="FiLM Pure (Ours)")
    
    ax_t.set_xlim(0, t_end)
    ax_t.set_ylim(regime["ylim_t"])
    ax_t.grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    
    ax_t.set_title(f"{regime['title']}\n{regime['subtitle']}", fontweight="bold", fontsize=11, loc="left", pad=6)
    ax_t.set_xlabel("Time (s)", fontweight="bold")
    ax_t.set_ylabel(r"Polymer $m_1(t)$ [mM]", fontweight="bold")

    # -------------------------------------------------------------
    # Right Panel: Phase Portrait (Attractor-Focused)
    # -------------------------------------------------------------
    ax_p = axes[i, 1]
    s_idx = int(regime["steps"] * 0.4)
    
    ax_p.plot(reg_trajs["base"][s_idx:, 3], reg_trajs["base"][s_idx:, 2], color=c_base, lw=1.0, alpha=0.40, linestyle="-")
    ax_p.plot(reg_trajs["div"][s_idx:, 3], reg_trajs["div"][s_idx:, 2], color=c_div, lw=1.2, alpha=0.75, linestyle="-")
    ax_p.plot(reg_trajs["gt"][s_idx:, 3], reg_trajs["gt"][s_idx:, 2], color=c_gt, lw=2.2, label="Ground Truth $\\gamma$")
    ax_p.plot(reg_trajs["comb"][s_idx:, 3], reg_trajs["comb"][s_idx:, 2], color=c_comb, lw=1.9, linestyle=(0, (4, 1.8)), label="FiLM + Div Combined")
    ax_p.plot(reg_trajs["prop"][s_idx:, 3], reg_trajs["prop"][s_idx:, 2], color=c_prop, lw=1.5, linestyle=(0, (2.5, 1.5, 1, 1.5)), label="FiLM Pure")
    
    ax_p.set_xlim(regime["xlim_p"])
    ax_p.set_ylim(regime["ylim_p"])
    ax_p.grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
    
    ax_p.set_title(rf"Phase Portrait: Orbital Geometry ($m_0$ vs $m_1$) — {regime['id']}",
                   fontweight="bold", fontsize=11, loc="left", pad=6)
    ax_p.set_xlabel(r"Pacemaker $m_0$ [mM]", fontweight="bold")
    ax_p.set_ylabel(r"Polymer $m_1$ [mM]", fontweight="bold")
    
    # Inset for Extrapolative blowup
    if regime["has_inset"]:
        ax_ins = inset_axes(ax_p, width="32%", height="32%", loc="upper right", borderpad=0.8)
        ax_ins.plot(reg_trajs["base"][s_idx:, 3], reg_trajs["base"][s_idx:, 2], color=c_base, lw=0.8, alpha=0.55)
        ax_ins.plot(reg_trajs["div"][s_idx:, 3], reg_trajs["div"][s_idx:, 2], color=c_div, lw=0.9, alpha=0.7)
        ax_ins.plot(reg_trajs["gt"][s_idx:, 3], reg_trajs["gt"][s_idx:, 2], color=c_gt, lw=1.3)
        ax_ins.plot(reg_trajs["comb"][s_idx:, 3], reg_trajs["comb"][s_idx:, 2], color=c_comb, lw=1.1, linestyle="--")
        
        ax_ins.set_xlim(regime["inset_xlim"])
        ax_ins.set_ylim(regime["inset_ylim"])
        ax_ins.set_title("Global Drift", fontsize=7.5, pad=2, fontweight="bold", color="#991b1b")
        ax_ins.tick_params(axis='both', which='major', labelsize=6.5, pad=1)
        ax_ins.grid(True, linestyle=":", color="#e2e8f0", alpha=0.7)
        
        rect = patches.Rectangle((regime["xlim_p"][0], regime["ylim_p"][0]),
                                 regime["xlim_p"][1] - regime["xlim_p"][0],
                                 regime["ylim_p"][1] - regime["ylim_p"][0],
                                 linewidth=1.0, edgecolor="#0284c7", facecolor="none", linestyle="--")
        ax_ins.add_patch(rect)

# Master Unified Legend across top
handles, labels = axes[0, 0].get_legend_handles_labels()
custom_order = [2, 3, 4, 1, 0]
ordered_handles = [handles[idx] for idx in custom_order]
ordered_labels = [labels[idx] for idx in custom_order]

fig.legend(ordered_handles, ordered_labels, loc="upper center", ncol=5,
           fontsize=10.5, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1",
           bbox_to_anchor=(0.515, 0.955), handlelength=2.2, handletextpad=0.5, columnspacing=1.8)

fig.suptitle("Global 4-Regime Dynamical Verification Grid: FiLM SciML Across the Hopf Bifurcation Surface",
             fontweight="bold", fontsize=14, y=0.986)

out_master = os.path.join(CURRENT_DIR, "global_4regime_benchmark_grid.png")
brain_master = os.path.join(r"C:\Users\satya\.gemini\antigravity-ide\brain\c0229372-9c2c-4e39-b054-0c23ec47512a", "global_4regime_benchmark_grid.png")

plt.savefig(out_master, dpi=220, bbox_inches="tight")
plt.close(fig)

import shutil
shutil.copyfile(out_master, brain_master)
print(f"Clean grid generated successfully to:\n  - {out_master}\n  - {brain_master}", flush=True)
