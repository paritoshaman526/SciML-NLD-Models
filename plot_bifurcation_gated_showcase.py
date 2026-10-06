import os
import sys
import numpy as np
import scipy.optimize
import torch
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
    ProposedFiLMPhysicsConstrainedMLP,
    DT,
    DEFAULT_GUESS,
    rk4_step,
    simulate_trajectory
)
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state, rollout_any
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML

def rollout_gated_model(model, ic, params, steps, ss_target=None):
    model.eval()
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]
    ss_t = torch.tensor(ss_target, dtype=torch.float32).unsqueeze(0) if ss_target is not None else None

    traj = [ic.copy()]
    with torch.no_grad():
        for _ in range(steps):
            curr = model(curr, p_t, target_ctot=c_tot, steady_state_target=ss_t)
            traj.append(curr.squeeze(0).cpu().numpy())
    return np.array(traj)

def main():
    print("Generating Comparative Showcase Figure for Bifurcation-Gated Architecture...")

    # Load 3 models:
    # 1. Original FiLM Pure
    m_pure = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_pure.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt"), map_location="cpu"))
    m_pure.eval()

    # 2. Naive Bifurcation-Aware FiLM (Seed 42)
    m_naive = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_naive.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_bifurcation_aware_s42.pt"), map_location="cpu"))
    m_naive.eval()

    # 3. New Bifurcation-Gated FiLM (Seed 2026)
    m_gated = BifurcationGatedFiLMSciML(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_gated.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_bifurcation_gated_s2026.pt"), map_location="cpu"))
    m_gated.eval()

    fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=220)
    plt.subplots_adjust(hspace=0.42, wspace=0.22)

    for idx, c_info in enumerate(SWEEP_CASES):
        r_i = idx // 3
        c_i = idx % 3
        ax = axes[r_i, c_i]

        p_arr = np.array(c_info["params"], dtype=np.float64)
        is_ss = c_info.get("is_steady_state", False)

        if is_ss:
            ic = find_steady_state(c_info["params"])
            ss_target = ic.copy()
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c_info["warmup"] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()
            ss_target = None

        steps = min(int(c_info["duration"] / DT), 10000)
        t_arr = np.linspace(0, steps * DT, steps)

        gt_tr = simulate_trajectory(ic, steps, DT, p_arr)
        pred_pure, _, _ = rollout_any(m_pure, ic, c_info["params"], steps, c_info["prominence"], c_info["min_dist"], mode="film")
        pred_naive, _, _ = rollout_any(m_naive, ic, c_info["params"], steps, c_info["prominence"], c_info["min_dist"], mode="film")
        pred_gated = rollout_gated_model(m_gated, ic, c_info["params"], steps, ss_target=ss_target)

        # Plot curves
        ax.plot(t_arr, gt_tr[:steps, 2], 'k-', lw=2.4, label="Ground Truth RK4")
        ax.plot(t_arr, pred_pure[:steps, 2], color='#2ca02c', linestyle='--', lw=1.3, alpha=0.75, label="Original FiLM Pure (Osc-Only)")
        ax.plot(t_arr, pred_naive[:steps, 2], color='#9467bd', linestyle='-.', lw=1.4, alpha=0.8, label="Naive Bifurcation FiLM (Damped X8/X11)")
        ax.plot(t_arr, pred_gated[:steps, 2], color='#d62728', linestyle='-', lw=2.0, label="Bifurcation-Gated FiLM (New Architecture, Seed 2026)")

        reg_badge = "Stable Focus (Flat Line)" if is_ss else "Sustained Limit Cycle"
        ax.set_title(f"{c_info['id']}: $k_a={c_info['params'][0]}$, $k_{{d2}}={c_info['params'][1]:.0f}$, $k_{{d3}}={c_info['params'][2]}$\n[{reg_badge}]", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(r"Polymer $m_1$ (mM)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Architectural Breakthrough: Bifurcation-Gated Neural ODE vs Naive FiLM across 12 Parameter Regimes (New Seed 2026)", fontsize=14, fontweight="bold", y=1.01)

    out_png1 = os.path.join(CURRENT_DIR, "bifurcation_gated_showcase_comparison.png")
    out_png2 = os.path.join(BRAIN_DIR, "bifurcation_gated_showcase_comparison.png")
    plt.savefig(out_png1, bbox_inches="tight", dpi=220)
    plt.savefig(out_png2, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Saved comparison figure to {out_png1} and {out_png2}")

if __name__ == "__main__":
    main()
