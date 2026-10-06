"""
======================================================================================
EXTENDED PARAMETER SWEEP: DIVERSE ka, kd2, kd3 (INCLUDING ka > 10)
======================================================================================
Evaluates generalization across 7 diverse regimes:
1. ka = 11.0, kd2 = 5000.0, kd3 = 0.3  (Deep extrapolation, ka > 10, high kd2, low kd3)
2. ka = 12.0, kd2 = 4500.0, kd3 = 0.5  (Deep extrapolation, ka = 12, moderate kd2, kd3)
3. ka = 14.0, kd2 = 5000.0, kd3 = 0.3  (Extreme extrapolation, ka = 14, stiff damped)
4. ka = 10.5, kd2 = 4600.0, kd3 = 0.6  (Boundary extrapolation, ka = 10.5)
5. ka = 7.0,  kd2 = 5000.0, kd3 = 0.4  (High kd2 = 5000, sustained large-amplitude cycle)
6. ka = 4.5,  kd2 = 3500.0, kd3 = 1.3  (High kd3 = 1.3, low kd2 = 3500)
7. ka = 2.5,  kd2 = 3200.0, kd3 = 0.8  (Low ka = 2.5, slow relaxation cycle)
======================================================================================
"""

import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\c0229372-9c2c-4e39-b054-0c23ec47512a"
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
from train_all_biochemical_models import FiLMWithRates
from run_complete_evaluation_suite import analyze_signal

SEEDS = [42, 101, 2024, 777, 999, 1234, 5555]

EXTENDED_REGIMES = [
    {
        "id": "E1",
        "name": "E1: Deep Extrapolation (ka=11.0, kd2=5000, kd3=0.3)",
        "params": (11.0, 5000.0, 0.3, 1.0),
        "duration": 2.0,
        "warmup": 0.2,
        "prominence": 5.0,
        "min_dist": 0.03,
        "is_damped": True
    },
    {
        "id": "E2",
        "name": "E2: Deep Extrapolation (ka=12.0, kd2=4500, kd3=0.5)",
        "params": (12.0, 4500.0, 0.5, 1.0),
        "duration": 2.0,
        "warmup": 0.2,
        "prominence": 5.0,
        "min_dist": 0.03,
        "is_damped": True
    },
    {
        "id": "E3",
        "name": "E3: Extreme Stiff Extrapolation (ka=14.0, kd2=5000, kd3=0.3)",
        "params": (14.0, 5000.0, 0.3, 1.0),
        "duration": 2.0,
        "warmup": 0.2,
        "prominence": 3.0,
        "min_dist": 0.03,
        "is_damped": True
    },
    {
        "id": "E4",
        "name": "E4: Transition Extrapolation (ka=10.5, kd2=4600, kd3=0.6)",
        "params": (10.5, 4600.0, 0.6, 1.0),
        "duration": 2.0,
        "warmup": 0.2,
        "prominence": 5.0,
        "min_dist": 0.03,
        "is_damped": True
    },
    {
        "id": "E5",
        "name": "E5: High kd2 Sustained Surge (ka=7.0, kd2=5000, kd3=0.4)",
        "params": (7.0, 5000.0, 0.4, 1.0),
        "duration": 2.5,
        "warmup": 5.0,
        "prominence": 20.0,
        "min_dist": 0.04,
        "is_damped": False
    },
    {
        "id": "E6",
        "name": "E6: High kd3 Moderate Cycle (ka=4.5, kd2=3500, kd3=1.3)",
        "params": (4.5, 3500.0, 1.3, 1.0),
        "duration": 2.5,
        "warmup": 5.0,
        "prominence": 10.0,
        "min_dist": 0.04,
        "is_damped": False
    },
    {
        "id": "E7",
        "name": "E7: Low ka Relaxation Cycle (ka=2.5, kd2=3200, kd3=0.8)",
        "params": (2.5, 3200.0, 0.8, 1.0),
        "duration": 3.0,
        "warmup": 5.0,
        "prominence": 20.0,
        "min_dist": 0.08,
        "is_damped": False
    }
]

def rollout_any(model, ic, params, steps, prominence, min_dist, mode="film"):
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]

    pred = [ic.copy()]
    with torch.no_grad():
        for _ in range(steps):
            if mode == "film":
                curr = model(curr, p_t, target_ctot=c_tot)
            elif mode == "baseline":
                curr = model(curr, p_t)
            elif mode == "projected_baseline":
                curr = model(curr, p_t)
                curr = torch.clamp(curr, min=0.0)
                sub_sum = curr[:, 0] + curr[:, 1] + curr[:, 2]
                scale = c_tot / torch.clamp(sub_sum, min=1e-6)
                curr[:, 0:3] = curr[:, 0:3] * scale
            pred.append(curr.squeeze(0).cpu().numpy())
    pred = np.array(pred)

    m = analyze_signal(pred[:, 2], prominence=prominence, min_dist=min_dist)
    drift = float(np.ptp(pred[:, 0] + pred[:, 1] + pred[:, 2]))
    return pred, m, drift

def main():
    print("=" * 80)
    print("STARTING EXTENDED PARAMETER GENERALIZATION STUDY (ka > 10, DIVERSE kd2 & kd3)")
    print("=" * 80)

    models_to_test = [
        "FiLM Pure",
        "FiLM FT (alpha=0)",
        "FiLM+Div",
        "Projected Baseline",
        "Baseline MLP (Unconstrained)"
    ]

    results_table = {reg["id"]: {} for reg in EXTENDED_REGIMES}

    for reg in EXTENDED_REGIMES:
        p_arr = np.array(reg["params"], dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(reg["warmup"] / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        steps = int(reg["duration"] / DT)
        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
        gt_m = analyze_signal(gt_traj[:, 2], prominence=reg["prominence"], min_dist=reg["min_dist"])

        print(f"\nEvaluating {reg['name']}...")
        print(f"  GT: Period={gt_m['period']:.4f}s, Amp/Decay={gt_m['decay_ratio'] if reg['is_damped'] else gt_m['steady_amp']:.2f}")

        reg_dict = {
            "gt_period": gt_m["period"],
            "gt_amp_or_decay": gt_m["decay_ratio"] if reg["is_damped"] else gt_m["steady_amp"],
            "is_damped": reg["is_damped"],
            "models": {}
        }

        for mname in models_to_test:
            p_errs, ratios, drifts = [], [], []

            for seed in SEEDS:
                if mname == "FiLM Pure":
                    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
                    ckpt = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
                    if not os.path.exists(ckpt) and seed == 42:
                        ckpt = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model.eval()
                    mode = "film"

                elif mname == "FiLM FT (alpha=0)":
                    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
                    ckpt = os.path.join(CURRENT_DIR, f"film_finetuned_nodiv_s{seed}.pt")
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model.eval()
                    mode = "film"

                elif mname == "FiLM+Div":
                    model = FiLMWithRates(hidden_dim=128, dt=DT, m0_floor=0.5)
                    ckpt = os.path.join(CURRENT_DIR, f"film_divergence_regularized_mlp_s{seed}.pt")
                    if not os.path.exists(ckpt) and seed == 42:
                        ckpt = os.path.join(CURRENT_DIR, "film_divergence_regularized_mlp.pt")
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model.eval()
                    mode = "film"

                elif mname == "Projected Baseline":
                    model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
                    ckpt = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
                    if not os.path.exists(ckpt) and seed == 42:
                        ckpt = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model.eval()
                    mode = "projected_baseline"

                elif mname == "Baseline MLP (Unconstrained)":
                    model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
                    ckpt = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
                    if not os.path.exists(ckpt) and seed == 42:
                        ckpt = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model.eval()
                    mode = "baseline"

                pred, m, drift = rollout_any(model, ic, reg["params"], steps, reg["prominence"], reg["min_dist"], mode=mode)
                p_err = abs(m["period"] - gt_m["period"]) / gt_m["period"] * 100.0 if gt_m["period"] > 0 and m["period"] > 0 else 100.0
                if reg["is_damped"]:
                    ratio = m["decay_ratio"] / max(gt_m["decay_ratio"], 1e-6)
                else:
                    ratio = m["steady_amp"] / max(gt_m["steady_amp"], 1e-6)

                p_errs.append(p_err)
                ratios.append(ratio)
                drifts.append(drift)

            reg_dict["models"][mname] = {
                "p_err_mean": float(np.mean(p_errs)),
                "p_err_std": float(np.std(p_errs)),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
                "drift_mean": float(np.mean(drifts)),
                "drift_std": float(np.std(drifts))
            }
            drift_fmt = f"{np.mean(drifts):.4f}mM" if np.mean(drifts) < 1.0 else f"{np.mean(drifts):.1f}mM"
            print(f"    {mname:28s} | P_Err: {np.mean(p_errs):5.2f} ± {np.std(p_errs):5.2f}% | Ratio: {np.mean(ratios):5.3f} ± {np.std(ratios):5.3f} | Drift: {drift_fmt}")

        results_table[reg["id"]] = reg_dict

    # Format Markdown Report
    lines = []
    lines.append("# Extended Parameter Sweep Across Deep Extrapolation ($k_a > 10$, Diverse $k_{d2}$, $k_{d3}$)\n\n")
    lines.append("> **Experimental Overview**:\n")
    lines.append("> - **Deep Extrapolation ($k_a > 10$)**: Evaluates system behavior well past the nominal training upper bound ($k_a = 8.0$) and beyond the Hopf bifurcation boundary ($k_a \\approx 7.1 - 8.2$).\n")
    lines.append("> - **Diverse $k_{d2}, k_{d3}$ Combinations**: Assesses robustness across high and low binding rates ($k_{d2} \\in [3200, 5000]$) and activation rates ($k_{d3} \\in [0.3, 1.3]$).\n")
    lines.append("> - **Multi-Seed Replication**: All metrics computed as `Mean ± Std` across all 7 canonical seeds (`[42, 101, 2024, 777, 999, 1234, 5555]`).\n\n")

    for reg in EXTENDED_REGIMES:
        rid = reg["id"]
        rd = results_table[rid]
        c_label = "Decay Match ($D_{\\text{model}}/D_{\\text{true}}$)" if rd["is_damped"] else "Amp Ratio ($A_{\\text{model}}/A_{\\text{true}}$)"
        reg_type = "Damped Stiff Extrapolation" if rd["is_damped"] else "Sustained Limit Cycle"
        lines.append(f"### {reg['name']} — [{reg_type}]\n")
        lines.append(f"- **Ground Truth**: Period $T = {rd['gt_period']:.4f}$ s, Target Metric $= {rd['gt_amp_or_decay']:.2f}$\n\n")
        lines.append(f"| Model | Period Error (%) | {c_label} | Mass Drift (mM) |\n")
        lines.append("| :--- | :---: | :---: | :---: |\n")

        for mname in models_to_test:
            d = rd["models"][mname]
            drift_str = f"{d['drift_mean']:.4f} ± {d['drift_std']:.4f}" if "FiLM" in mname or "Projected" in mname else f"{d['drift_mean']:.1f} ± {d['drift_std']:.1f}"
            bold_m = f"**{mname}**" if "FiLM" in mname else mname
            lines.append(f"| {bold_m} | {d['p_err_mean']:.2f} ± {d['p_err_std']:.2f}% | {d['ratio_mean']:.3f} ± {d['ratio_std']:.3f} | {drift_str} |\n")
        lines.append("\n---\n\n")

    out_file = os.path.join(CURRENT_DIR, "extended_parameter_sweep_report.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\nSaved extended parameter sweep report to '{out_file}'.")

    # Generate a visual comparison figure for Seed 42 across ka > 10
    fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), dpi=250)
    plt.subplots_adjust(hspace=0.35, wspace=0.20)

    # Plot E1 (ka=11), E2 (ka=12), E3 (ka=14)
    plot_subsets = [
        (EXTENDED_REGIMES[0], axes[0, 0], axes[0, 1]),
        (EXTENDED_REGIMES[1], axes[1, 0], axes[1, 1]),
        (EXTENDED_REGIMES[2], axes[2, 0], axes[2, 1])
    ]

    # Load seed 42 models
    m_film = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_film.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt"), map_location="cpu"))
    m_film.eval()

    m_base = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    m_base.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt"), map_location="cpu"))
    m_base.eval()

    for reg_info, ax_t, ax_p in plot_subsets:
        p_arr = np.array(reg_info["params"], dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(reg_info["warmup"] / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        steps = int(reg_info["duration"] / DT)
        n_pts = min(steps, 15000)
        t_arr = np.linspace(0, n_pts * DT, n_pts)

        gt_tr = simulate_trajectory(ic, steps, DT, p_arr)
        pred_f, _, _ = rollout_any(m_film, ic, reg_info["params"], steps, reg_info["prominence"], reg_info["min_dist"], mode="film")
        pred_p, _, _ = rollout_any(m_base, ic, reg_info["params"], steps, reg_info["prominence"], reg_info["min_dist"], mode="projected_baseline")
        pred_b, _, _ = rollout_any(m_base, ic, reg_info["params"], steps, reg_info["prominence"], reg_info["min_dist"], mode="baseline")

        # Time series
        ax_t.plot(t_arr, gt_tr[:n_pts, 2], 'k-', lw=2.0, label="Ground Truth (RK4)")
        ax_t.plot(t_arr, pred_f[:n_pts, 2], 'g--', lw=1.6, label="Proposed FiLM")
        ax_t.plot(t_arr, pred_p[:n_pts, 2], 'b-.', lw=1.2, label="Projected Baseline")
        ax_t.plot(t_arr, pred_b[:n_pts, 2], 'r:', lw=1.0, alpha=0.7, label="Vanilla Baseline")
        ax_t.set_title(f"{reg_info['name']}\nTime Series Evolution", fontsize=10, fontweight="bold")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel(r"$m_1$ (mM)")
        ax_t.grid(True, linestyle=":", alpha=0.5)

        # Phase portrait
        ax_p.plot(gt_tr[5000:n_pts, 3], gt_tr[5000:n_pts, 2], 'k-', lw=2.0, label="Ground Truth")
        ax_p.plot(pred_f[5000:n_pts, 3], pred_f[5000:n_pts, 2], 'g--', lw=1.6, label="Proposed FiLM")
        ax_p.plot(pred_p[5000:n_pts, 3], pred_p[5000:n_pts, 2], 'b-.', lw=1.2, label="Projected Baseline")
        ax_p.plot(pred_b[5000:n_pts, 3], pred_b[5000:n_pts, 2], 'r:', lw=1.0, alpha=0.7, label="Vanilla Baseline")
        ax_p.set_title(f"Orbital Geometry ($m_0$ vs $m_1$)", fontsize=10, fontweight="bold")
        ax_p.set_xlabel(r"Pacemaker $m_0$ (mM)")
        ax_p.set_ylabel(r"Polymer $m_1$ (mM)")
        ax_p.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Deep Extrapolation Grid Across $k_a > 10$ Regimes (Seed 42)", fontsize=13, fontweight="bold", y=1.01)

    out_img = os.path.join(CURRENT_DIR, "deep_extrapolation_ka_sweep.png")
    brain_img = os.path.join(BRAIN_DIR, "deep_extrapolation_ka_sweep.png")
    plt.tight_layout()
    plt.savefig(out_img, dpi=250, bbox_inches="tight")
    plt.close()
    import shutil
    shutil.copyfile(out_img, brain_img)
    print(f"Saved visual extrapolation grid to:\n  - {out_img}\n  - {brain_img}")

if __name__ == "__main__":
    main()
