"""
======================================================================================
7-SEED SCIENTIFIC REPLICATION & ABLATION STUDY: BIOCHEMICAL POLYMERIZATION OSCILLATOR
======================================================================================
Performs rigorous 7-seed multi-seed training and benchmark evaluation across:
  - 7 Independent Seeds: [42, 101, 2024, 777, 999, 1234, 5555]
  - 4 Model Architectures:
      1. Baseline Unconstrained MLP (Generic Neural ODE)
      2. Divergence-Regularized MLP (Ablation: Phase-Space Volume Matching)
      3. Proposed FiLM Physics-Constrained SciML (Exact Conservation + Positivity)
      4. FiLM + Divergence Combined (Champion Model: Invariants + Volume Contraction)
  - 4 Dynamic Operating Regimes (25,000 steps / 2.5s horizon):
      Case A: Nominal Slide Benchmark (k_a=6.0, k_d2=4208, k_d3=1.0) [Interpolation]
      Case B: Slower High-Amplitude (k_a=3.0, k_d2=4200, k_d3=0.9) [Interpolation]
      Case C: Giant Amplitude Surge (k_a=5.0, k_d2=4000, k_d3=0.5) [Extrapolation]
      Case D: Fast Stiff Cycle (k_a=9.0, k_d2=4500, k_d3=0.6) [Extrapolation]

Outputs:
  - Checkpoints: ..._s{seed}.pt for each model and seed
  - Results JSON: biochemical_7seed_results.json
  - Publication Table 1: biochemical_7seed_table1.md (Mean +/- Std)
  - Multi-Seed Figure: biochemical_7seed_replication.png
======================================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP,
    build_dataset_with_ic_splits,
    simulate_trajectory,
    rk4_step,
    DEFAULT_GUESS,
    DT,
    set_seed
)

from train_all_biochemical_models import (
    FiLMWithRates,
    TRAINING_PARAMS,
    train_baseline,
    train_divergence_baseline,
    train_proposed_film,
    train_film_divergence_combined
)

DEFAULT_7_SEEDS = [42, 101, 2024, 777, 999, 1234, 5555]

REGIMES = [
    {
        "id": "Case A",
        "name": "Case A: Nominal Slide (Interp)",
        "params": (6.0, 4208.0, 1.0, 1.0),
        "type": "Interpolation",
        "duration": 2.5,
        "prominence": 10.0,
        "min_dist": 0.04
    },
    {
        "id": "Case B",
        "name": "Case B: Slower High-Amp (Interp)",
        "params": (3.0, 4200.0, 0.9, 1.0),
        "type": "Interpolation",
        "duration": 3.0,
        "prominence": 20.0,
        "min_dist": 0.10
    },
    {
        "id": "Case C",
        "name": "Case C: Giant Surge (Extrap)",
        "params": (5.0, 4000.0, 0.5, 1.0),
        "type": "Extrapolation",
        "duration": 2.5,
        "prominence": 30.0,
        "min_dist": 0.06
    },
    {
        "id": "Case D",
        "name": "Case D: Fast Stiff Cycle (Extrap)",
        "params": (9.0, 4500.0, 0.6, 1.0),
        "type": "Extrapolation",
        "duration": 2.5,
        "prominence": 3.0,
        "min_dist": 0.03
    }
]

MODELS_CONFIG = [
    {
        "key": "baseline",
        "name": "Baseline Unconstrained MLP",
        "short": "Baseline",
        "ckpt_pattern": "baseline_unconstrained_mlp_s{seed}.pt",
        "class": UnconstrainedBaselineMLP,
        "is_constrained": False,
        "color": "#dc2626"  # Red
    },
    {
        "key": "div",
        "name": "Divergence-Regularized MLP",
        "short": "Div-Only",
        "ckpt_pattern": "divergence_regularized_mlp_s{seed}.pt",
        "class": UnconstrainedBaselineMLP,
        "is_constrained": False,
        "color": "#d97706"  # Amber
    },
    {
        "key": "film",
        "name": "Proposed FiLM SciML",
        "short": "FiLM Pure",
        "ckpt_pattern": "best_proposed_biochemical_mlp_s{seed}.pt",
        "class": ProposedFiLMPhysicsConstrainedMLP,
        "is_constrained": True,
        "color": "#16a34a"  # Green
    },
    {
        "key": "film_div",
        "name": "FiLM + Divergence Combined",
        "short": "FiLM+Div",
        "ckpt_pattern": "film_divergence_regularized_mlp_s{seed}.pt",
        "class": FiLMWithRates,
        "is_constrained": True,
        "color": "#2563eb"  # Blue
    }
]

def analyze_waveform(signal: np.ndarray, prominence: float = 10.0, min_dist: float = 0.04, dt: float = DT) -> Dict[str, float]:
    """Extracts steady-state period, amplitude, and self-decay retention from signal."""
    dist_steps = max(int(min_dist / dt), 10)
    peaks, _ = find_peaks(signal, prominence=prominence, distance=dist_steps)
    troughs, _ = find_peaks(-signal, prominence=prominence, distance=dist_steps)

    if len(peaks) > 1:
        period = float(np.mean(np.diff(peaks)) * dt)
    else:
        period = 0.0

    n_pairs = min(len(peaks), len(troughs))
    if n_pairs > 0:
        amps_cycles = [signal[peaks[i]] - signal[troughs[i]] for i in range(n_pairs)]
        steady_amp = float(np.mean(amps_cycles[int(len(amps_cycles) * 0.5):])) if len(amps_cycles) > 1 else float(amps_cycles[0])
    else:
        steady_amp = float(np.ptp(signal))

    quarters = np.array_split(signal, 4)
    q1_amp = float(np.ptp(quarters[0]))
    q4_amp = float(np.ptp(quarters[3]))
    self_decay_pct = (q4_amp / max(q1_amp, 1e-6)) * 100.0

    return {
        "period": period,
        "amp": steady_amp,
        "q1_amp": q1_amp,
        "q4_amp": q4_amp,
        "self_decay_pct": self_decay_pct
    }

def train_seed_models(seed: int, retrain: bool = False, m0_floor: float = 0.5):
    """Trains all 4 models for a given seed if checkpoints are not present."""
    print(f"\n{'=' * 80}\n[SEED {seed}] CHECKING / TRAINING CHECKPOINTS\n{'=' * 80}")
    set_seed(seed)

    # 1. Dataset Generation for this seed
    train_ds = None
    def get_dataset():
        nonlocal train_ds
        if train_ds is None:
            print(f"Generating training dataset for seed {seed}...", flush=True)
            train_ds, _ = build_dataset_with_ic_splits(
                TRAINING_PARAMS, traj_duration=1.2, dt=DT, num_train_ics=4, num_val_ics=2
            )
        return train_ds

    # Checkpoint paths
    ckpt_base = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
    ckpt_div = os.path.join(CURRENT_DIR, f"divergence_regularized_mlp_s{seed}.pt")
    ckpt_film = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
    ckpt_film_div = os.path.join(CURRENT_DIR, f"film_divergence_regularized_mlp_s{seed}.pt")

    # Fallback paths for seed 42 default
    if seed == 42:
        if not os.path.exists(ckpt_base) and os.path.exists(os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")):
            ckpt_base = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
        if not os.path.exists(ckpt_div) and os.path.exists(os.path.join(CURRENT_DIR, "divergence_regularized_mlp.pt")):
            ckpt_div = os.path.join(CURRENT_DIR, "divergence_regularized_mlp.pt")
        if not os.path.exists(ckpt_film) and os.path.exists(os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")):
            ckpt_film = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
        if not os.path.exists(ckpt_film_div) and os.path.exists(os.path.join(CURRENT_DIR, "film_divergence_regularized_mlp.pt")):
            ckpt_film_div = os.path.join(CURRENT_DIR, "film_divergence_regularized_mlp.pt")

    # 1. Train Baseline
    if retrain or not os.path.exists(ckpt_base):
        print(f"Training Model 1 (Baseline, seed {seed})...")
        train_baseline(get_dataset(), total_steps=3000, lr=2e-3, k_max=25, seed=seed)
    else:
        print(f"Found existing Baseline: {os.path.basename(ckpt_base)}")

    # 2. Train Divergence Baseline (warm-started from same-seed baseline)
    if retrain or not os.path.exists(ckpt_div):
        print(f"Training Model 2 (Divergence, seed {seed})...")
        train_divergence_baseline(get_dataset(), total_steps=2000, lr=1e-3, alpha_div=1e-4, seed=seed, base_ckpt=ckpt_base)
    else:
        print(f"Found existing Divergence: {os.path.basename(ckpt_div)}")

    # 3. Train Proposed FiLM
    if retrain or not os.path.exists(ckpt_film):
        print(f"Training Model 3 (Proposed FiLM, seed {seed})...")
        train_proposed_film(get_dataset(), total_steps=3000, lr=2e-3, seed=seed, m0_floor=m0_floor)
    else:
        print(f"Found existing Proposed FiLM: {os.path.basename(ckpt_film)}")

    # 4. Train FiLM + Divergence (warm-started from same-seed FiLM)
    if retrain or not os.path.exists(ckpt_film_div):
        print(f"Training Model 4 (FiLM+Div, seed {seed})...")
        train_film_divergence_combined(get_dataset(), total_steps=600, lr=1.5e-4, alpha_div=1e-4, seed=seed, m0_floor=m0_floor, base_ckpt=ckpt_film)
    else:
        print(f"Found existing FiLM+Div: {os.path.basename(ckpt_film_div)}")

def evaluate_models_for_seed(seed: int, m0_floor: float = 0.5) -> Dict[str, Any]:
    """Loads models for the seed and performs 25,000-step rollouts across all 4 regimes."""
    print(f"\n--- Evaluating Models for Seed {seed} across 4 Operating Regimes ---")
    set_seed(seed)

    # Load all models
    models = {}
    for mcfg in MODELS_CONFIG:
        ckpt_name = mcfg["ckpt_pattern"].format(seed=seed)
        ckpt_path = os.path.join(CURRENT_DIR, ckpt_name)
        if not os.path.exists(ckpt_path) and seed == 42:
            default_map = {
                "baseline": "baseline_unconstrained_mlp.pt",
                "div": "divergence_regularized_mlp.pt",
                "film": "best_proposed_biochemical_mlp.pt",
                "film_div": "film_divergence_regularized_mlp.pt"
            }
            ckpt_path = os.path.join(CURRENT_DIR, default_map[mcfg["key"]])

        if mcfg["key"] in ["film", "film_div"]:
            model = mcfg["class"](hidden_dim=128, dt=DT, m0_floor=m0_floor)
        else:
            model = mcfg["class"](hidden_dim=128, dt=DT)

        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        models[mcfg["key"]] = model

    seed_results = {}
    for regime in REGIMES:
        reg_id = regime["id"]
        p = regime["params"]
        p_arr = np.array(p, dtype=np.float64)
        duration = regime["duration"]
        steps = int(duration / DT)
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

        # 1. Warm-up onto limit cycle
        y = DEFAULT_GUESS.copy()
        for _ in range(int(20.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        # 2. Simulate Ground Truth RK4
        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
        gt_metrics = analyze_waveform(gt_traj[:, 2], prominence=regime["prominence"], min_dist=regime["min_dist"])
        gt_period = gt_metrics["period"]
        gt_amp = gt_metrics["amp"]

        reg_data = {
            "gt": {
                "period": gt_period,
                "amp": gt_amp,
                "self_decay_pct": gt_metrics["self_decay_pct"],
                "mass_drift": 0.0,
                "mse": 0.0
            },
            "models": {}
        }

        # 3. Rollout each model
        for mcfg in MODELS_CONFIG:
            mkey = mcfg["key"]
            model = models[mkey]
            curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
            c_tot0 = curr[:, 0:1] + curr[:, 1:2] + curr[:, 2:3] if mcfg["is_constrained"] else None

            pred_traj = [ic.copy()]
            with torch.no_grad():
                for _ in range(steps):
                    if mcfg["is_constrained"]:
                        curr = model(curr, p_t, target_ctot=c_tot0)
                    else:
                        curr = model(curr, p_t)
                    pred_traj.append(curr.squeeze(0).cpu().numpy())
            pred_traj = np.array(pred_traj)

            m_metrics = analyze_waveform(pred_traj[:, 2], prominence=regime["prominence"], min_dist=regime["min_dist"])
            pred_period = m_metrics["period"]
            pred_amp = m_metrics["amp"]

            period_err_pct = abs(pred_period - gt_period) / gt_period * 100.0 if gt_period > 0 and pred_period > 0 else 100.0
            amp_err_pct = abs(pred_amp - gt_amp) / gt_amp * 100.0 if gt_amp > 0 and pred_amp > 0 else 100.0
            
            c_tot_pred = pred_traj[:, 0] + pred_traj[:, 1] + pred_traj[:, 2]
            mass_drift = float(np.ptp(c_tot_pred))
            mse = float(np.mean((pred_traj - gt_traj) ** 2))

            reg_data["models"][mkey] = {
                "period": pred_period,
                "period_err_pct": period_err_pct,
                "amp": pred_amp,
                "amp_err_pct": amp_err_pct,
                "self_decay_pct": m_metrics["self_decay_pct"],
                "mass_drift": mass_drift,
                "mse": mse
            }

        seed_results[reg_id] = reg_data
        print(f"  [{reg_id}] GT Amp={gt_amp:.1f}, T={gt_period:.4f}s | "
              f"FiLM T_err={reg_data['models']['film']['period_err_pct']:.1f}%, Amp_ret={reg_data['models']['film']['self_decay_pct']:.1f}% | "
              f"Base Amp_ret={reg_data['models']['baseline']['self_decay_pct']:.1f}%, Drift={reg_data['models']['baseline']['mass_drift']:.1f}mM")

    return seed_results

def compute_multiseed_statistics(all_results: Dict[int, Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    """Computes mean and std across seeds for each regime, model, and metric."""
    stats = {}
    for regime in REGIMES:
        reg_id = regime["id"]
        stats[reg_id] = {
            "name": regime["name"],
            "type": regime["type"],
            "gt_period": float(np.mean([all_results[s][reg_id]["gt"]["period"] for s in seeds])),
            "gt_amp": float(np.mean([all_results[s][reg_id]["gt"]["amp"] for s in seeds])),
            "models": {}
        }
        for mcfg in MODELS_CONFIG:
            mkey = mcfg["key"]
            metrics = {
                "period_err_pct": [],
                "amp_err_pct": [],
                "self_decay_pct": [],
                "mass_drift": [],
                "mse": []
            }
            for s in seeds:
                m_run = all_results[s][reg_id]["models"][mkey]
                for k in metrics:
                    metrics[k].append(m_run[k])

            stats[reg_id]["models"][mkey] = {
                "name": mcfg["name"],
                "short": mcfg["short"],
                "period_err_mean": float(np.mean(metrics["period_err_pct"])),
                "period_err_std": float(np.std(metrics["period_err_pct"])),
                "amp_err_mean": float(np.mean(metrics["amp_err_pct"])),
                "amp_err_std": float(np.std(metrics["amp_err_pct"])),
                "self_decay_mean": float(np.mean(metrics["self_decay_pct"])),
                "self_decay_std": float(np.std(metrics["self_decay_pct"])),
                "mass_drift_mean": float(np.mean(metrics["mass_drift"])),
                "mass_drift_std": float(np.std(metrics["mass_drift"])),
                "mse_mean": float(np.mean(metrics["mse"])),
                "mse_std": float(np.std(metrics["mse"]))
            }
    return stats

def generate_markdown_table(stats: Dict[str, Any], seeds: List[int]) -> str:
    """Formats statistical results into a paper-ready Markdown Table 1."""
    md = []
    md.append(f"# Table 1: 7-Seed Empirical Ablation & Dynamical Generalization ({len(seeds)} Seeds)\n")
    md.append(f"**Random Seeds**: `{seeds}`  \n")
    md.append("**Horizon**: 25,000 steps ($2.5\\text{s} - 3.0\\text{s}$) across 4 Diverse Operating Regimes.  \n")
    md.append("**Values**: Mean $\\pm$ Standard Deviation across independent training runs.\n\n")

    md.append("| Regime | Model | Period Error $\\Delta T$ (%) | Amp Error $\\Delta A$ (%) | Amplitude Retention (%) | Mass Drift $\\Delta c_{\\text{tot}}$ (mM) | Rollout MSE |\n")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

    for regime in REGIMES:
        reg_id = regime["id"]
        rstats = stats[reg_id]
        first_row = True
        for mcfg in MODELS_CONFIG:
            mkey = mcfg["key"]
            m_stat = rstats["models"][mkey]
            reg_label = f"**{reg_id}**<br>*{rstats['type']}*" if first_row else ""
            first_row = False

            p_err = f"{m_stat['period_err_mean']:.2f} ± {m_stat['period_err_std']:.2f}%"
            a_err = f"{m_stat['amp_err_mean']:.2f} ± {m_stat['amp_err_std']:.2f}%"
            ret = f"{m_stat['self_decay_mean']:.1f} ± {m_stat['self_decay_std']:.1f}%"
            drift = f"{m_stat['mass_drift_mean']:.4f} ± {m_stat['mass_drift_std']:.4f}" if mcfg["is_constrained"] else f"{m_stat['mass_drift_mean']:.1f} ± {m_stat['mass_drift_std']:.1f}"
            mse_val = f"{m_stat['mse_mean']:.2e} ± {m_stat['mse_std']:.2e}"

            bold_name = f"**{m_stat['short']}**" if "FiLM" in m_stat['short'] else m_stat['short']
            md.append(f"| {reg_label} | {bold_name} | {p_err} | {a_err} | {ret} | {drift} | {mse_val} |\n")
        md.append("| | | | | | | |\n")

    return "".join(md)

def plot_multiseed_figure(stats: Dict[str, Any], seeds: List[int], out_file: str = "biochemical_7seed_replication.png"):
    """Generates a publication-grade bar chart with error bars across regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    cases = [r["id"] for r in REGIMES]
    x = np.arange(len(cases))
    width = 0.20

    metrics_plot = [
        ("period_err", "Period Error ΔT (%)", axes[0, 0], False, [0, 60]),
        ("amp_err", "Amplitude Error ΔA (%)", axes[0, 1], False, [0, 80]),
        ("self_decay", "Amplitude Retention Q4/Q1 (%)", axes[1, 0], True, [40, 115]),
        ("mass_drift", "Stoichiometric Mass Drift Δc_tot (mM)", axes[1, 1], False, None)
    ]

    for metric_prefix, title, ax, has_target, y_lim in metrics_plot:
        for i, mcfg in enumerate(MODELS_CONFIG):
            mkey = mcfg["key"]
            means = [stats[cid]["models"][mkey][f"{metric_prefix}_mean"] for cid in cases]
            stds = [stats[cid]["models"][mkey][f"{metric_prefix}_std"] for cid in cases]
            pos = x + (i - 1.5) * width
            ax.bar(pos, means, width, yerr=stds, capsize=4, label=mcfg["short"], color=mcfg["color"], alpha=0.88, edgecolor="black", linewidth=0.8)

        if has_target:
            ax.axhline(100.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Ideal (100%)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{cid}\n({stats[cid]['type']})" for cid in cases], fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle=":")
        if y_lim:
            ax.set_ylim(y_lim)
        if metric_prefix == "mass_drift":
            ax.set_yscale("log")
            ax.set_ylabel("Log Scale (mM)", fontsize=10)
        else:
            ax.set_ylabel("Percentage (%)", fontsize=10)

    axes[0, 0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.suptitle(f"Multi-Seed Generalization Across 4 Stiff Regimes ({len(seeds)} Independent Training Seeds)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(CURRENT_DIR, out_file), dpi=300)
    plt.close()
    print(f"Saved publication figure to '{out_file}'", flush=True)

def main():
    parser = argparse.ArgumentParser(description="7-Seed Replication Suite for Biochemical SciML")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_7_SEEDS, help="List of seeds to run")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if checkpoints exist")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate existing checkpoints")
    parser.add_argument("--m0_floor", type=float, default=0.5, help="Pacemaker positivity floor (default: 0.5)")
    args = parser.parse_args()

    seeds = args.seeds
    print("=" * 80)
    print(f"LAUNCHING 7-SEED REPLICATION PIPELINE ACROSS {len(seeds)} SEEDS: {seeds}")
    print("=" * 80)

    # 1. Training Phase
    if not args.eval_only:
        for seed in seeds:
            t0_seed = time.time()
            train_seed_models(seed, retrain=args.retrain, m0_floor=args.m0_floor)
            print(f"[Seed {seed}] Training completed in {time.time()-t0_seed:.1f}s.", flush=True)

    # 2. Evaluation Phase
    all_results = {}
    for seed in seeds:
        all_results[seed] = evaluate_models_for_seed(seed, m0_floor=args.m0_floor)

    # 3. Statistical Aggregation
    stats = compute_multiseed_statistics(all_results, seeds)

    # 4. Save JSON Report
    json_path = os.path.join(CURRENT_DIR, "biochemical_7seed_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "stats": stats, "raw_runs": all_results}, f, indent=2)
    print(f"\nSaved raw multi-seed results to '{json_path}'.")

    # 5. Generate Markdown Table 1
    table_md = generate_markdown_table(stats, seeds)
    table_path = os.path.join(CURRENT_DIR, "biochemical_7seed_table1.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"Saved publication Table 1 to '{table_path}'.\n")
    print(table_md)

    # 6. Generate Publication Figure
    plot_multiseed_figure(stats, seeds, "biochemical_7seed_replication.png")

    print("\n" + "=" * 80)
    print(f"ALL {len(seeds)} SEEDS COMPLETED SUCCESSFULLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
