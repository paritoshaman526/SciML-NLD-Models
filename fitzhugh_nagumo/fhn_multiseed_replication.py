"""
================================================================================
TIER 2 SCIENTIFIC VALIDATION: MULTI-SEED REPLICATION & ATTRACTOR STABILITY
FOR THE FITZHUGH-NAGUMO NEURONAL MODEL (TABLE 1)
================================================================================
Conducts rigorous multi-seed evaluation across 7 independent seeds:
  [42, 101, 777, 999, 1234, 2024, 5555]
plus the Unconstrained Baseline SciML model.

Evaluates:
  1. Retention / Self-Decay Ratio (%): Q4 PTP / Q1 PTP (internal numerical damping)
  2. Amplitude-vs-Truth Ratio (%): Model Q4 PTP / Ground Truth Q4 PTP
  3. Period accuracy (s) and relative error (%)
  4. SHA-256 model checkpoint integrity verification
================================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import hashlib
import json
import numpy as np
import torch
from scipy.signal import find_peaks

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fhn_global_sciml import (
    ProposedFiLMFHNMLP,
    UnconstrainedBaselineFHNMLP,
    simulate_trajectory,
    rk4_step,
    measure_period_and_amplitude,
    analytical_hopf_currents,
    DT
)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]


def get_checkpoint_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]


def simulate_model_trajectory(model, y0, steps, params):
    traj = np.empty((steps + 1, 2), dtype=np.float64)
    traj[0] = y0
    curr = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

    model.eval()
    step_fn = model.rk4_step_forward if hasattr(model, 'rk4_step_forward') else model
    with torch.no_grad():
        for i in range(steps):
            curr = step_fn(curr, p_t)
            traj[i + 1] = curr.squeeze(0).cpu().numpy()
    return traj



def compute_cycle_metrics(traj_v, dt=DT):
    peaks, _ = find_peaks(traj_v, distance=int(10.0 / dt), prominence=0.5)
    troughs, _ = find_peaks(-traj_v, distance=int(10.0 / dt), prominence=0.5)
    n_pairs = min(len(peaks), len(troughs))
    ptp_per_cycle = [traj_v[peaks[i]] - traj_v[troughs[i]] for i in range(n_pairs)]
    ret_cycle = (ptp_per_cycle[-1] / ptp_per_cycle[0] * 100.0) if len(ptp_per_cycle) > 0 else 0.0

    quarters = np.array_split(traj_v, 4)
    amps = [np.ptp(q) for q in quarters]
    ret_quarter = amps[-1] / max(amps[0], 1e-6) * 100.0

    period = float(np.mean(np.diff(peaks)) * dt) if len(peaks) > 1 else 0.0
    return {
        "peaks": len(peaks),
        "period": period,
        "q1_amp": float(amps[0]),
        "q4_amp": float(amps[-1]),
        "ret_quarter": float(ret_quarter),
        "ret_cycle": float(ret_cycle),
        "min": float(traj_v.min()),
        "max": float(traj_v.max())
    }


def main():
    print("=" * 115, flush=True)
    print("TIER 2 SCIENTIFIC VALIDATION: 7-SEED REPLICATION FOR FITZHUGH-NAGUMO SciML", flush=True)
    print("=" * 115, flush=True)

    scratch_dir = os.path.join(CURRENT_DIR, "scratch")

    # Benchmark parameter set (Standard canonical FitzHugh-Nagumo):
    # I = 0.8, a = 0.7, b = 0.8, eps = 0.08
    params = np.array([0.8, 0.7, 0.8, 0.08], dtype=np.float64)
    I1_h, I2_h = analytical_hopf_currents(0.7, 0.8, 0.08)
    print(f"Benchmark Regime: I={params[0]:.2f}, a={params[1]:.2f}, b={params[2]:.2f}, eps={params[3]:.2f}")
    print(f"Analytical Hopf Range: I in [{I1_h:.4f}, {I2_h:.4f}] (Limit-Cycle Spiking Regime)\n", flush=True)

    # 1. Warm-up 100s to locate steady limit cycle for ground truth
    y_warm = np.array([-1.0, -0.5], dtype=np.float64)
    for _ in range(int(100.0 / DT)):
        y_warm = rk4_step(y_warm, DT, params)
    base_ic = y_warm.copy()

    n_rollout = 3000  # 150 seconds rollout (~4 full cycles)
    print(f"Simulating Ground Truth RK4 Reference from limit cycle IC ({n_rollout} steps = {n_rollout*DT:.1f}s)...", flush=True)
    gt_traj = simulate_trajectory(base_ic, n_rollout, DT, params)
    m_gt = compute_cycle_metrics(gt_traj[:, 0], DT)
    print(f"  Ground Truth RK4: T = {m_gt['period']:.4f}s | Q4 Amp = {m_gt['q4_amp']:.4f} | Retention = {m_gt['ret_quarter']:.2f}%\n", flush=True)

    # 2. Evaluate Proposed FiLM SciML Across 7 Seeds
    seed_metrics = []
    print(f"{'Model / Seed':<20} | {'SHA-256':<12} | {'Period T (s)':<12} | {'Q4 Amp':<10} | {'Self-Decay %':<14} | {'Amp vs GT %':<12}")
    print("-" * 95)

    for seed in SEEDS:
        ckpt_path = os.path.join(scratch_dir, f"fhn_sciml_s{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Waiting/Missing: {ckpt_path}")
            continue
        sha = get_checkpoint_sha256(ckpt_path)
        model = ProposedFiLMFHNMLP(hidden_dim=128, dt=DT)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        pred_traj = simulate_model_trajectory(model, base_ic, n_rollout, params)
        m = compute_cycle_metrics(pred_traj[:, 0], DT)
        amp_vs_gt = (m["q4_amp"] / m_gt["q4_amp"]) * 100.0

        seed_metrics.append({
            "seed": seed,
            "sha": sha,
            "period": m["period"],
            "q4_amp": m["q4_amp"],
            "ret_quarter": m["ret_quarter"],
            "amp_vs_gt": amp_vs_gt
        })
        print(f"FiLM SciML (s={seed:<4d}) | {sha:<12} | {m['period']:<12.4f} | {m['q4_amp']:<10.4f} | {m['ret_quarter']:<14.2f} | {amp_vs_gt:<12.2f}")

    # Aggregate Statistics
    if seed_metrics:
        periods = [item["period"] for item in seed_metrics]
        q4_amps = [item["q4_amp"] for item in seed_metrics]
        retentions = [item["ret_quarter"] for item in seed_metrics]
        amp_gts = [item["amp_vs_gt"] for item in seed_metrics]

        print("-" * 95)
        print(f"{'Proposed Mean':<20} | {'--':<12} | {np.mean(periods):<12.4f} | {np.mean(q4_amps):<10.4f} | {np.mean(retentions):<14.2f} | {np.mean(amp_gts):<12.2f}")
        print(f"{'Proposed Std':<20} | {'--':<12} | {np.std(periods):<12.4f} | {np.std(q4_amps):<10.4f} | {np.std(retentions):<14.2f} | {np.std(amp_gts):<12.2f}")

    # 3. Evaluate Unconstrained Baseline (Seed 42)
    base_ckpt = os.path.join(scratch_dir, "fhn_baseline_s42.pt")
    if os.path.exists(base_ckpt):
        sha_base = get_checkpoint_sha256(base_ckpt)
        base_model = UnconstrainedBaselineFHNMLP(hidden_dim=128, dt=DT)
        base_model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
        base_model.eval()

        pred_base = simulate_model_trajectory(base_model, base_ic, n_rollout, params)
        m_b = compute_cycle_metrics(pred_base[:, 0], DT)
        amp_vs_gt_b = (m_b["q4_amp"] / m_gt["q4_amp"]) * 100.0
        print("-" * 95)
        print(f"{'Unconstrained MLP':<20} | {sha_base:<12} | {m_b['period']:<12.4f} | {m_b['q4_amp']:<10.4f} | {m_b['ret_quarter']:<14.2f} | {amp_vs_gt_b:<12.2f}")

    print("=" * 115, flush=True)

    # Save metrics JSON
    out_json = os.path.join(scratch_dir, "table1_fhn_replication.json")
    with open(out_json, "w") as f:
        json.dump({
            "gt_reference": m_gt,
            "proposed_seeds": seed_metrics,
            "proposed_mean": {
                "period": float(np.mean(periods)),
                "q4_amp": float(np.mean(q4_amps)),
                "ret_quarter": float(np.mean(retentions)),
                "amp_vs_gt": float(np.mean(amp_gts))
            },
            "proposed_std": {
                "period": float(np.std(periods)),
                "q4_amp": float(np.std(q4_amps)),
                "ret_quarter": float(np.std(retentions)),
                "amp_vs_gt": float(np.std(amp_gts))
            }
        }, f, indent=2)
    print(f"Results saved to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
