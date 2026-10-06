"""
======================================================================================
VAN DER POL 7-SEED MULTI-SEED REPLICATION & TABLE 1 BENCHMARK
======================================================================================
Evaluates the canonical 7 seeds: [42, 101, 777, 999, 1234, 2024, 5555]
Generates publication-standard Table 1 metrics at benchmark mu = 1.0.
======================================================================================
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import (
    ProposedFiLMVDPMLP, UnconstrainedBaselineVDPMLP,
    simulate_trajectory, measure_period_and_amplitude,
    build_vdp_dataset, train_model, set_seed, DT
)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
BENCHMARK_MU = 1.0
SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
os.makedirs(SCRATCH_DIR, exist_ok=True)

def compute_checksum(model):
    hasher = hashlib.sha256()
    for p in model.parameters():
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]

def main():
    print("=" * 100)
    print("VAN DER POL 7-SEED MULTI-SEED TRAINING & TABLE 1 BENCHMARK")
    print("=" * 100)

    # 1. Parameter Grid for Dataset
    param_grid = [-0.8, -0.4, -0.15, -0.05, 0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
    dataset = build_vdp_dataset(param_grid, traj_duration=50.0, dt=DT, num_ics_per_param=4)

    # 2. Ground Truth Reference via RK4 at mu = 1.0
    print("\nComputing Ground Truth RK4 Reference at mu = 1.0...")
    y_warm = np.array([2.0, 0.0], dtype=np.float64)
    # Warmup 40s (several cycles)
    for _ in range(int(40.0 / DT)):
        from vdp_global_sciml import rk4_step
        y_warm = rk4_step(y_warm, DT, BENCHMARK_MU)
    
    gt_steps = int(40.0 / DT)
    gt_traj = simulate_trajectory(y_warm, gt_steps, DT, BENCHMARK_MU)
    gt_period, gt_amp = measure_period_and_amplitude(gt_traj[:, 0], dt=DT)
    print(f"Ground Truth RK4: Period T = {gt_period:.4f} s | Limit-Cycle Amplitude = {gt_amp:.4f}")

    results = {
        "ground_truth": {
            "period": gt_period,
            "amplitude": gt_amp,
            "self_decay_retention": 100.0,
            "mu": BENCHMARK_MU
        },
        "proposed_seeds": {},
        "proposed_residual_s42": {},
        "baseline_unconstrained": {}
    }

    # 3. Train and Benchmark Proposed Model A (From-Scratch) across 7 Seeds
    print("\n" + "=" * 80)
    print("TRAINING PROPOSED FiLM SCI-ML (FROM-SCRATCH) ACROSS 7 CANONICAL SEEDS")
    print("=" * 80)

    for s in SEEDS:
        print(f"\n--- Training Seed {s} ---")
        ckpt_path = os.path.join(SCRATCH_DIR, f"vdp_sciml_fromscratch_s{s}.pt")
        model = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
        
        t0 = time.time()
        train_model(model, dataset, epochs=40, batch_size=32, lr=2e-3, seed=s)
        train_time = time.time() - t0
        print(f"Seed {s} trained in {train_time:.1f} s")
        torch.save(model.state_dict(), ckpt_path)

        # Rollout benchmark
        model.eval()
        p_t = torch.tensor([[BENCHMARK_MU]], dtype=torch.float32)
        curr = torch.tensor([y_warm], dtype=torch.float32)
        
        pred_x = [y_warm[0]]
        with torch.no_grad():
            for _ in range(gt_steps):
                curr = model.rk4_step_forward(curr, p_t)
                pred_x.append(curr[0, 0].item())
        pred_x = np.array(pred_x)

        period, amp = measure_period_and_amplitude(pred_x, dt=DT)
        
        # Self-decay retention: check amplitude over second half vs first half of asymptotic window
        n_half = len(pred_x) // 2
        amp_q3 = float(np.ptp(pred_x[n_half:n_half + n_half // 2]))
        amp_q4 = float(np.ptp(pred_x[n_half + n_half // 2:]))
        retention = (amp_q4 / amp_q3 * 100.0) if amp_q3 > 0.1 else 0.0
        amp_pct = (amp / gt_amp * 100.0) if gt_amp > 0 else 0.0
        checksum = compute_checksum(model)

        results["proposed_seeds"][str(s)] = {
            "checksum": checksum,
            "period": period,
            "amplitude": amp,
            "retention": retention,
            "amp_pct": amp_pct
        }
        print(f"Seed {s:4d} | Checksum: {checksum} | Period: {period:.4f} s | Amp: {amp:.4f} ({amp_pct:.2f}%) | Ret: {retention:.2f}%")

    # 4. Train Model B (Physics-Residual) Seed 42 for Controlled Dichotomy
    print("\n" + "=" * 80)
    print("TRAINING PROPOSED FiLM SCI-ML (PHYSICS-RESIDUAL) SEED 42")
    print("=" * 80)
    model_res = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=True)
    train_model(model_res, dataset, epochs=40, batch_size=32, lr=2e-3, seed=42)
    torch.save(model_res.state_dict(), os.path.join(SCRATCH_DIR, "vdp_sciml_residual_s42.pt"))
    
    model_res.eval()
    p_t = torch.tensor([[BENCHMARK_MU]], dtype=torch.float32)
    curr = torch.tensor([y_warm], dtype=torch.float32)
    pred_res_x = [y_warm[0]]
    with torch.no_grad():
        for _ in range(gt_steps):
            curr = model_res.rk4_step_forward(curr, p_t)
            pred_res_x.append(curr[0, 0].item())
    pred_res_x = np.array(pred_res_x)
    p_res, a_res = measure_period_and_amplitude(pred_res_x, dt=DT)
    n_h = len(pred_res_x) // 2
    ret_res = (np.ptp(pred_res_x[n_h + n_h // 2:]) / np.ptp(pred_res_x[n_h:n_h + n_h // 2])) * 100.0
    results["proposed_residual_s42"] = {
        "checksum": compute_checksum(model_res),
        "period": p_res,
        "amplitude": a_res,
        "retention": ret_res,
        "amp_pct": (a_res / gt_amp * 100.0)
    }

    # 5. Train Unconstrained Baseline Generic MLP (Seed 42)
    print("\n" + "=" * 80)
    print("TRAINING UNCONSTRAINED BASELINE GENERIC MLP (SEED 42)")
    print("=" * 80)
    base_model = UnconstrainedBaselineVDPMLP(hidden_dim=64, dt=DT)
    train_model(base_model, dataset, epochs=40, batch_size=32, lr=2e-3, seed=42)
    torch.save(base_model.state_dict(), os.path.join(SCRATCH_DIR, "vdp_baseline_s42.pt"))

    base_model.eval()
    curr = torch.tensor([y_warm], dtype=torch.float32)
    pred_base_x = [y_warm[0]]
    with torch.no_grad():
        for _ in range(gt_steps):
            curr = base_model.rk4_step_forward(curr, p_t)
            pred_base_x.append(curr[0, 0].item())
    pred_base_x = np.array(pred_base_x)
    p_base, a_base = measure_period_and_amplitude(pred_base_x, dt=DT)
    n_h = len(pred_base_x) // 2
    ret_base = (np.ptp(pred_base_x[n_h + n_h // 2:]) / np.ptp(pred_base_x[n_h:n_h + n_h // 2])) * 100.0
    results["baseline_unconstrained"] = {
        "checksum": compute_checksum(base_model),
        "period": p_base,
        "amplitude": a_base,
        "retention": ret_base,
        "amp_pct": (a_base / gt_amp * 100.0)
    }

    # Summary Statistics for Model A
    periods = [v["period"] for v in results["proposed_seeds"].values()]
    amps = [v["amplitude"] for v in results["proposed_seeds"].values()]
    rets = [v["retention"] for v in results["proposed_seeds"].values()]
    amp_pcts = [v["amp_pct"] for v in results["proposed_seeds"].values()]

    results["ensemble_summary"] = {
        "period_mean": float(np.mean(periods)),
        "period_std": float(np.std(periods)),
        "amp_mean": float(np.mean(amps)),
        "amp_std": float(np.std(amps)),
        "retention_mean": float(np.mean(rets)),
        "retention_std": float(np.std(rets)),
        "amp_pct_mean": float(np.mean(amp_pcts)),
        "amp_pct_std": float(np.std(amp_pcts))
    }

    # Print Table 1
    print("\n" + "=" * 100)
    print("PUBLICATION TABLE 1: VAN DER POL 7-SEED MULTI-SEED REPLICATION BENCHMARK")
    print("=" * 100)
    print(f"Ground Truth RK4 Reference: Period T = {gt_period:.4f} s | Amplitude = {gt_amp:.4f} | Retention = 100.00%")
    print("-" * 100)
    print(f"{'Model / Architecture':26s} | {'Seed':5s} | {'Checksum':16s} | {'Period T (s)':12s} | {'Amplitude':10s} | {'Retention (%)':13s} | {'Amp vs GT (%)':13s}")
    print("-" * 100)
    for s in SEEDS:
        v = results["proposed_seeds"][str(s)]
        print(f"{'FiLM From-Scratch (Model A)':26s} | {s:5d} | {v['checksum']:16s} | {v['period']:12.4f} | {v['amplitude']:10.4f} | {v['retention']:13.2f} | {v['amp_pct']:13.2f}")
    print("-" * 100)
    summ = results["ensemble_summary"]
    print(f"{'Ensemble Mean (Model A)':26s} | {'--':5s} | {'--':16s} | {summ['period_mean']:12.4f} | {summ['amp_mean']:10.4f} | {summ['retention_mean']:13.2f} | {summ['amp_pct_mean']:13.2f}")
    print(f"{'Ensemble Std (Model A)':26s}  | {'--':5s} | {'--':16s} | {summ['period_std']:12.4f} | {summ['amp_std']:10.4f} | {summ['retention_std']:13.2f} | {summ['amp_pct_std']:13.2f}")
    print("-" * 100)
    res_v = results["proposed_residual_s42"]
    print(f"{'FiLM Residual (Model B)':26s} | {42:5d} | {res_v['checksum']:16s} | {res_v['period']:12.4f} | {res_v['amplitude']:10.4f} | {res_v['retention']:13.2f} | {res_v['amp_pct']:13.2f}")
    base_v = results["baseline_unconstrained"]
    print(f"{'Unconstrained Baseline MLP':26s} | {42:5d} | {base_v['checksum']:16s} | {base_v['period']:12.4f} | {base_v['amplitude']:10.4f} | {base_v['retention']:13.2f} | {base_v['amp_pct']:13.2f}")
    print("=" * 100)

    # Save to JSON
    json_path = os.path.join(SCRATCH_DIR, "vdp_table1_replication.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

if __name__ == "__main__":
    main()
