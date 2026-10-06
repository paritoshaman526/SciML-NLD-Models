"""
======================================================================================
PROTOCOL EXTENSIONS & RIGOROUS COMPONENT ABLATIONS
======================================================================================
1. FiLM Fine-Tuning with alpha_div = 0 across all 7 seeds (same 600 steps, lr=1.5e-4, K=10->30)
2. Per-Seed Table for Case B and Case D across all models
3. Learned Hopf Bifurcation Boundary from Neural vs True Jacobian Eigenvalues
4. Pacemaker m0 Floor Sensitivity Sweep & Single-Component Ablations
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
from typing import Dict, List, Any, Tuple
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
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
    system,
    DEFAULT_GUESS,
    DT,
    set_seed
)

from train_all_biochemical_models import (
    FiLMWithRates,
    TRAINING_PARAMS,
    sample_batch
)

SEEDS = [42, 101, 2024, 777, 999, 1234, 5555]

CASES = {
    "Case B": {
        "name": "Case B: Slower High-Amp (Interp)",
        "params": (3.0, 4200.0, 0.9, 1.0),
        "duration": 3.0,
        "prominence": 20.0,
        "min_dist": 0.10
    },
    "Case D": {
        "name": "Case D: Fast Stiff Cycle (Extrap)",
        "params": (9.0, 4500.0, 0.6, 1.0),
        "duration": 2.5,
        "prominence": 3.0,
        "min_dist": 0.03
    }
}

def analyze_waveform(signal: np.ndarray, prominence: float = 10.0, min_dist: float = 0.04, dt: float = DT) -> Dict[str, float]:
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

# ==============================================================================
# TASK 1: TRAIN FiLM FINE-TUNED WITH alpha_div = 0 ON 7 SEEDS
# ==============================================================================
def train_film_finetuned_nodiv(train_ds, seed: int, total_steps: int = 600, lr: float = 1.5e-4):
    out_name = f"film_finetuned_nodiv_s{seed}.pt"
    out_path = os.path.join(CURRENT_DIR, out_name)
    if os.path.exists(out_path):
        print(f"  [Seed {seed}] Found existing fine-tuned checkpoint: {out_name}")
        model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
        model.load_state_dict(torch.load(out_path, map_location="cpu"))
        model.eval()
        return model

    print(f"  [Seed {seed}] Training FiLM Fine-Tuned (alpha_div=0, steps={total_steps}, lr={lr})...", flush=True)
    set_seed(seed)
    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)

    base_ckpt = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
    if not os.path.exists(base_ckpt) and seed == 42:
        base_ckpt = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [10, 20, 30]))
        state_b, params_b, future_b = sample_batch(train_ds, K)
        c_tot0 = state_b[:, 0:1] + state_b[:, 1:2] + state_b[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b, target_ctot=c_tot0)
            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            loss = loss + state_l + 0.05 * deriv_l
        loss = loss / K

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), out_path)
    print(f"  [Seed {seed}] Saved {out_name} in {time.time()-t0:.1f}s.")
    model.eval()
    return model

def run_task1():
    print("\n" + "=" * 80)
    print("TASK 1: TRAINING FiLM FINE-TUNED WITH alpha_div = 0 (SAME 7 SEEDS, 600 STEPS)")
    print("=" * 80)
    models_nodiv = {}
    train_ds = None
    for seed in SEEDS:
        out_name = f"film_finetuned_nodiv_s{seed}.pt"
        out_path = os.path.join(CURRENT_DIR, out_name)
        if not os.path.exists(out_path) and train_ds is None:
            train_ds, _ = build_dataset_with_ic_splits(
                TRAINING_PARAMS, traj_duration=1.2, dt=DT, num_train_ics=4, num_val_ics=2
            )
        models_nodiv[seed] = train_film_finetuned_nodiv(train_ds, seed)
    return models_nodiv

# ==============================================================================
# TASK 2: PER-SEED TABLE FOR CASES B AND D
# ==============================================================================
def rollout_eval(model, ic, params, steps, prominence, min_dist, is_constrained=True):
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot0 = curr[:, 0:1] + curr[:, 1:2] + curr[:, 2:3] if is_constrained else None

    pred = [ic.copy()]
    with torch.no_grad():
        for _ in range(steps):
            if is_constrained:
                curr = model(curr, p_t, target_ctot=c_tot0)
            else:
                curr = model(curr, p_t)
            pred.append(curr.squeeze(0).cpu().numpy())
    pred = np.array(pred)

    m = analyze_waveform(pred[:, 2], prominence=prominence, min_dist=min_dist)
    c_tot_pred = pred[:, 0] + pred[:, 1] + pred[:, 2]
    drift = float(np.ptp(c_tot_pred))
    return pred, m, drift

def run_task2():
    print("\n" + "=" * 80)
    print("TASK 2: GENERATING PER-SEED BREAKDOWN TABLE FOR CASE B AND CASE D")
    print("=" * 80)

    per_seed_results = {"Case B": {}, "Case D": {}}

    for case_id in ["Case B", "Case D"]:
        cinfo = CASES[case_id]
        p = cinfo["params"]
        p_arr = np.array(p, dtype=np.float64)
        duration = cinfo["duration"]
        steps = int(duration / DT)

        # Warm up to limit cycle
        y = DEFAULT_GUESS.copy()
        for _ in range(int(20.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        # Ground Truth Reference
        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
        gt_m = analyze_waveform(gt_traj[:, 2], prominence=cinfo["prominence"], min_dist=cinfo["min_dist"])
        gt_p = gt_m["period"]
        gt_a = gt_m["amp"]

        per_seed_results[case_id]["gt"] = {"period": gt_p, "amp": gt_a, "ret": gt_m["self_decay_pct"]}

        for seed in SEEDS:
            # Load all 5 models for this seed
            # 1. Baseline
            m_base = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
            ckpt_b = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
            m_base.load_state_dict(torch.load(ckpt_b, map_location="cpu"))
            m_base.eval()

            # 2. Div-only
            m_div = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
            ckpt_d = os.path.join(CURRENT_DIR, f"divergence_regularized_mlp_s{seed}.pt")
            m_div.load_state_dict(torch.load(ckpt_d, map_location="cpu"))
            m_div.eval()

            # 3. FiLM Pure
            m_film = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
            ckpt_f = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
            m_film.load_state_dict(torch.load(ckpt_f, map_location="cpu"))
            m_film.eval()

            # 4. FiLM FT (nodiv)
            m_ft = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
            ckpt_ft = os.path.join(CURRENT_DIR, f"film_finetuned_nodiv_s{seed}.pt")
            m_ft.load_state_dict(torch.load(ckpt_ft, map_location="cpu"))
            m_ft.eval()

            # 5. FiLM+Div
            m_fdiv = FiLMWithRates(hidden_dim=128, dt=DT, m0_floor=0.5)
            ckpt_fd = os.path.join(CURRENT_DIR, f"film_divergence_regularized_mlp_s{seed}.pt")
            m_fdiv.load_state_dict(torch.load(ckpt_fd, map_location="cpu"))
            m_fdiv.eval()

            variants = [
                ("Baseline", m_base, False),
                ("Div-Only", m_div, False),
                ("FiLM Pure", m_film, True),
                ("FiLM FT (NoDiv)", m_ft, True),
                ("FiLM+Div", m_fdiv, True)
            ]

            seed_dict = {}
            for vname, vmodel, is_c in variants:
                pred_tr, m_res, drift = rollout_eval(vmodel, ic, p, steps, cinfo["prominence"], cinfo["min_dist"], is_constrained=is_c)
                p_err = abs(m_res["period"] - gt_p) / gt_p * 100.0 if gt_p > 0 and m_res["period"] > 0 else 100.0
                a_err = abs(m_res["amp"] - gt_a) / gt_a * 100.0 if gt_a > 0 and m_res["amp"] > 0 else 100.0
                mse = float(np.mean((pred_tr - gt_traj) ** 2))
                seed_dict[vname] = {
                    "period": m_res["period"],
                    "period_err_pct": p_err,
                    "amp": m_res["amp"],
                    "amp_err_pct": a_err,
                    "ret_pct": m_res["self_decay_pct"],
                    "drift": drift,
                    "mse": mse
                }

            per_seed_results[case_id][seed] = seed_dict

    # Format Markdown Table for Cases B and D
    lines = []
    lines.append("# Per-Seed Comprehensive Evaluation: Case B & Case D Across 7 Seeds\n\n")
    for case_id in ["Case B", "Case D"]:
        cinfo = CASES[case_id]
        gt_info = per_seed_results[case_id]["gt"]
        lines.append(f"## {cinfo['name']}\n")
        lines.append(f"**Ground Truth RK4**: Period $T = {gt_info['period']:.4f}$ s, Steady Amp $= {gt_info['amp']:.1f}$, Retention $= {gt_info['ret']:.1f}%\n\n")
        lines.append("| Seed | Model | Period $T$ (s) | Period Err (%) | Amplitude | Amp Err (%) | Retention (%) | Mass Drift (mM) | Rollout MSE |\n")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        variants = ["Baseline", "Div-Only", "FiLM Pure", "FiLM FT (NoDiv)", "FiLM+Div"]
        for seed in SEEDS:
            first = True
            for vname in variants:
                d = per_seed_results[case_id][seed][vname]
                s_label = f"**{seed}**" if first else ""
                first = False
                drift_str = f"{d['drift']:.4f}" if "FiLM" in vname else f"{d['drift']:.1f}"
                bold_name = f"**{vname}**" if "FiLM" in vname else vname
                lines.append(f"| {s_label} | {bold_name} | {d['period']:.4f} | {d['period_err_pct']:.2f}% | {d['amp']:.1f} | {d['amp_err_pct']:.2f}% | {d['ret_pct']:.1f}% | {drift_str} | {d['mse']:.2e} |\n")
            lines.append("| | | | | | | | | |\n")

        # Summary Row (Mean +/- Std)
        lines.append(f"### {case_id} Statistical Summary (Mean ± Std Across 7 Seeds)\n\n")
        lines.append("| Model | Period Err (%) | Amp Err (%) | Retention (%) | Mass Drift (mM) | Rollout MSE |\n")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for vname in variants:
            p_errs = [per_seed_results[case_id][s][vname]["period_err_pct"] for s in SEEDS]
            a_errs = [per_seed_results[case_id][s][vname]["amp_err_pct"] for s in SEEDS]
            rets = [per_seed_results[case_id][s][vname]["ret_pct"] for s in SEEDS]
            drifts = [per_seed_results[case_id][s][vname]["drift"] for s in SEEDS]
            mses = [per_seed_results[case_id][s][vname]["mse"] for s in SEEDS]

            drift_fmt = f"{np.mean(drifts):.4f} ± {np.std(drifts):.4f}" if "FiLM" in vname else f"{np.mean(drifts):.1f} ± {np.std(drifts):.1f}"
            bold_name = f"**{vname}**" if "FiLM" in vname else vname
            lines.append(f"| {bold_name} | {np.mean(p_errs):.2f} ± {np.std(p_errs):.2f}% | {np.mean(a_errs):.2f} ± {np.std(a_errs):.2f}% | {np.mean(rets):.1f} ± {np.std(rets):.1f}% | {drift_fmt} | {np.mean(mses):.2e} ± {np.std(mses):.2e} |\n")
        lines.append("\n---\n\n")

    table_path = os.path.join(CURRENT_DIR, "cases_b_and_d_per_seed_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Saved per-seed table for Cases B & D to '{table_path}'.\n")
    return per_seed_results, "".join(lines)

# ==============================================================================
# TASK 3: LEARNED HOPF BIFURCATION BOUNDARY FROM JACOBIAN EIGENVALUES
# ==============================================================================
def find_steady_state_analytical(params):
    def res(y):
        return system(y, params)
    y_guess = np.array([6300.0, 4200.0, 180.0, 4.5])
    opt = scipy.optimize.least_squares(res, y_guess, bounds=([0, 0, 0, 1e-5], [10002, 10002, 10002, 100]))
    return opt.x

def compute_true_jacobian(y, params, eps=1e-6):
    J = np.zeros((4, 4))
    f0 = system(y, params)
    for i in range(4):
        yp = y.copy()
        yp[i] += eps
        J[:, i] = (system(yp, params) - f0) / eps
    return J

def compute_neural_jacobian(model, y, params):
    p_t = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    c_tot = y_t[:, 0:1] + y_t[:, 1:2] + y_t[:, 2:3]

    def rate_fn(s):
        s_next = model(s, p_t, target_ctot=c_tot)
        return (s_next - s) / DT

    J_t = torch.autograd.functional.jacobian(rate_fn, y_t).squeeze()
    return J_t.detach().numpy()

def run_task3():
    print("\n" + "=" * 80)
    print("TASK 3: COMPUTING LEARNED HOPF BIFURCATION BOUNDARY VIA JACOBIAN EIGENVALUES")
    print("=" * 80)

    # Load champion seed 42 model (and multi-seed models)
    model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    ckpt = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # Parameter Sweep along ka: ka from 1.0 to 14.0 (nominal kd2=4208, kd3=1.0, kp=1.0)
    ka_sweep = np.linspace(1.5, 14.0, 50)
    
    true_max_real = []
    neural_max_real = []
    true_osc_freq = []
    neural_osc_freq = []

    print(f"Sweeping ka from {ka_sweep[0]:.1f} to {ka_sweep[-1]:.1f} (50 points)...")
    for ka in ka_sweep:
        p = (float(ka), 4208.0, 1.0, 1.0)
        y_ss = find_steady_state_analytical(p)

        # True Jacobian
        J_true = compute_true_jacobian(y_ss, p)
        e_true = np.linalg.eigvals(J_true)
        # Find complex conjugate pair
        complex_mask = np.abs(np.imag(e_true)) > 1.0
        if np.any(complex_mask):
            max_re_true = float(np.max(np.real(e_true[complex_mask])))
            freq_true = float(np.abs(np.imag(e_true[complex_mask][0])))
        else:
            max_re_true = float(np.max(np.real(e_true)))
            freq_true = 0.0

        # Neural Jacobian
        J_neural = compute_neural_jacobian(model, y_ss, p)
        e_neural = np.linalg.eigvals(J_neural)
        complex_mask_n = np.abs(np.imag(e_neural)) > 1.0
        if np.any(complex_mask_n):
            max_re_neural = float(np.max(np.real(e_neural[complex_mask_n])))
            freq_neural = float(np.abs(np.imag(e_neural[complex_mask_n][0])))
        else:
            max_re_neural = float(np.max(np.real(e_neural)))
            freq_neural = 0.0

        true_max_real.append(max_re_true)
        neural_max_real.append(max_re_neural)
        true_osc_freq.append(freq_true)
        neural_osc_freq.append(freq_neural)

    true_max_real = np.array(true_max_real)
    neural_max_real = np.array(neural_max_real)

    # Locate zero crossings (Hopf bifurcation points where Re(lambda) = 0)
    def find_zero_crossings(x, y):
        crossings = []
        for i in range(len(y) - 1):
            if (y[i] <= 0 and y[i+1] > 0) or (y[i] >= 0 and y[i+1] < 0):
                # linear interpolation
                x_cross = x[i] - y[i] * (x[i+1] - x[i]) / (y[i+1] - y[i])
                crossings.append(float(x_cross))
        return crossings

    true_hopf_ka = find_zero_crossings(ka_sweep, true_max_real)
    neural_hopf_ka = find_zero_crossings(ka_sweep, neural_max_real)

    print("\nHopf Bifurcation Boundary Analysis:")
    print(f"  True System Hopf Boundary (Re(lambda)=0): ka = {true_hopf_ka}")
    print(f"  Learned SciML Hopf Boundary (Re(lambda)=0): ka = {neural_hopf_ka}")

    # Plot Jacobian Eigenvalue Spectrum & Hopf Boundary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    # Left: Real part of leading complex eigenvalue
    ax1.plot(ka_sweep, true_max_real, 'k-', lw=2.2, label=r'Ground Truth $\max(\mathrm{Re}(\lambda))$')
    ax1.plot(ka_sweep, neural_max_real, 'g--', lw=2.0, label=r'Proposed FiLM $\max(\mathrm{Re}(\lambda_{\theta}))$')
    ax1.axhline(0.0, color='gray', linestyle=':', alpha=0.7)

    for th in true_hopf_ka:
        ax1.axvline(th, color='black', linestyle='-.', lw=1.2, label=f'True Hopf ($k_a={th:.2f}$)')
    for nh in neural_hopf_ka:
        ax1.axvline(nh, color='green', linestyle=':', lw=1.5, label=f'Learned Hopf ($k_a={nh:.2f}$)')

    ax1.set_xlabel(r'Bifurcation Parameter $k_a$ (Activation Rate)', fontsize=11)
    ax1.set_ylabel(r'Stability Metric $\mathrm{Re}(\lambda)$', fontsize=11)
    ax1.set_title(r'Jacobian Leading Real Eigenvalue Spectrum' + '\n' + r'(Hopf Crossing: Re$(\lambda) > 0 \to$ Limit Cycle)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # Right: Imaginary part / oscillation frequency
    ax2.plot(ka_sweep, true_osc_freq, 'k-', lw=2.2, label=r'Ground Truth Angular Frequency $\omega$')
    ax2.plot(ka_sweep, neural_osc_freq, 'g--', lw=2.0, label=r'Learned SciML Frequency $\omega_{\theta}$')
    ax2.set_xlabel(r'Bifurcation Parameter $k_a$', fontsize=11)
    ax2.set_ylabel(r'Angular Frequency $\omega = \mathrm{Im}(\lambda)$ (rad/s)', fontsize=11)
    ax2.set_title(r'Oscillation Frequency Along the Parameter Manifold' + '\n' + r'($T = 2\pi / \omega$ at Limit Cycle Onset)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)

    hopf_fig_path = os.path.join(CURRENT_DIR, "biochemical_learned_hopf_boundary.png")
    plt.tight_layout()
    plt.savefig(hopf_fig_path, dpi=300)
    plt.close()
    print(f"Saved Hopf bifurcation eigenvalue analysis to '{hopf_fig_path}'.")

    hopf_results = {
        "ka_sweep": ka_sweep.tolist(),
        "true_max_real": true_max_real.tolist(),
        "neural_max_real": neural_max_real.tolist(),
        "true_hopf_ka": true_hopf_ka,
        "neural_hopf_ka": neural_hopf_ka
    }
    return hopf_results

# ==============================================================================
# TASK 4: m0 FLOOR SWEEP & SINGLE-COMPONENT ABLATIONS
# ==============================================================================
def run_task4():
    print("\n" + "=" * 80)
    print("TASK 4: PACEMAKER m0 FLOOR SWEEP & SINGLE-COMPONENT ABLATIONS")
    print("=" * 80)

    # Part A: m0 Floor Sweep across 0.5, 0.2, 0.1, 0.05, 0.01, 1e-4
    floors = [0.5, 0.2, 0.1, 0.05, 0.01, 1e-4]
    p_slide = (6.0, 4208.0, 1.0, 1.0)
    p_arr = np.array(p_slide, dtype=np.float64)
    duration = 2.0
    steps = int(duration / DT)

    y = DEFAULT_GUESS.copy()
    for _ in range(int(20.0 / DT)):
        y = rk4_step(y, DT, p_arr)
    ic = y.copy()

    gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
    gt_m = analyze_waveform(gt_traj[:, 2], prominence=10.0, min_dist=0.04)

    floor_results = []
    print("\n--- Sweeping Pacemaker m0 Positivity Floor (clamp min=floor) ---")
    for fl in floors:
        model_fl = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=fl)
        ckpt = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
        model_fl.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model_fl.eval()

        pred, m_res, drift = rollout_eval(model_fl, ic, p_slide, steps, prominence=10.0, min_dist=0.04, is_constrained=True)
        min_m0_traj = float(np.min(pred[:, 3]))
        p_err = abs(m_res["period"] - gt_m["period"]) / gt_m["period"] * 100.0
        a_err = abs(m_res["amp"] - gt_m["amp"]) / gt_m["amp"] * 100.0

        res_row = {
            "floor": fl,
            "period": m_res["period"],
            "period_err_pct": p_err,
            "amp": m_res["amp"],
            "amp_err_pct": a_err,
            "ret_pct": m_res["self_decay_pct"],
            "min_m0_reached": min_m0_traj,
            "drift": drift
        }
        floor_results.append(res_row)
        print(f"  Floor = {fl:7.4f} | T = {m_res['period']:.4f}s (err {p_err:.2f}%) | Amp = {m_res['amp']:.1f} (err {a_err:.2f}%) | Min m0 = {min_m0_traj:.4f} | Drift = {drift:.4f}mM")

    # Part B: Single-Component Architectural Ablations
    print("\n--- Evaluating Single-Component Architectural Ablations ---")
    # We evaluate 4 distinct variants:
    # 1. Full Proposed Model (FiLM + Mass Conservation + Positivity Clamp m0>=0.5)
    # 2. Ablation: No Mass Conservation (remove subunits normalization)
    # 3. Ablation: No Positivity Clamp (m0 floor set to 0.0)
    # 4. Ablation: Baseline Unconstrained (No Constraints, No FiLM)

    class AblationNoMassConservation(ProposedFiLMPhysicsConstrainedMLP):
        def forward(self, state, params, target_ctot=None):
            s_norm = state / self.state_scales
            p_norm = params / self.param_scales
            p_embed = self.p_net(p_norm)
            h = self.block1(s_norm, p_embed)
            h = self.block2(h, p_embed)
            h = self.block3(h, p_embed)
            out = self.head(h)

            J1 = torch.nn.functional.softplus(out[:, 0:1]) * self.rate_scales[0]
            J2 = torch.nn.functional.softplus(out[:, 1:2]) * self.rate_scales[1]
            J3 = torch.nn.functional.softplus(out[:, 2:3]) * self.rate_scales[2]
            rm0 = out[:, 3:4] * self.rate_scales[3]

            rd = -J1 + J2
            ra1 = J1 - J3
            rm1 = J3 - J2

            d_next  = torch.clamp(state[:, 0:1] + self.dt * rd, min=0.0)
            a1_next = torch.clamp(state[:, 1:2] + self.dt * ra1, min=0.0)
            m1_next = torch.clamp(state[:, 2:3] + self.dt * rm1, min=0.0)
            m0_next = torch.clamp(state[:, 3:4] + self.dt * rm0, min=0.5)
            # NO MASS CONSERVATION NORMALIZATION
            return torch.cat([d_next, a1_next, m1_next, m0_next], dim=1)

    ablation_results = {}
    ckpt_film = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    ckpt_base = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")

    # 1. Full Model
    m_full = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_full.load_state_dict(torch.load(ckpt_film, map_location="cpu"))
    m_full.eval()
    pred_full, m_full_res, drift_full = rollout_eval(m_full, ic, p_slide, steps, 10.0, 0.04, is_constrained=True)

    # 2. No Mass Conservation
    m_nomass = AblationNoMassConservation(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_nomass.load_state_dict(torch.load(ckpt_film, map_location="cpu"))
    m_nomass.eval()
    pred_nomass, m_nomass_res, drift_nomass = rollout_eval(m_nomass, ic, p_slide, steps, 10.0, 0.04, is_constrained=False)

    # 3. No Positivity Clamp (floor=0.0)
    m_noclamp = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.0)
    m_noclamp.load_state_dict(torch.load(ckpt_film, map_location="cpu"))
    m_noclamp.eval()
    pred_noclamp, m_noclamp_res, drift_noclamp = rollout_eval(m_noclamp, ic, p_slide, steps, 10.0, 0.04, is_constrained=True)

    # 4. Baseline Unconstrained (No Constraints, No FiLM)
    m_base = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    m_base.load_state_dict(torch.load(ckpt_base, map_location="cpu"))
    m_base.eval()
    pred_base, m_base_res, drift_base = rollout_eval(m_base, ic, p_slide, steps, 10.0, 0.04, is_constrained=False)

    ablation_items = [
        ("Full Proposed SciML", m_full_res, drift_full, pred_full),
        ("w/o Exact Mass Conservation", m_nomass_res, drift_nomass, pred_nomass),
        ("w/o Positivity Clamp (m0>=0.0)", m_noclamp_res, drift_noclamp, pred_noclamp),
        ("Unconstrained Baseline MLP", m_base_res, drift_base, pred_base)
    ]

    for name, mres, dr, pr in ablation_items:
        perr = abs(mres["period"] - gt_m["period"]) / gt_m["period"] * 100.0
        aerr = abs(mres["amp"] - gt_m["amp"]) / gt_m["amp"] * 100.0
        mse = float(np.mean((pr - gt_traj) ** 2))
        ablation_results[name] = {
            "period_err_pct": perr,
            "amp_err_pct": aerr,
            "ret_pct": mres["self_decay_pct"],
            "drift": dr,
            "mse": mse
        }
        print(f"  {name:30s} | T_err={perr:5.2f}% | Amp_err={aerr:5.2f}% | Ret={mres['self_decay_pct']:5.1f}% | Drift={dr:8.3f}mM | MSE={mse:.2e}")

    # Save Markdown Report for Task 4
    md = []
    md.append("# Sensitivity Sweep of Pacemaker Floor & Single-Component Ablations\n\n")
    md.append("### A. Pacemaker $m_0$ Floor Sweep ($[0.5, 0.2, 0.1, 0.05, 0.01, 10^{-4}]$)\n\n")
    md.append("| Clamp Floor | Period $T$ (s) | Period Err (%) | Amplitude | Amp Err (%) | Retention (%) | Minimum $m_0$ in Trajectory | Mass Drift (mM) |\n")
    md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for r in floor_results:
        md.append(f"| {r['floor']:7.4f} | {r['period']:.4f} | {r['period_err_pct']:.2f}% | {r['amp']:.1f} | {r['amp_err_pct']:.2f}% | {r['ret_pct']:.1f}% | {r['min_m0_reached']:.4f} | {r['drift']:.4f} |\n")

    md.append("\n### B. Single-Component Architectural Ablation Study\n\n")
    md.append("| Architecture Variant | Period Err (%) | Amp Err (%) | Retention (%) | Mass Drift (mM) | Rollout MSE |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for name, d in ablation_results.items():
        drift_str = f"{d['drift']:.4f}" if d['drift'] < 1.0 else f"{d['drift']:.1f}"
        bold_name = f"**{name}**" if "Full" in name else name
        md.append(f"| {bold_name} | {d['period_err_pct']:.2f}% | {d['amp_err_pct']:.2f}% | {d['ret_pct']:.1f}% | {drift_str} | {d['mse']:.2e} |\n")

    ablation_path = os.path.join(CURRENT_DIR, "floor_sweep_and_ablations_report.md")
    with open(ablation_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\nSaved floor sweep and ablation report to '{ablation_path}'.")

    return floor_results, ablation_results

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING ADVANCED ABLATION, PER-SEED PROTOCOL & HOPF BOUNDARY SUITE")
    print("=" * 80)

    # 1. Train and evaluate FiLM FT (alpha_div=0)
    run_task1()

    # 2. Per-Seed Table for Cases B and D
    run_task2()

    # 3. Learned Hopf Boundary from Jacobian Eigenvalues
    run_task3()

    # 4. Floor Sweep & Component Ablations
    run_task4()

    print("\n" + "=" * 80)
    print(f"ALL PROTOCOL EXTENSIONS COMPLETED SUCCESSFULLY IN {time.time()-t_start:.1f}s.")
    print("=" * 80)

if __name__ == "__main__":
    main()
