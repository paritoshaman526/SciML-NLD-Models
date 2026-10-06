"""
======================================================================================
COMPLETE BIOCHEMICAL DYNAMICAL BENCHMARK SUITE:
- Divergence Controls (Baseline & FiLM fine-tuned with alpha_div = 0 across 7 seeds)
- Damped Regime Treatment for Case D (Decay Ratio instead of limit cycle amplitude)
- Pacemaker m0 Floor Sweep {0.5, 0.05, 0.005} on Cases B and C
- Single-Component Architectural Ablations (Isolated FiLM, Softplus, Mass Projection)
- Per-Seed Breakdown for Cases B and D
- Comprehensive Table 1 with Mean ± Std, Median, and Min-Max over 7 seeds
- Seed 42 4-Regime Verification Grid Figure Generation
======================================================================================
"""

import os
import sys

# Windows OpenBLAS/OMP thread safety
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import shutil
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import find_peaks

torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\c0229372-9c2c-4e39-b054-0c23ec47512a"
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
    sample_batch
)

SEEDS = [42, 101, 2024, 777, 999, 1234, 5555]

CASES = {
    "Case A": {
        "id": "Case A",
        "name": "Case A: Nominal Slide [In-Sample]",
        "params": (6.0, 4208.0, 1.0, 1.0),
        "duration": 2.5,
        "prominence": 10.0,
        "min_dist": 0.04,
        "is_damped": False
    },
    "Case B": {
        "id": "Case B",
        "name": "Case B: Slower High-Amplitude [Interpolation]",
        "params": (3.0, 4200.0, 0.9, 1.0),
        "duration": 3.0,
        "prominence": 20.0,
        "min_dist": 0.10,
        "is_damped": False
    },
    "Case C": {
        "id": "Case C",
        "name": "Case C: Giant Amplitude Surge [Extrapolation]",
        "params": (5.0, 4000.0, 0.5, 1.0),
        "duration": 2.5,
        "prominence": 30.0,
        "min_dist": 0.04,
        "is_damped": False
    },
    "Case D": {
        "id": "Case D",
        "name": "Case D: Damped Stiff Regime [Extrapolation]",
        "params": (9.0, 4500.0, 0.6, 1.0),
        "duration": 2.5,
        "prominence": 2.0,
        "min_dist": 0.03,
        "is_damped": True
    }
}

# ==============================================================================
# HELPER METRICS
# ==============================================================================
def analyze_signal(signal: np.ndarray, prominence: float = 10.0, min_dist: float = 0.04, dt: float = DT, is_damped: bool = False) -> Dict[str, float]:
    dist_steps = max(int(min_dist / dt), 10)
    peaks, _ = find_peaks(signal, prominence=prominence, distance=dist_steps)
    troughs, _ = find_peaks(-signal, prominence=prominence, distance=dist_steps)

    n_pairs = min(len(peaks), len(troughs))
    if len(peaks) > 1:
        period = float(np.mean(np.diff(peaks)) * dt)
    else:
        period = 0.0

    if n_pairs > 0:
        amps_cycles = [signal[peaks[i]] - signal[troughs[i]] for i in range(n_pairs)]
        steady_amp = float(np.mean(amps_cycles[int(len(amps_cycles) * 0.5):])) if len(amps_cycles) > 1 else float(amps_cycles[0])
        amp_initial = float(amps_cycles[0])
        amp_final = float(amps_cycles[-1])
        decay_ratio = amp_final / max(amp_initial, 1e-6)
    else:
        steady_amp = float(np.ptp(signal))
        amp_initial = steady_amp
        amp_final = steady_amp
        decay_ratio = 1.0

    quarters = np.array_split(signal, 4)
    q1_amp = float(np.ptp(quarters[0]))
    q4_amp = float(np.ptp(quarters[3]))
    retention_pct = (q4_amp / max(q1_amp, 1e-6)) * 100.0

    return {
        "period": period,
        "steady_amp": steady_amp,
        "amp_initial": amp_initial,
        "amp_final": amp_final,
        "decay_ratio": decay_ratio,
        "retention_pct": retention_pct,
        "n_cycles": n_pairs
    }

def rollout_model(model, ic: np.ndarray, params: tuple, steps: int, prominence: float, min_dist: float, is_constrained: bool = True, is_damped: bool = False) -> Tuple[np.ndarray, Dict[str, float], float]:
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

    m = analyze_signal(pred[:, 2], prominence=prominence, min_dist=min_dist, is_damped=is_damped)
    c_tot_pred = pred[:, 0] + pred[:, 1] + pred[:, 2]
    drift = float(np.ptp(c_tot_pred))
    return pred, m, drift

# ==============================================================================
# 1. DIVERGENCE CONTROLS: BASELINE FINE-TUNED WITH alpha_div = 0 ON 7 SEEDS
# ==============================================================================
def train_baseline_finetuned_nodiv(train_ds, seed: int, total_steps: int = 2000, lr: float = 1e-3):
    out_name = f"baseline_finetuned_nodiv_s{seed}.pt"
    out_path = os.path.join(CURRENT_DIR, out_name)
    if os.path.exists(out_path):
        print(f"  [Seed {seed}] Found existing: {out_name}")
        model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
        model.load_state_dict(torch.load(out_path, map_location="cpu"))
        model.eval()
        return model

    base_ckpt = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
    if not os.path.exists(base_ckpt) and seed == 42:
        base_ckpt = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")

    print(f"  [Seed {seed}] Fine-Tuning Baseline with alpha_div=0.0 (steps={total_steps}, lr={lr})...", flush=True)
    set_seed(seed)
    model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [10, 20, 30]))
        state_b, params_b, future_b = sample_batch(train_ds, K)
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, params_b)
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

def ensure_divergence_controls():
    print("\n" + "=" * 80)
    print("STEP 1: ENSURING DIVERGENCE CONTROLS (alpha_div = 0) FOR ALL 7 SEEDS")
    print("=" * 80)
    train_ds = None
    for seed in SEEDS:
        out_name = f"baseline_finetuned_nodiv_s{seed}.pt"
        out_path = os.path.join(CURRENT_DIR, out_name)
        if not os.path.exists(out_path):
            if train_ds is None:
                train_ds, _ = build_dataset_with_ic_splits(
                    TRAINING_PARAMS, traj_duration=1.2, dt=DT, num_train_ics=4, num_val_ics=2
                )
            train_baseline_finetuned_nodiv(train_ds, seed)
        else:
            print(f"  [Seed {seed}] Baseline FT control ({out_name}) ready.")

        # FiLM FT control
        film_ft_name = f"film_finetuned_nodiv_s{seed}.pt"
        if os.path.exists(os.path.join(CURRENT_DIR, film_ft_name)):
            print(f"  [Seed {seed}] FiLM FT control ({film_ft_name}) ready.")

# ==============================================================================
# 2. SINGLE-COMPONENT ARCHITECTURAL ABLATIONS (ONE AT A TIME)
# ==============================================================================
class ProposedFiLM(ProposedFiLMPhysicsConstrainedMLP):
    """Full Proposed Model."""
    pass

class AblationNoFiLM(nn.Module):
    """Ablation 1: Remove FiLM conditioning. Replace with input concatenation [s, p]."""
    def __init__(self, hidden_dim: int = 128, dt: float = DT, m0_floor: float = 0.5):
        super().__init__()
        self.dt = dt
        self.m0_floor = m0_floor
        self.state_scales = nn.Parameter(torch.tensor([6500.0, 4500.0, 1000.0, 10.0]), requires_grad=False)
        self.param_scales = nn.Parameter(torch.tensor([10.0, 5000.0, 2.0, 2.0]), requires_grad=False)
        self.rate_scales  = nn.Parameter(torch.tensor([60000.0, 60000.0, 60000.0, 100.0]), requires_grad=False)

        # Direct concatenation of 4 state + 4 params = 8 inputs
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, state, params, target_ctot=None):
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        x = torch.cat([s_norm, p_norm], dim=-1)
        out = self.net(x)

        J1 = torch.nn.functional.softplus(out[:, 0:1]) * self.rate_scales[0]
        J2 = torch.nn.functional.softplus(out[:, 1:2]) * self.rate_scales[1]
        J3 = torch.nn.functional.softplus(out[:, 2:3]) * self.rate_scales[2]
        rm0 = out[:, 3:4] * self.rate_scales[3]

        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2

        d_raw  = torch.clamp(state[:, 0:1] + self.dt * rd, min=0.0)
        a1_raw = torch.clamp(state[:, 1:2] + self.dt * ra1, min=0.0)
        m1_raw = torch.clamp(state[:, 2:3] + self.dt * rm1, min=0.0)
        m0_next = torch.clamp(state[:, 3:4] + self.dt * rm0, min=self.m0_floor)

        if target_ctot is None:
            target_ctot = state[:, 0:1] + state[:, 1:2] + state[:, 2:3]

        subunits_sum = torch.clamp(d_raw + a1_raw + m1_raw, min=1e-6)
        scale = target_ctot / subunits_sum
        return torch.cat([d_raw * scale, a1_raw * scale, m1_raw * scale, m0_next], dim=1)

class AblationNoSoftplus(ProposedFiLMPhysicsConstrainedMLP):
    """Ablation 2: Remove Softplus fluxes. Output unconstrained linear fluxes."""
    def forward(self, state, params, target_ctot=None):
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)
        h = self.block1(s_norm, p_embed)
        h = self.block2(h, p_embed)
        h = self.block3(h, p_embed)
        out = self.head(h)

        # Linear unconstrained fluxes (can be negative!)
        J1 = out[:, 0:1] * self.rate_scales[0]
        J2 = out[:, 1:2] * self.rate_scales[1]
        J3 = out[:, 2:3] * self.rate_scales[2]
        rm0 = out[:, 3:4] * self.rate_scales[3]

        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2

        d_raw  = torch.clamp(state[:, 0:1] + self.dt * rd, min=0.0)
        a1_raw = torch.clamp(state[:, 1:2] + self.dt * ra1, min=0.0)
        m1_raw = torch.clamp(state[:, 2:3] + self.dt * rm1, min=0.0)
        m0_next = torch.clamp(state[:, 3:4] + self.dt * rm0, min=self.m0_floor)

        if target_ctot is None:
            target_ctot = state[:, 0:1] + state[:, 1:2] + state[:, 2:3]

        subunits_sum = torch.clamp(d_raw + a1_raw + m1_raw, min=1e-6)
        scale = target_ctot / subunits_sum
        return torch.cat([d_raw * scale, a1_raw * scale, m1_raw * scale, m0_next], dim=1)

class AblationNoMassProjection(ProposedFiLMPhysicsConstrainedMLP):
    """Ablation 3: Remove Mass Projection. Do not normalize to target_ctot."""
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
        m0_next = torch.clamp(state[:, 3:4] + self.dt * rm0, min=self.m0_floor)
        # NO SUBUNIT PROJECTION
        return torch.cat([d_next, a1_next, m1_next, m0_next], dim=1)

def train_ablation_model(model_cls, out_name: str, seed: int = 42, total_steps: int = 3000, lr: float = 2e-3):
    out_path = os.path.join(CURRENT_DIR, out_name)
    if os.path.exists(out_path):
        print(f"  Found existing ablation model: {out_name}")
        model = model_cls(hidden_dim=128, dt=DT, m0_floor=0.5)
        model.load_state_dict(torch.load(out_path, map_location="cpu"))
        model.eval()
        return model

    print(f"  Training Ablation Model: {out_name} (Seed {seed}, Steps {total_steps})...", flush=True)
    train_ds, _ = build_dataset_with_ic_splits(TRAINING_PARAMS, traj_duration=1.2, dt=DT, num_train_ics=4, num_val_ics=2)
    set_seed(seed)
    model = model_cls(hidden_dim=128, dt=DT, m0_floor=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [5, 15, 25]))
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
    print(f"  Saved ablation checkpoint to {out_name} in {time.time()-t0:.1f}s.")
    model.eval()
    return model

# ==============================================================================
# 3. PACEMAKER m0 FLOOR SWEEP ON CASES B AND C ({0.5, 0.05, 0.005})
# ==============================================================================
def run_m0_floor_sweep():
    print("\n" + "=" * 80)
    print("STEP 3: PACEMAKER m0 FLOOR SWEEP {0.5, 0.05, 0.005} ON CASES B AND C")
    print("=" * 80)

    floors = [0.5, 0.05, 0.005]
    sweep_results = {}

    target_cases = ["Case B", "Case C"]
    for case_id in target_cases:
        cinfo = CASES[case_id]
        p = cinfo["params"]
        p_arr = np.array(p, dtype=np.float64)
        duration = cinfo["duration"]
        steps = int(duration / DT)

        y = DEFAULT_GUESS.copy()
        for _ in range(int(20.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
        gt_m = analyze_signal(gt_traj[:, 2], prominence=cinfo["prominence"], min_dist=cinfo["min_dist"], is_damped=False)
        true_min_m0 = float(np.min(gt_traj[:, 3]))

        case_sweep = []
        for fl in floors:
            # Multi-seed evaluation for each floor
            p_errs, a_ratios, drifts, min_m0s = [], [], [], []
            for seed in SEEDS:
                model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=fl)
                ckpt = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
                if not os.path.exists(ckpt) and seed == 42:
                    ckpt = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()

                pred, m, drift = rollout_model(model, ic, p, steps, cinfo["prominence"], cinfo["min_dist"], is_constrained=True)
                p_err = abs(m["period"] - gt_m["period"]) / gt_m["period"] * 100.0 if gt_m["period"] > 0 and m["period"] > 0 else 100.0
                a_rat = m["steady_amp"] / gt_m["steady_amp"] if gt_m["steady_amp"] > 0 else 0.0
                min_m0 = float(np.min(pred[:, 3]))

                p_errs.append(p_err)
                a_ratios.append(a_rat)
                drifts.append(drift)
                min_m0s.append(min_m0)

            res_entry = {
                "floor": fl,
                "period_err_mean": float(np.mean(p_errs)),
                "period_err_std": float(np.std(p_errs)),
                "amp_ratio_mean": float(np.mean(a_ratios)),
                "amp_ratio_std": float(np.std(a_ratios)),
                "min_m0_mean": float(np.mean(min_m0s)),
                "drift_mean": float(np.mean(drifts))
            }
            case_sweep.append(res_entry)
            print(f"  [{case_id}] Floor = {fl:5.3f} | Period Err: {np.mean(p_errs):.2f} ± {np.std(p_errs):.2f}% | Amp Ratio: {np.mean(a_ratios):.3f} ± {np.std(a_ratios):.3f} | Min m0 reached: {np.mean(min_m0s):.4f} (True: {true_min_m0:.4f}) | Drift: {np.mean(drifts):.4f}mM")

        sweep_results[case_id] = {
            "true_min_m0": true_min_m0,
            "gt_period": gt_m["period"],
            "gt_amp": gt_m["steady_amp"],
            "results": case_sweep
        }
    return sweep_results

# ==============================================================================
# 4. SINGLE-COMPONENT ABLATION STUDY EXECUTION
# ==============================================================================
def run_single_component_ablations():
    print("\n" + "=" * 80)
    print("STEP 4: RUNNING ISOLATED SINGLE-COMPONENT ARCHITECTURAL ABLATIONS")
    print("=" * 80)

    # Train or load the three clean ablations
    m_nofilm = train_ablation_model(AblationNoFiLM, "ablation_nofilm_s42.pt", seed=42)
    m_nosoft = train_ablation_model(AblationNoSoftplus, "ablation_nosoftplus_s42.pt", seed=42)
    m_nomass = train_ablation_model(AblationNoMassProjection, "ablation_nomass_s42.pt", seed=42)

    # Load Full Proposed FiLM and Unconstrained Baseline (Seed 42)
    m_full = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    ckpt_film = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    m_full.load_state_dict(torch.load(ckpt_film, map_location="cpu"))
    m_full.eval()

    m_base = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    ckpt_base = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
    m_base.load_state_dict(torch.load(ckpt_base, map_location="cpu"))
    m_base.eval()

    ablation_models = [
        ("Full Proposed SciML", m_full, True),
        ("w/o FiLM (Concat [s, p])", m_nofilm, True),
        ("w/o Softplus Fluxes (Unconstrained Rates)", m_nosoft, True),
        ("w/o Mass Projection (No Normalization)", m_nomass, False),
        ("Unconstrained Baseline MLP", m_base, False)
    ]

    # Evaluate across all 4 benchmark cases
    ablation_results = {}
    for case_id in ["Case A", "Case B", "Case C", "Case D"]:
        cinfo = CASES[case_id]
        p = cinfo["params"]
        p_arr = np.array(p, dtype=np.float64)
        duration = cinfo["duration"]
        steps = int(duration / DT)
        is_damped = cinfo["is_damped"]

        y = DEFAULT_GUESS.copy()
        for _ in range(int(20.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
        gt_m = analyze_signal(gt_traj[:, 2], prominence=cinfo["prominence"], min_dist=cinfo["min_dist"], is_damped=is_damped)

        case_dict = {}
        for name, model, is_c in ablation_models:
            pred, m, drift = rollout_model(model, ic, p, steps, cinfo["prominence"], cinfo["min_dist"], is_constrained=is_c, is_damped=is_damped)
            p_err = abs(m["period"] - gt_m["period"]) / gt_m["period"] * 100.0 if gt_m["period"] > 0 and m["period"] > 0 else 100.0
            if is_damped:
                # Compare decay ratio to true decay ratio
                metric_ratio = m["decay_ratio"] / max(gt_m["decay_ratio"], 1e-6)
            else:
                metric_ratio = m["steady_amp"] / max(gt_m["steady_amp"], 1e-6)
            mse = float(np.mean((pred - gt_traj) ** 2))
            case_dict[name] = {
                "period_err": p_err,
                "metric_ratio": metric_ratio,
                "drift": drift,
                "mse": mse
            }
            print(f"  [{case_id}] {name:32s} | P_Err: {p_err:5.2f}% | Ratio: {metric_ratio:5.3f} | Drift: {drift:8.4f}mM | MSE: {mse:.2e}")
        ablation_results[case_id] = case_dict

    return ablation_results

# ==============================================================================
# 5. MULTI-SEED PROTOCOL & COMPREHENSIVE TABLE 1
# ==============================================================================
def run_multiseed_table1():
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATING ALL 6 MODELS ACROSS 7 SEEDS & 4 CASES FOR TABLE 1")
    print("=" * 80)

    model_names = [
        "Baseline (Standard)",
        "Baseline FT (alpha_div=0 Control)",
        "Baseline Div-Only",
        "FiLM Pure",
        "FiLM FT (alpha_div=0 Control)",
        "FiLM+Div"
    ]

    all_data = {cid: {mname: {"p_errs": [], "ratios": [], "drifts": [], "per_seed": {}} for mname in model_names} for cid in CASES}

    for case_id in CASES:
        cinfo = CASES[case_id]
        p = cinfo["params"]
        p_arr = np.array(p, dtype=np.float64)
        duration = cinfo["duration"]
        steps = int(duration / DT)
        is_damped = cinfo["is_damped"]

        y = DEFAULT_GUESS.copy()
        for _ in range(int(20.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
        gt_m = analyze_signal(gt_traj[:, 2], prominence=cinfo["prominence"], min_dist=cinfo["min_dist"], is_damped=is_damped)

        for seed in SEEDS:
            # 1. Baseline Standard
            m_b = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
            ck_b = os.path.join(CURRENT_DIR, f"baseline_unconstrained_mlp_s{seed}.pt")
            if not os.path.exists(ck_b) and seed == 42:
                ck_b = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
            m_b.load_state_dict(torch.load(ck_b, map_location="cpu"))
            m_b.eval()

            # 2. Baseline FT Control (alpha=0)
            m_bft = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
            ck_bft = os.path.join(CURRENT_DIR, f"baseline_finetuned_nodiv_s{seed}.pt")
            m_bft.load_state_dict(torch.load(ck_bft, map_location="cpu"))
            m_bft.eval()

            # 3. Baseline Div-Only
            m_bdiv = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
            ck_bdiv = os.path.join(CURRENT_DIR, f"divergence_regularized_mlp_s{seed}.pt")
            if not os.path.exists(ck_bdiv) and seed == 42:
                ck_bdiv = os.path.join(CURRENT_DIR, "divergence_regularized_mlp.pt")
            m_bdiv.load_state_dict(torch.load(ck_bdiv, map_location="cpu"))
            m_bdiv.eval()

            # 4. FiLM Pure
            m_fp = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
            ck_fp = os.path.join(CURRENT_DIR, f"best_proposed_biochemical_mlp_s{seed}.pt")
            if not os.path.exists(ck_fp) and seed == 42:
                ck_fp = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
            m_fp.load_state_dict(torch.load(ck_fp, map_location="cpu"))
            m_fp.eval()

            # 5. FiLM FT Control (alpha=0)
            m_fft = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
            ck_fft = os.path.join(CURRENT_DIR, f"film_finetuned_nodiv_s{seed}.pt")
            m_fft.load_state_dict(torch.load(ck_fft, map_location="cpu"))
            m_fft.eval()

            # 6. FiLM+Div
            m_fdiv = FiLMWithRates(hidden_dim=128, dt=DT, m0_floor=0.5)
            ck_fdiv = os.path.join(CURRENT_DIR, f"film_divergence_regularized_mlp_s{seed}.pt")
            if not os.path.exists(ck_fdiv) and seed == 42:
                ck_fdiv = os.path.join(CURRENT_DIR, "film_divergence_regularized_mlp.pt")
            m_fdiv.load_state_dict(torch.load(ck_fdiv, map_location="cpu"))
            m_fdiv.eval()

            models_eval = [
                ("Baseline (Standard)", m_b, False),
                ("Baseline FT (alpha_div=0 Control)", m_bft, False),
                ("Baseline Div-Only", m_bdiv, False),
                ("FiLM Pure", m_fp, True),
                ("FiLM FT (alpha_div=0 Control)", m_fft, True),
                ("FiLM+Div", m_fdiv, True)
            ]

            for mname, model_obj, is_c in models_eval:
                pred, m, drift = rollout_model(model_obj, ic, p, steps, cinfo["prominence"], cinfo["min_dist"], is_constrained=is_c, is_damped=is_damped)
                p_err = abs(m["period"] - gt_m["period"]) / gt_m["period"] * 100.0 if gt_m["period"] > 0 and m["period"] > 0 else 100.0
                if is_damped:
                    ratio = m["decay_ratio"] / max(gt_m["decay_ratio"], 1e-6)
                else:
                    ratio = m["steady_amp"] / max(gt_m["steady_amp"], 1e-6)

                all_data[case_id][mname]["p_errs"].append(p_err)
                all_data[case_id][mname]["ratios"].append(ratio)
                all_data[case_id][mname]["drifts"].append(drift)
                all_data[case_id][mname]["per_seed"][seed] = {
                    "period": m["period"],
                    "period_err": p_err,
                    "amp_or_decay": m["decay_ratio"] if is_damped else m["steady_amp"],
                    "ratio": ratio,
                    "drift": drift
                }

    # Format Markdown Table 1 and Per-Seed Reports
    t1_lines = []
    t1_lines.append("# Table 1: Global Benchmark Performance Across 7 Seeds\n\n")
    t1_lines.append("> **Evaluation Protocol**:\n")
    t1_lines.append("> - **Seeds**: 7 independent seeds (`[42, 101, 2024, 777, 999, 1234, 5555]`)\n")
    t1_lines.append("> - **Statistics**: Reported as `Mean ± Std`, `[Median]`, and `(Min – Max)` across all 7 seeds\n")
    t1_lines.append("> - **Regime Classification**: Case A is **In-Sample**; Cases B & C are **Sustained Limit Cycles**; Case D is a **Damped Stiff Regime** (evaluated by decay ratio $D_{\\text{model}} / D_{\\text{true}}$)\n\n")

    for case_id in ["Case A", "Case B", "Case C", "Case D"]:
        cinfo = CASES[case_id]
        ratio_header = "Decay Ratio ($D_{\\text{model}}/D_{\\text{true}}$)" if cinfo["is_damped"] else "Amp Ratio ($A_{\\text{model}}/A_{\\text{true}}$)"
        t1_lines.append(f"### {cinfo['name']}\n\n")
        t1_lines.append(f"| Model | Period Error (%) <br> `Mean ± Std [Med] (Min-Max)` | {ratio_header} <br> `Mean ± Std [Med] (Min-Max)` | Mass Drift (mM) <br> `Mean ± Std [Med] (Min-Max)` |\n")
        t1_lines.append("| :--- | :---: | :---: | :---: |\n")

        for mname in model_names:
            pe = all_data[case_id][mname]["p_errs"]
            ra = all_data[case_id][mname]["ratios"]
            dr = all_data[case_id][mname]["drifts"]

            pe_str = f"{np.mean(pe):.2f} ± {np.std(pe):.2f}% <br> `[{np.median(pe):.2f}%]` <br> ({np.min(pe):.2f} – {np.max(pe):.2f}%)"
            ra_str = f"{np.mean(ra):.3f} ± {np.std(ra):.3f} <br> `[{np.median(ra):.3f}]` <br> ({np.min(ra):.3f} – {np.max(ra):.3f})"
            if "FiLM" in mname:
                dr_str = f"{np.mean(dr):.4f} ± {np.std(dr):.4f} <br> `[{np.median(dr):.4f}]` <br> ({np.min(dr):.4f} – {np.max(dr):.4f})"
                bold_m = f"**{mname}**"
            else:
                dr_str = f"{np.mean(dr):.1f} ± {np.std(dr):.1f} <br> `[{np.median(dr):.1f}]` <br> ({np.min(dr):.1f} – {np.max(dr):.1f})"
                bold_m = mname

            t1_lines.append(f"| {bold_m} | {pe_str} | {ra_str} | {dr_str} |\n")
        t1_lines.append("\n---\n\n")

    t1_path = os.path.join(CURRENT_DIR, "biochemical_publication_table1.md")
    with open(t1_path, "w", encoding="utf-8") as f:
        f.write("".join(t1_lines))
    print(f"Saved complete Table 1 to '{t1_path}'.")

    # Per-seed breakdown table for Cases B and D
    ps_lines = []
    ps_lines.append("# Per-Seed Granular Breakdown for Cases B and D\n\n")
    for case_id in ["Case B", "Case D"]:
        cinfo = CASES[case_id]
        is_d = cinfo["is_damped"]
        metric_col = "Decay Ratio" if is_d else "Amplitude"
        ratio_col = "Decay Match ($D_{\\text{model}}/D_{\\text{true}}$)" if is_d else "Amp Ratio ($A_{\\text{model}}/A_{\\text{true}}$)"
        ps_lines.append(f"### {cinfo['name']}\n\n")
        ps_lines.append(f"| Seed | Model | Period $T$ (s) | Period Err (%) | {metric_col} | {ratio_col} | Mass Drift (mM) |\n")
        ps_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for seed in SEEDS:
            first = True
            for mname in model_names:
                d = all_data[case_id][mname]["per_seed"][seed]
                s_label = f"**{seed}**" if first else ""
                first = False
                drift_str = f"{d['drift']:.4f}" if "FiLM" in mname else f"{d['drift']:.1f}"
                bold_m = f"**{mname}**" if "FiLM" in mname else mname
                ps_lines.append(f"| {s_label} | {bold_m} | {d['period']:.4f} | {d['period_err']:.2f}% | {d['amp_or_decay']:.2f} | {d['ratio']:.3f} | {drift_str} |\n")
            ps_lines.append("| | | | | | | |\n")
        ps_lines.append("\n---\n\n")

    ps_path = os.path.join(CURRENT_DIR, "cases_b_and_d_per_seed_granular.md")
    with open(ps_path, "w", encoding="utf-8") as f:
        f.write("".join(ps_lines))
    print(f"Saved per-seed granular report to '{ps_path}'.")

    return all_data

# ==============================================================================
# 6. REGENERATE SINGLE-SEED 4-CASE FIGURE LABELED "SEED 42"
# ==============================================================================
def regenerate_seed42_figure():
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING 4-REGIME VERIFICATION GRID (LABELED 'SEED 42')")
    print("=" * 80)

    # Load seed 42 checkpoints
    ckpt_film = os.path.join(CURRENT_DIR, "best_proposed_biochemical_mlp.pt")
    model_prop = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    model_prop.load_state_dict(torch.load(ckpt_film, map_location="cpu"))
    model_prop.eval()

    ckpt_comb = os.path.join(CURRENT_DIR, "film_divergence_regularized_mlp.pt")
    model_comb = FiLMWithRates(hidden_dim=128, dt=DT, m0_floor=0.5)
    model_comb.load_state_dict(torch.load(ckpt_comb, map_location="cpu"))
    model_comb.eval()

    ckpt_base = os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt")
    model_base = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    model_base.load_state_dict(torch.load(ckpt_base, map_location="cpu"))
    model_base.eval()

    ckpt_div = os.path.join(CURRENT_DIR, "divergence_regularized_mlp.pt")
    model_div = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
    model_div.load_state_dict(torch.load(ckpt_div, map_location="cpu"))
    model_div.eval()

    regimes_def = [
        {
            "id": "Case A",
            "title": r"Case A: Nominal Slide [In-Sample]",
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
            "title": r"Case D: Damped Stiff Regime [Extrapolation]",
            "subtitle": r"$k_a = 9.0\,\mathrm{s}^{-1},\ k_{d2} = 4500.0\,\mathrm{mM}^{-1}\mathrm{s}^{-1},\ k_{d3} = 0.6\,\mathrm{s}^{-1}$",
            "params": (9.0, 4500.0, 0.6, 1.0),
            "steps": 25000,
            "t_plot": 2.5,
            "ylim_t": (180, 285),
            "xlim_p": (4.5, 6.0),
            "ylim_p": (220, 280),
            "has_inset": True,
            "inset_xlim": (-2, 25),
            "inset_ylim": (-50, 1800)
        }
    ]

    c_gt   = "#0f172a"
    c_base = "#ef4444"
    c_div  = "#3b82f6"
    c_prop = "#15803d"
    c_comb = "#8b5cf6"

    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.size': 9.5,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'lines.linewidth': 1.6
    })

    fig, axes = plt.subplots(4, 2, figsize=(14.5, 17.5), dpi=300)
    plt.subplots_adjust(hspace=0.34, wspace=0.18, top=0.935, bottom=0.035, left=0.07, right=0.96)

    for i, regime in enumerate(regimes_def):
        p_arr = np.array(regime["params"], dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(20.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()

        steps = regime["steps"]
        n_pts = int(regime["t_plot"] / DT)
        time_arr = np.linspace(0, regime["t_plot"], n_pts)

        # Simulate models
        gt_full = simulate_trajectory(ic, steps, DT, p_arr)
        pred_p, _, _ = rollout_model(model_prop, ic, regime["params"], steps, 10.0, 0.04, is_constrained=True)
        pred_c, _, _ = rollout_model(model_comb, ic, regime["params"], steps, 10.0, 0.04, is_constrained=True)
        pred_b, _, _ = rollout_model(model_base, ic, regime["params"], steps, 10.0, 0.04, is_constrained=False)
        pred_d, _, _ = rollout_model(model_div, ic, regime["params"], steps, 10.0, 0.04, is_constrained=False)

        ax_t = axes[i, 0]
        ax_p = axes[i, 1]

        # Time series
        ax_t.plot(time_arr, gt_full[:n_pts, 2], color=c_gt, label="Ground Truth (RK4)", lw=2.2, zorder=5)
        ax_t.plot(time_arr, pred_b[:n_pts, 2], color=c_base, label="Baseline MLP", lw=1.2, alpha=0.85, linestyle=":")
        ax_t.plot(time_arr, pred_d[:n_pts, 2], color=c_div, label="Div-Regularized", lw=1.3, alpha=0.9, linestyle="-.")
        ax_t.plot(time_arr, pred_c[:n_pts, 2], color=c_comb, label="FiLM + Div Combined", lw=1.8, linestyle="--")
        ax_t.plot(time_arr, pred_p[:n_pts, 2], color=c_prop, label="FiLM Pure", lw=1.5, linestyle="-")

        ax_t.set_xlim(0, regime["t_plot"])
        ax_t.set_ylim(regime["ylim_t"])
        ax_t.grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
        ax_t.set_title(f"{regime['title']}\n{regime['subtitle']}", fontweight="bold", fontsize=10.5, loc="left", pad=4)
        ax_t.set_xlabel("Time (s)", fontweight="bold")
        ax_t.set_ylabel(r"Polymer $m_1$ (mM)", fontweight="bold")

        # Phase portrait
        s_idx = max(0, steps - 8000)
        ax_p.plot(gt_full[s_idx:, 3], gt_full[s_idx:, 2], color=c_gt, lw=2.4, zorder=5, label="Ground Truth (RK4)")
        ax_p.plot(pred_b[s_idx:, 3], pred_b[s_idx:, 2], color=c_base, lw=1.1, alpha=0.7, linestyle=":", label="Baseline MLP")
        ax_p.plot(pred_d[s_idx:, 3], pred_d[s_idx:, 2], color=c_div, lw=1.3, alpha=0.85, linestyle="-.", label="Div-Regularized")
        ax_p.plot(pred_c[s_idx:, 3], pred_c[s_idx:, 2], color=c_comb, lw=1.8, linestyle="--", label="FiLM + Div Combined")
        ax_p.plot(pred_p[s_idx:, 3], pred_p[s_idx:, 2], color=c_prop, lw=1.5, linestyle="-", label="FiLM Pure")

        ax_p.set_xlim(regime["xlim_p"])
        ax_p.set_ylim(regime["ylim_p"])
        ax_p.grid(True, linestyle=":", color="#cbd5e1", alpha=0.6)
        ax_p.set_title(rf"Phase Portrait: Orbital Geometry ($m_0$ vs $m_1$) — {regime['id']}", fontweight="bold", fontsize=10.5, loc="left", pad=4)
        ax_p.set_xlabel(r"Pacemaker $m_0$ (mM)", fontweight="bold")
        ax_p.set_ylabel(r"Polymer $m_1$ (mM)", fontweight="bold")

        # Inset for Extrapolative blowup
        if regime["has_inset"]:
            ax_ins = inset_axes(ax_p, width="32%", height="32%", loc="upper right", borderpad=0.8)
            ax_ins.plot(pred_b[s_idx:, 3], pred_b[s_idx:, 2], color=c_base, lw=0.8, alpha=0.55)
            ax_ins.plot(pred_d[s_idx:, 3], pred_d[s_idx:, 2], color=c_div, lw=0.9, alpha=0.7)
            ax_ins.plot(gt_full[s_idx:, 3], gt_full[s_idx:, 2], color=c_gt, lw=1.3)
            ax_ins.plot(pred_c[s_idx:, 3], pred_c[s_idx:, 2], color=c_comb, lw=1.1, linestyle="--")
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    custom_order = [0, 4, 3, 2, 1]
    ordered_handles = [handles[idx] for idx in custom_order]
    ordered_labels = [labels[idx] for idx in custom_order]

    fig.legend(ordered_handles, ordered_labels, loc="upper center", ncol=5,
               fontsize=10.5, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1",
               bbox_to_anchor=(0.515, 0.965), handlelength=2.2, handletextpad=0.5, columnspacing=1.8)

    fig.suptitle("Global 4-Regime Dynamical Verification Grid (Seed 42): FiLM SciML Across the Hopf Bifurcation Surface",
                 fontweight="bold", fontsize=13.5, y=0.990)

    out_master = os.path.join(CURRENT_DIR, "global_4regime_benchmark_grid_seed42.png")
    brain_master = os.path.join(BRAIN_DIR, "global_4regime_benchmark_grid_seed42.png")

    plt.savefig(out_master, dpi=250, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(out_master, brain_master)
    print(f"Generated Seed 42 verification grid successfully:\n  - {out_master}\n  - {brain_master}")

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING ADVANCED EVALUATION & RIGOROUS CONTROLS SUITE")
    print("=" * 80)

    # 1. Divergence Controls for Baseline & FiLM across 7 seeds
    ensure_divergence_controls()

    # 2. m0 floor sweep on Cases B and C
    sweep_results = run_m0_floor_sweep()

    # 3. Single-component ablations
    ablation_results = run_single_component_ablations()

    # 4. Multi-seed Table 1 and per-seed numbers
    all_data = run_multiseed_table1()

    # 5. Regenerate Seed 42 figure
    regenerate_seed42_figure()

    print("\n" + "=" * 80)
    print(f"ALL PROTOCOL EXTENSIONS COMPLETED SUCCESSFULLY IN {time.time()-t_start:.1f}s.")
    print("=" * 80)

if __name__ == "__main__":
    main()
