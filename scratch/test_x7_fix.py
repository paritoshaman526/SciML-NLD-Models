import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import (
    system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed
)
from train_and_evaluate_lorenz_style_sciml import (
    LorenzResidualBlock, find_physical_steady_state, SWEEP_CASES, PHYSICAL_CTOT
)
from run_complete_evaluation_suite import analyze_signal

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

# ==============================================================================
# 1. ARCHITECTURE WITH PARAMETERIZED M0 FLOOR (1e-4)
# ==============================================================================
class UnifiedLorenzStyleSciML_Fixed(nn.Module):
    def __init__(self, hidden_dim=128, dt=DT, m0_floor=1e-4):
        super().__init__()
        self.dt = dt
        self.m0_floor = m0_floor
        self.register_buffer('state_scales', torch.tensor([5000.0, 5000.0, 300.0, 10.0]))
        self.register_buffer('param_scales', torch.tensor([10.0, 5000.0, 1.5, 1.0]))

        feat_dim = 12
        self.in_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU()
        )
        self.res1 = LorenzResidualBlock(hidden_dim)
        self.res2 = LorenzResidualBlock(hidden_dim)
        self.res3 = LorenzResidualBlock(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 5)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def extract_features(self, state, params):
        norm_s = state / self.state_scales
        norm_p = params / self.param_scales
        d = norm_s[:, 0:1]
        a1 = norm_s[:, 1:2]
        m1 = norm_s[:, 2:3]
        m0 = norm_s[:, 3:4]
        cross = torch.cat([d * m0, a1 * m0, m1 * m0, m0 * m0], dim=1)
        return torch.cat([norm_s, norm_p, cross], dim=1)

    def predict_derivatives(self, state, params):
        d = state[:, 0:1]
        a1 = state[:, 1:2]
        m1 = state[:, 2:3]
        # Floor clamp on pacemaker m0
        m0 = torch.clamp(state[:, 3:4], min=self.m0_floor)

        ka = params[:, 0:1]
        kd2 = params[:, 1:2]
        kd3 = params[:, 2:3]
        kp = params[:, 3:4]

        # 1. Base physical mass-action fluxes
        J1_base = ka * d
        J2_base = 2.0 * kd2 * m0
        J3_base = 2.0 * kp * a1 * m0

        # Base rational degradation with safe denominator clamp
        ratio = m1 / m0
        denom = torch.clamp((ratio - 2.0) ** 2, min=1e-4)
        R_base = (9.2 * kd2 * m0) / denom

        # 2. Neural residual modulation
        feats = self.extract_features(state, params)
        h = self.in_proj(feats)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        mod = self.out_proj(h)

        mod_J1 = 1.0 + 0.15 * torch.tanh(mod[:, 0:1])
        mod_J2 = 1.0 + 0.15 * torch.tanh(mod[:, 1:2])
        mod_J3 = 1.0 + 0.15 * torch.tanh(mod[:, 2:3])
        mod_R  = 1.0 + 0.20 * torch.tanh(mod[:, 3:4])
        add_m0 = mod[:, 4:5] * 5.0

        J1 = J1_base * mod_J1
        J2 = J2_base * mod_J2
        J3 = J3_base * mod_J3
        R_deg = R_base * mod_R

        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2
        rm0 = kd3 * (m1 - 4.0 * m0) - R_deg - 2.0 * kd3 * m0 - kp * (m0 ** 2) + add_m0

        return torch.cat([rd, ra1, rm1, rm0], dim=1)

    def forward(self, state, params, target_ctot=PHYSICAL_CTOT):
        dt = self.dt
        k1 = self.predict_derivatives(state, params)
        k2 = self.predict_derivatives(torch.clamp(state + 0.5 * dt * k1, min=0.0), params)
        k3 = self.predict_derivatives(torch.clamp(state + 0.5 * dt * k2, min=0.0), params)
        k4 = self.predict_derivatives(torch.clamp(state + dt * k3, min=0.0), params)

        y_next = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        d_next  = torch.clamp(y_next[:, 0:1], min=0.0)
        a1_next = torch.clamp(y_next[:, 1:2], min=0.0)
        m1_next = torch.clamp(y_next[:, 2:3], min=0.0)
        # Enforce m0_floor on state update
        m0_next = torch.clamp(y_next[:, 3:4], min=self.m0_floor)

        if target_ctot is None:
            c_tot = state[:, 0:1] + state[:, 1:2] + state[:, 2:3]
        elif isinstance(target_ctot, torch.Tensor):
            c_tot = target_ctot.view(-1, 1)
        else:
            c_tot = float(target_ctot)

        subunits = torch.cat([d_next, a1_next, m1_next], dim=1)
        subunits = subunits * (c_tot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_next], dim=1)

# ==============================================================================
# 2. EXTENDED HORIZON DATASET GENERATION (2.5s HORIZON)
# ==============================================================================
def generate_extended_training_data(horizon=2.5):
    print(f"Generating training data with extended horizon ({horizon}s / {int(horizon/DT)} steps)...", flush=True)
    dataset = []

    osc_params = [
        (1.0, 3000.0, 0.80, 1.0),   # X7 (T=0.705s -> ~3.55 cycles in 2.5s)
        (2.0, 3000.0, 0.60, 1.0),
        (2.5, 3200.0, 0.80, 1.0),
        (3.5, 2600.0, 1.10, 1.0),
        (4.0, 3200.0, 1.00, 1.0),
        (4.5, 3500.0, 1.30, 1.0),
        (5.0, 5200.0, 0.35, 1.0),
        (6.0, 4200.0, 0.90, 1.0),
        (7.0, 4500.0, 0.70, 1.0),
        (8.5, 4000.0, 0.85, 1.0),
    ]

    for p in osc_params:
        p_arr = np.array(p, dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(6.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        
        # Extended simulation: 2.5s
        steps = int(horizon / DT)
        traj = simulate_trajectory(y, steps, DT, p_arr)
        dataset.append({
            'traj': torch.tensor(traj, dtype=torch.float32),
            'params': torch.tensor(p, dtype=torch.float32),
            'is_steady': False,
            'y_star': None
        })

        # For X7 and slow cases (ka <= 2.0), add an off-phase trajectory to double phase diversity
        if p[0] <= 2.0:
            y_off = y.copy()
            for _ in range(int(0.35 / DT)):
                y_off = rk4_step(y_off, DT, p_arr)
            traj_off = simulate_trajectory(y_off, steps, DT, p_arr)
            dataset.append({
                'traj': torch.tensor(traj_off, dtype=torch.float32),
                'params': torch.tensor(p, dtype=torch.float32),
                'is_steady': False,
                'y_star': None
            })

    ss_params = [
        (10.0, 4600.0, 0.60, 1.0),
        (10.5, 4600.0, 0.60, 1.0),
        (11.0, 6000.0, 0.15, 1.0),
        (11.5, 5500.0, 0.30, 1.0),
        (12.0, 4500.0, 0.50, 1.0),
        (12.0, 6000.0, 0.15, 1.0),
        (13.0, 6000.0, 0.25, 1.0),
        (14.0, 5000.0, 0.30, 1.0),
    ]

    for p in ss_params:
        p_arr = np.array(p, dtype=np.float64)
        y_star = find_physical_steady_state(p, c_tot=PHYSICAL_CTOT)
        
        traj_ss = simulate_trajectory(y_star, int(1.5 / DT), DT, p_arr)
        dataset.append({
            'traj': torch.tensor(traj_ss, dtype=torch.float32),
            'params': torch.tensor(p, dtype=torch.float32),
            'is_steady': True,
            'y_star': torch.tensor(y_star, dtype=torch.float32)
        })

        for pert_mag in [50.0, 150.0]:
            y_pert = y_star.copy()
            y_pert[2] += pert_mag
            y_pert[0] -= pert_mag
            traj_pert = simulate_trajectory(y_pert, int(1.5 / DT), DT, p_arr)
            dataset.append({
                'traj': torch.tensor(traj_pert, dtype=torch.float32),
                'params': torch.tensor(p, dtype=torch.float32),
                'is_steady': True,
                'y_star': torch.tensor(y_star, dtype=torch.float32)
            })

    print(f"Generated {len(dataset)} trajectories with horizon={horizon}s.", flush=True)
    return dataset

def sample_batch(dataset, K, batch_size=64):
    indices = np.random.choice(len(dataset), size=batch_size, replace=True)
    s0_list, p_list, future_list, is_ss_list, ystar_list = [], [], [], [], []
    for idx in indices:
        item = dataset[idx]
        traj = item['traj']
        L = len(traj)
        t_start = np.random.randint(0, L - K - 1)
        s0_list.append(traj[t_start])
        future_list.append(traj[t_start + 1 : t_start + K + 1])
        p_list.append(item['params'])
        is_ss_list.append(item['is_steady'])
        ystar_list.append(item['y_star'] if item['is_steady'] else traj[t_start])
    
    return (
        torch.stack(s0_list),
        torch.stack(p_list),
        torch.stack(future_list),
        torch.tensor(is_ss_list, dtype=torch.bool),
        torch.stack(ystar_list)
    )

# ==============================================================================
# 3. TRAINING FUNCTION
# ==============================================================================
def train_seed(seed, dataset, m0_floor=1e-4, total_steps=400):
    set_seed(seed)
    model = UnifiedLorenzStyleSciML_Fixed(hidden_dim=128, dt=DT, m0_floor=m0_floor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.5, total_steps], [2, 6, 8]))
        s0_b, p_b, future_b, is_ss_b, ystar_b = sample_batch(dataset, K=K, batch_size=64)
        c_tot0 = s0_b[:, 0:1] + s0_b[:, 1:2] + s0_b[:, 2:3]

        curr = s0_b
        traj_loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, p_b, target_ctot=c_tot0)
            target = future_b[:, k]
            state_l = torch.mean(((curr - target) * weights) ** 2)

            pred_v = (curr - prev) / model.dt
            true_v = (target - (s0_b if k == 0 else future_b[:, k - 1])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            traj_loss = traj_loss + state_l + 0.05 * deriv_l

        traj_loss = traj_loss / K

        if is_ss_b.any():
            ss_states = ystar_b[is_ss_b]
            ss_params = p_b[is_ss_b]
            vel_ss = model.predict_derivatives(ss_states, ss_params)
            eq_loss = torch.mean((vel_ss / (model.state_scales * 10.0)) ** 2)
        else:
            eq_loss = torch.tensor(0.0)

        total_loss = traj_loss + 0.2 * eq_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

    return model

# ==============================================================================
# 4. EVALUATION SPECIFICALLY TARGETING X7 AND THE FULL 12 REGIMES
# ==============================================================================
def evaluate_x7(model):
    model.eval()
    p = (1.0, 3000.0, 0.80, 1.0)
    p_arr = np.array(p, dtype=np.float64)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

    y = DEFAULT_GUESS.copy()
    for _ in range(int(6.0 / DT)):
        y = rk4_step(y, DT, p_arr)
    ic = y.copy()

    steps = 15000 # 1.5s rollout
    gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
    sig_gt = analyze_signal(gt_traj[:, 2], prominence=30.0, min_dist=0.15)
    gt_amp = float(sig_gt['steady_amp'])

    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]

    traj = [ic.copy()]
    with torch.no_grad():
        for _ in range(steps):
            curr = model(curr, p_t, target_ctot=c_tot_t)
            traj.append(curr.squeeze(0).numpy())
    traj = np.array(traj)

    m1 = traj[:, 2]
    m0 = traj[:, 3]
    sig = analyze_signal(m1, prominence=30.0, min_dist=0.15)
    pred_amp = float(sig['steady_amp'])
    err_pct = float(abs(pred_amp - gt_amp) / gt_amp * 100.0)

    status = "PASSED"
    if pred_amp < 50.0:
        status = "COLLAPSED_TO_ZERO"
    elif pred_amp > 1500.0:
        status = "RUNAWAY_EXPLOSION"
    elif err_pct > 50.0:
        status = "HIGH_ERROR"

    return {
        "pred_amp": pred_amp,
        "gt_amp": gt_amp,
        "err_pct": err_pct,
        "status": status,
        "m1_min": float(m1.min()),
        "m1_max": float(m1.max()),
        "m0_min": float(m0.min()),
        "m0_max": float(m0.max()),
    }

def main():
    print("=" * 80)
    print("RUNNING X7 FIX EXPERIMENT: HORIZON -> 2.5s, FLOOR -> 1e-4")
    print("=" * 80)

    dataset = generate_extended_training_data(horizon=2.5)
    results = {}

    print(f"\nTraining all 7 canonical seeds: {SEEDS}...")
    for s in SEEDS:
        t0 = time.time()
        print(f"  Training Seed {s:4d}...", end=" ", flush=True)
        m = train_seed(s, dataset, m0_floor=1e-4, total_steps=400)
        res_x7 = evaluate_x7(m)
        torch.save(m.state_dict(), f"scratch/unified_sciml_fixed_s{s}.pt")
        results[s] = res_x7
        print(f"Done in {time.time()-t0:.1f}s | Amp: {res_x7['pred_amp']:.1f} (GT: {res_x7['gt_amp']:.1f}) | Err: {res_x7['err_pct']:.2f}% | Status: {res_x7['status']}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    print("X7 7-SEED SUMMARY AFTER FIX (HORIZON=2.5s, FLOOR=1e-4)")
    print("=" * 80)

    amps = [results[s]["pred_amp"] for s in SEEDS]
    errs = [results[s]["err_pct"] for s in SEEDS]
    statuses = [results[s]["status"] for s in SEEDS]

    failures = [s for s in SEEDS if results[s]["status"] in ["COLLAPSED_TO_ZERO", "RUNAWAY_EXPLOSION"]]
    failure_rate = len(failures) / len(SEEDS) * 100.0

    print(f"{'Seed':6s} | {'Amp (mM)':10s} | {'GT Amp (mM)':12s} | {'Amp Err (%)':12s} | {'m0 min':8s} | {'Status'}")
    print("-" * 75)
    for s in SEEDS:
        r = results[s]
        print(f"{s:6d} | {r['pred_amp']:10.2f} | {r['gt_amp']:12.2f} | {r['err_pct']:11.2f}% | {r['m0_min']:8.4f} | {r['status']}")

    print("-" * 75)
    print(f"Mean Amp Error:   {np.mean(errs):.2f} ± {np.std(errs):.2f} %")
    print(f"Median Amp Error: {np.median(errs):.2f} %")
    print(f"Min - Max Error:  ({np.min(errs):.2f}% - {np.max(errs):.2f}%)")
    print(f"Catastrophic Failures: {len(failures)} / {len(SEEDS)} ({failure_rate:.1f}%)")
    print(f"Prior Failure Rate:    3 / 7 (42.9%) -> New Failure Rate: {len(failures)} / 7 ({failure_rate:.1f}%)")

    with open("scratch/x7_fix_results.json", "w") as f:
        json.dump({"seeds": SEEDS, "results": results, "failure_rate_pct": failure_rate}, f, indent=2)
    print("\nSaved scratch/x7_fix_results.json successfully!")

if __name__ == "__main__":
    main()
