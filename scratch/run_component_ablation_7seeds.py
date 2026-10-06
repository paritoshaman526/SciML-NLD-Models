"""
D.3: Single-Component Architectural Ablation Study
Trains 3 ablation variants and the full Unified model across 7 canonical seeds.
"""
import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed
from train_and_evaluate_lorenz_style_sciml import find_physical_steady_state, SWEEP_CASES, PHYSICAL_CTOT
from run_complete_evaluation_suite import analyze_signal
from scratch.test_x7_fix import (
    UnifiedLorenzStyleSciML_Fixed, generate_extended_training_data, sample_batch, SEEDS
)

# ============================================================
# ABLATION 1: No FiLM - replace with input concatenation
# ============================================================
class AblationNoFiLM(nn.Module):
    def __init__(self, hidden_dim=128, dt=DT, m0_floor=1e-4):
        super().__init__()
        self.dt = dt
        self.m0_floor = m0_floor
        self.register_buffer('state_scales', torch.tensor([5000.0, 5000.0, 300.0, 10.0]))
        self.register_buffer('param_scales', torch.tensor([10.0, 5000.0, 1.5, 1.0]))
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 5)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def predict_derivatives(self, state, params):
        d = state[:, 0:1]; a1 = state[:, 1:2]
        m1 = state[:, 2:3]; m0 = torch.clamp(state[:, 3:4], min=self.m0_floor)
        ka = params[:, 0:1]; kd2 = params[:, 1:2]; kd3 = params[:, 2:3]; kp = params[:, 3:4]
        J1_base = ka * d; J2_base = 2.0 * kd2 * m0; J3_base = 2.0 * kp * a1 * m0
        ratio = m1 / m0
        denom = torch.clamp((ratio - 2.0) ** 2, min=1e-4)
        R_base = (9.2 * kd2 * m0) / denom
        s_norm = state / self.state_scales; p_norm = params / self.param_scales
        mod = self.net(torch.cat([s_norm, p_norm], dim=1))
        mod_J1 = 1.0 + 0.15 * torch.tanh(mod[:, 0:1])
        mod_J2 = 1.0 + 0.15 * torch.tanh(mod[:, 1:2])
        mod_J3 = 1.0 + 0.15 * torch.tanh(mod[:, 2:3])
        mod_R  = 1.0 + 0.20 * torch.tanh(mod[:, 3:4])
        add_m0 = mod[:, 4:5] * 5.0
        J1 = J1_base * mod_J1; J2 = J2_base * mod_J2; J3 = J3_base * mod_J3; R_deg = R_base * mod_R
        rd = -J1 + J2; ra1 = J1 - J3; rm1 = J3 - J2
        rm0 = kd3 * (m1 - 4.0 * m0) - R_deg - 2.0 * kd3 * m0 - kp * (m0 ** 2) + add_m0
        return torch.cat([rd, ra1, rm1, rm0], dim=1)

    def forward(self, state, params, target_ctot=None):
        dt = self.dt
        k1 = self.predict_derivatives(state, params)
        k2 = self.predict_derivatives(torch.clamp(state + 0.5*dt*k1, min=0.0), params)
        k3 = self.predict_derivatives(torch.clamp(state + 0.5*dt*k2, min=0.0), params)
        k4 = self.predict_derivatives(torch.clamp(state + dt*k3, min=0.0), params)
        y_next = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        d_n = torch.clamp(y_next[:, 0:1], min=0.0)
        a1_n = torch.clamp(y_next[:, 1:2], min=0.0)
        m1_n = torch.clamp(y_next[:, 2:3], min=0.0)
        m0_n = torch.clamp(y_next[:, 3:4], min=self.m0_floor)
        if target_ctot is None:
            c_tot = state[:, 0:1] + state[:, 1:2] + state[:, 2:3]
        elif isinstance(target_ctot, torch.Tensor):
            c_tot = target_ctot.view(-1, 1)
        else:
            c_tot = float(target_ctot)
        subunits = torch.cat([d_n, a1_n, m1_n], dim=1)
        subunits = subunits * (c_tot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_n], dim=1)

# ============================================================
# ABLATION 2: No Softplus - unbounded modulations
# ============================================================
class AblationNoSoftplus(UnifiedLorenzStyleSciML_Fixed):
    def predict_derivatives(self, state, params):
        d = state[:, 0:1]; a1 = state[:, 1:2]
        m1 = state[:, 2:3]; m0 = torch.clamp(state[:, 3:4], min=self.m0_floor)
        ka = params[:, 0:1]; kd2 = params[:, 1:2]; kd3 = params[:, 2:3]; kp = params[:, 3:4]
        J1_base = ka * d; J2_base = 2.0 * kd2 * m0; J3_base = 2.0 * kp * a1 * m0
        ratio = m1 / m0
        denom = torch.clamp((ratio - 2.0) ** 2, min=1e-4)
        R_base = (9.2 * kd2 * m0) / denom
        feats = self.extract_features(state, params)
        h = self.in_proj(feats); h = self.res1(h); h = self.res2(h); h = self.res3(h)
        mod = self.out_proj(h)
        # ABLATION: raw linear (no tanh bounding)
        mod_J1 = 1.0 + 0.15 * mod[:, 0:1]
        mod_J2 = 1.0 + 0.15 * mod[:, 1:2]
        mod_J3 = 1.0 + 0.15 * mod[:, 2:3]
        mod_R  = 1.0 + 0.20 * mod[:, 3:4]
        add_m0 = mod[:, 4:5] * 5.0
        J1 = J1_base * mod_J1; J2 = J2_base * mod_J2; J3 = J3_base * mod_J3; R_deg = R_base * mod_R
        rd = -J1 + J2; ra1 = J1 - J3; rm1 = J3 - J2
        rm0 = kd3 * (m1 - 4.0*m0) - R_deg - 2.0*kd3*m0 - kp*(m0**2) + add_m0
        return torch.cat([rd, ra1, rm1, rm0], dim=1)

# ============================================================
# ABLATION 3: No Mass Projection
# ============================================================
class AblationNoMassProjection(UnifiedLorenzStyleSciML_Fixed):
    def forward(self, state, params, target_ctot=None):
        dt = self.dt
        k1 = self.predict_derivatives(state, params)
        k2 = self.predict_derivatives(torch.clamp(state + 0.5*dt*k1, min=0.0), params)
        k3 = self.predict_derivatives(torch.clamp(state + 0.5*dt*k2, min=0.0), params)
        k4 = self.predict_derivatives(torch.clamp(state + dt*k3, min=0.0), params)
        y_next = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        d_n = torch.clamp(y_next[:, 0:1], min=0.0)
        a1_n = torch.clamp(y_next[:, 1:2], min=0.0)
        m1_n = torch.clamp(y_next[:, 2:3], min=0.0)
        m0_n = torch.clamp(y_next[:, 3:4], min=self.m0_floor)
        return torch.cat([d_n, a1_n, m1_n, m0_n], dim=1)

ABLATION_CONFIGS = [
    ("Unified_Full",        UnifiedLorenzStyleSciML_Fixed, "Full Unified SciML (proposed)"),
    ("NoFiLM",             AblationNoFiLM,               "w/o FiLM (concat [s,p])"),
    ("NoSoftplus",         AblationNoSoftplus,           "w/o Softplus (unbounded mod)"),
    ("NoMassProjection",   AblationNoMassProjection,     "w/o Mass Projection"),
]

def train_seed(seed, dataset, model_cls, total_steps=400):
    set_seed(seed)
    model = model_cls(hidden_dim=128, dt=DT, m0_floor=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps*0.5, total_steps], [2, 6, 8]))
        s0_b, p_b, future_b, is_ss_b, ystar_b = sample_batch(dataset, K=K, batch_size=64)
        c_tot0 = s0_b[:, 0:1] + s0_b[:, 1:2] + s0_b[:, 2:3]
        curr = s0_b; traj_loss = 0.0
        for k in range(K):
            prev = curr
            curr = model(curr, p_b, target_ctot=c_tot0)
            target = future_b[:, k]
            state_l = torch.mean(((curr - target) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (target - (s0_b if k == 0 else future_b[:, k-1])) / model.dt
            deriv_l = torch.mean(((pred_v - true_v) / (model.state_scales * 100.0)) ** 2)
            traj_loss = traj_loss + state_l + 0.05 * deriv_l
        traj_loss = traj_loss / K
        if is_ss_b.any():
            vel_ss = model.predict_derivatives(ystar_b[is_ss_b], p_b[is_ss_b])
            eq_loss = torch.mean((vel_ss / (model.state_scales * 10.0)) ** 2)
        else:
            eq_loss = torch.tensor(0.0)
        total_loss = traj_loss + 0.2 * eq_loss
        optimizer.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step(); scheduler.step()
    return model

def evaluate_all_cases(model):
    model.eval()
    metrics = {}
    for c_info in SWEEP_CASES:
        cid = c_info['id']
        p_arr = np.array(c_info['params'], dtype=np.float64)
        p_t = torch.tensor(c_info['params'], dtype=torch.float32).unsqueeze(0)
        is_ss = c_info['is_steady_state']
        if is_ss:
            y_star = find_physical_steady_state(c_info['params'], c_tot=PHYSICAL_CTOT)
            delta_m1 = 0.15 * y_star[2]
            y_pert = y_star.copy(); y_pert[2] += delta_m1; y_pert[0] -= delta_m1; y_pert[3] *= 1.20
            curr = torch.tensor(y_pert, dtype=torch.float32).unsqueeze(0)
            c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]
            with torch.no_grad():
                for _ in range(4000):
                    curr = model(curr, p_t, target_ctot=c_tot_t)
            final_m1 = float(curr[0, 2]); drift = float(abs(final_m1 - y_star[2]))
            metrics[cid] = {"regime": "Stable Focus", "drift_mM": drift,
                            "pct_recovered": float(max(0, 1 - drift/delta_m1)*100)}
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c_info.get('warmup', 5.0) / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()
            steps = 15000 if cid == "X7" else 6000
            gt_traj = simulate_trajectory(ic, steps, DT, p_arr)
            sig_gt = analyze_signal(gt_traj[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])
            gt_amp = float(sig_gt['steady_amp'])
            curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
            c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]
            traj = [ic.copy()]
            with torch.no_grad():
                for _ in range(steps):
                    curr = model(curr, p_t, target_ctot=c_tot_t)
                    traj.append(curr.squeeze(0).numpy())
            traj = np.array(traj)
            sig = analyze_signal(traj[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])
            pred_amp = float(sig['steady_amp'])
            err = float(abs(pred_amp - gt_amp) / max(gt_amp, 1e-3) * 100.0)
            status = "PASSED"
            if cid == "X7":
                if pred_amp < 50.0: status = "COLLAPSED"
                elif pred_amp > 1500.0: status = "EXPLOSION"
            if err > 50.0 and status == "PASSED": status = "HIGH_ERROR"
            metrics[cid] = {"regime": "Limit Cycle", "gt_amp": gt_amp,
                            "pred_amp": pred_amp, "amp_err_pct": err, "status": status}
    return metrics

def main():
    print("=" * 80)
    print("D.3: COMPONENT ABLATION — 4 VARIANTS x 7 SEEDS x 12 REGIMES")
    print("=" * 80)
    dataset = generate_extended_training_data(horizon=2.5)
    all_results = {}
    for abl_key, model_cls, abl_label in ABLATION_CONFIGS:
        print(f"\n{'='*60}\nABLATION: {abl_label}\n{'='*60}")
        abl_results = {}
        for s in SEEDS:
            t0 = time.time()
            m = train_seed(s, dataset, model_cls)
            ev = evaluate_all_cases(m)
            abl_results[str(s)] = ev
            x7_err = ev['X7'].get('amp_err_pct', -1)
            x4_d = ev['X4']['drift_mM']
            print(f"  Seed {s:4d} | {time.time()-t0:.0f}s | X7: {x7_err:.1f}% | X4 drift: {x4_d:.2f} mM", flush=True)
        all_results[abl_key] = {"label": abl_label, "per_seed": abl_results}

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Ablation':35s} | {'X7 Mean':10s} | {'X7 Std':8s} | {'X7 Fails':8s} | {'X4 drift':10s} | {'X8 Err':8s}")
    print("-" * 90)
    for abl_key, res in all_results.items():
        lbl = res['label']
        x7e = [res['per_seed'][str(s)]['X7'].get('amp_err_pct', 100.0) for s in SEEDS]
        x7f = sum(1 for e in x7e if e > 50.0)
        x4d = [res['per_seed'][str(s)]['X4']['drift_mM'] for s in SEEDS]
        x8e = [res['per_seed'][str(s)]['X8'].get('amp_err_pct', 100.0) for s in SEEDS]
        print(f"  {lbl:33s} | {np.mean(x7e):6.2f}%     | {np.std(x7e):6.2f}%  | {x7f}/7      | {np.mean(x4d):6.2f} mM  | {np.mean(x8e):.4f}%")

    outpath = "scratch/component_ablation_7seed_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {outpath} successfully!")

if __name__ == "__main__":
    main()
