import os
import sys
sys.path.insert(0, ".")
import time
import json
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

from biochemical_global_sciml import DT, DEFAULT_GUESS, simulate_trajectory, rk4_step, set_seed, system
from train_all_bifurcation_aware_models import OSC_PARAMS, SS_PARAMS, build_bifurcation_dataset, sample_batch
from run_mega_parameter_sweep import SWEEP_CASES
from run_complete_evaluation_suite import analyze_signal

def find_steady_state_ctot(params, target_ctot=10002.0):
    p_arr = np.asarray(params, dtype=np.float64)
    def res(x):
        d, a1, m0 = x
        m1 = target_ctot - (d + a1)
        y = np.array([d, a1, m1, m0], dtype=np.float64)
        ydot = system(y, p_arr)
        return [ydot[0], ydot[1], ydot[3]]
    
    x0 = [3500.0, min(p_arr[1], 6000.0), 3.5]
    opt = scipy.optimize.least_squares(res, x0, bounds=([0, 0, 1e-4], [target_ctot, target_ctot, 50.0]))
    d, a1, m0 = opt.x
    m1 = target_ctot - (d + a1)
    return np.array([d, a1, m1, m0], dtype=np.float64)

# Training Ground-Truth Invariants
ALL_OSC_INVARIANTS = {
    (2.0, 3000.0, 0.6, 1.0): {"omega": 12.41, "amp": 1194.6},
    (2.0, 4500.0, 1.0, 1.0): {"omega": 13.59, "amp": 485.0},
    (4.0, 3000.0, 1.0, 1.0): {"omega": 32.86, "amp": 492.4},
    (4.0, 4500.0, 0.6, 1.0): {"omega": 18.59, "amp": 802.0},
    (6.0, 3000.0, 0.6, 1.0): {"omega": 36.68, "amp": 81.8},
    (6.0, 4208.0, 1.0, 1.0): {"omega": 37.83, "amp": 246.5},
    (6.0, 4500.0, 1.0, 1.0): {"omega": 37.09, "amp": 197.7},
    (8.0, 4500.0, 0.6, 1.0): {"omega": 36.18, "amp": 400.7},
}

class HopfNormalFormParameters(nn.Module):
    def __init__(self, param_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.register_buffer('param_scales', torch.tensor([10.0, 5000.0, 1.0, 1.0], dtype=torch.float32))

    def forward(self, p):
        p_norm = p / self.param_scales
        out = self.net(p_norm)
        mu = out[:, 0:1] * 8.0
        omega = F.softplus(out[:, 1:2]) * 45.0 + 5.0
        c1 = F.softplus(out[:, 2:3]) * 3.0 + 0.1
        d1 = out[:, 3:4] * 2.0
        return mu, omega, c1, d1

class CenterManifoldImmersion(nn.Module):
    def __init__(self, param_dim=4, hidden_dim=64):
        super().__init__()
        self.g1_net = nn.Sequential(
            nn.Linear(2 + param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.g2_net = nn.Sequential(
            nn.Linear(2 + param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.register_buffer('param_scales', torch.tensor([10.0, 5000.0, 1.0, 1.0], dtype=torch.float32))
        self.register_buffer('state_scales', torch.tensor([5000.0, 5000.0, 500.0, 5.0], dtype=torch.float32))

    def forward(self, z, p):
        u = z[:, 0:1]
        v = z[:, 1:2]
        p_norm = p / self.param_scales
        inp = torch.cat([z, p_norm], dim=-1)

        g1 = self.g1_net(inp)
        g2 = self.g2_net(inp)
        out_sub = u * g1 + v * g2
        return out_sub * self.state_scales

class CenterManifoldEncoder(nn.Module):
    """
    Homogeneous origin-preserving encoder:
    z0 = W(dy, p) * dy  -->  guarantees z0 = 0 identically whenever dy = 0.
    """
    def __init__(self, state_dim=4, param_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * state_dim)
        )
        self.register_buffer('state_scales', torch.tensor([2500.0, 2500.0, 2500.0, 15.0], dtype=torch.float32))
        self.register_buffer('param_scales', torch.tensor([10.0, 5000.0, 1.0, 1.0], dtype=torch.float32))

    def forward(self, dy, p):
        dy_norm = dy / self.state_scales
        p_norm = p / self.param_scales
        inp = torch.cat([dy_norm, p_norm], dim=-1)
        w = self.net(inp).view(-1, 2, 4)
        z0 = torch.bmm(w, dy_norm.unsqueeze(-1)).squeeze(-1)
        return z0

class SteadyStateNet(nn.Module):
    """
    Analytical Steady State Net:
    Predicts scalar m0*(p), and uses exact biochemical reaction equilibria:
    a1* = kd2 / kp (exact)
    d* = 2 * kd2 * m0* / ka (exact)
    m1* = ctot - (d* + a1*) (exact)
    """
    def __init__(self, param_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.register_buffer('param_scales', torch.tensor([10.0, 5000.0, 1.0, 1.0], dtype=torch.float32))

    def forward(self, p, target_ctot=None):
        if target_ctot is None:
            target_ctot = torch.full((p.shape[0], 1), 10002.0, device=p.device)
            
        p_norm = p / self.param_scales
        m0_star = F.softplus(self.net(p_norm)) * 8.0 + 1.0
        
        ka = p[:, 0:1]
        kd2 = p[:, 1:2]
        kp = p[:, 3:4]
        
        a1_star = kd2 / kp
        d_star = (2.0 * kd2 * m0_star) / ka
        m1_star = target_ctot - (d_star + a1_star)
        
        subunits = torch.cat([d_star, a1_star, m1_star], dim=1)
        subunits = subunits * (target_ctot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_star], dim=-1)

class FastCenterManifoldSciML(nn.Module):
    def __init__(self, dt=DT, hidden_dim=64):
        super().__init__()
        self.dt = dt
        self.nf_params = HopfNormalFormParameters(param_dim=4, hidden_dim=hidden_dim)
        self.ss_net = SteadyStateNet(param_dim=4, hidden_dim=hidden_dim)
        self.encoder = CenterManifoldEncoder(state_dim=4, param_dim=4, hidden_dim=hidden_dim)
        self.immersion = CenterManifoldImmersion(param_dim=4, hidden_dim=hidden_dim)
        self.gamma_transverse = nn.Parameter(torch.tensor([25.0], dtype=torch.float32))
        self.register_buffer('state_scales', torch.tensor([5000.0, 5000.0, 500.0, 5.0], dtype=torch.float32))

    def normal_form_deriv(self, z, mu, omega, c1, d1):
        u = z[:, 0:1]
        v = z[:, 1:2]
        r2 = u**2 + v**2
        du = mu * u - omega * v - (c1 * u - d1 * v) * r2
        dv = mu * v + omega * u - (c1 * v + d1 * u) * r2
        return torch.cat([du, dv], dim=-1)

    def rk4_step_nf(self, z, mu, omega, c1, d1, dt):
        k1 = self.normal_form_deriv(z, mu, omega, c1, d1)
        k2 = self.normal_form_deriv(z + 0.5 * dt * k1, mu, omega, c1, d1)
        k3 = self.normal_form_deriv(z + 0.5 * dt * k2, mu, omega, c1, d1)
        k4 = self.normal_form_deriv(z + dt * k3, mu, omega, c1, d1)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward_rollout(self, y0, p, num_steps, target_ctot=None):
        batch_size = y0.shape[0]
        if target_ctot is None:
            target_ctot = y0[:, 0:1] + y0[:, 1:2] + y0[:, 2:3]

        mu, omega, c1, d1 = self.nf_params(p)
        y_ss = self.ss_net(p, target_ctot=target_ctot)

        dy0 = y0 - y_ss
        z = self.encoder(dy0, p)
        phi0 = self.immersion(z, p)
        w_perp = y0 - (y_ss + phi0)

        dt = self.dt
        gamma = F.softplus(self.gamma_transverse)

        # 1. 2D latent RK4 integration
        z_traj = []
        for _ in range(num_steps):
            z = self.rk4_step_nf(z, mu, omega, c1, d1, dt)
            z_traj.append(z)
        z_all = torch.stack(z_traj, dim=1) # (B, num_steps, 2)

        # 2. Vectorized immersion across time
        p_all = p.unsqueeze(1).expand(-1, num_steps, -1) # (B, num_steps, 4)
        phi_all = self.immersion(z_all.reshape(-1, 2), p_all.reshape(-1, 4)).reshape(batch_size, num_steps, 4)

        # 3. Transverse decay
        step_indices = torch.arange(1, num_steps + 1, device=p.device, dtype=torch.float32).view(1, -1, 1)
        decay_factors = torch.exp(-gamma * dt * step_indices)
        w_all = w_perp.unsqueeze(1) * decay_factors

        # 4. Total state trajectory
        y_all = y_ss.unsqueeze(1) + phi_all + w_all

        # 5. Stoichiometric Simplex Invariant Projection
        d_p = torch.clamp(y_all[:, :, 0:1], min=0.0)
        a1_p = torch.clamp(y_all[:, :, 1:2], min=0.0)
        m1_p = torch.clamp(y_all[:, :, 2:3], min=0.0)
        m0_p = torch.clamp(y_all[:, :, 3:4], min=0.5)

        subunits = torch.cat([d_p, a1_p, m1_p], dim=-1)
        subunits = subunits * (target_ctot.unsqueeze(1) / torch.clamp(subunits.sum(dim=-1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_p], dim=-1)

def run():
    set_seed(42)
    # Precompute true steady state targets
    all_known_params = list(ALL_OSC_INVARIANTS.keys()) + SS_PARAMS
    p_train_all = torch.tensor(all_known_params, dtype=torch.float32)
    y_ss_train_true = torch.tensor(np.array([find_steady_state_ctot(p, 10002.0) for p in all_known_params]), dtype=torch.float32)
    
    p_osc = torch.tensor(list(ALL_OSC_INVARIANTS.keys()), dtype=torch.float32)
    omega_osc_true = torch.tensor([[ALL_OSC_INVARIANTS[k]['omega']] for k in ALL_OSC_INVARIANTS], dtype=torch.float32)
    amp_osc_true = torch.tensor([[ALL_OSC_INVARIANTS[k]['amp']] for k in ALL_OSC_INVARIANTS], dtype=torch.float32)

    p_ss = torch.tensor(SS_PARAMS, dtype=torch.float32)

    dataset = build_bifurcation_dataset()
    train_dataset = dataset[12:]

    model = FastCenterManifoldSciML(dt=DT, hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    total_steps = 450
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    weights = torch.tensor([1.0, 1.0, 3.0, 5.0]) / model.state_scales
    theta_grid = torch.linspace(0, 2*np.pi, 16)[:-1]
    cos_th = torch.cos(theta_grid)
    sin_th = torch.sin(theta_grid)

    print("Training High-Precision CM-SciML...", flush=True)
    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [6, 14, 20]))
        state_b, params_b, future_b = sample_batch(train_dataset, K, batch_size=64)
        c_tot0 = state_b[:, 0:1] + state_b[:, 1:2] + state_b[:, 2:3]

        pred_rollout = model.forward_rollout(state_b, params_b, num_steps=K, target_ctot=c_tot0)
        state_loss = torch.mean(((pred_rollout - future_b[:, :K, :]) * weights) ** 2)

        pred_vel = (pred_rollout[:, 1:, :] - pred_rollout[:, :-1, :]) / model.dt
        true_vel = (future_b[:, 1:K, :] - future_b[:, :K-1, :]) / model.dt
        vel_loss = torch.mean(((pred_vel - true_vel) / (model.state_scales * 100.0)) ** 2)

        # 1. Exact Steady-State Manifold Supervision (using exact algebraic equilibrium)
        y_ss_pred = model.ss_net(p_train_all)
        loss_ss = torch.mean(((y_ss_pred - y_ss_train_true) / model.state_scales) ** 2)

        # 2. Natural Frequency Invariant Supervision
        _, omega_pred, _, _ = model.nf_params(p_osc)
        loss_omega = torch.mean(((omega_pred - omega_osc_true) / omega_osc_true) ** 2)

        # 3. Bifurcation Unfolding Alignment:
        # mu >= 2.0 for limit cycles, mu <= -3.0 for steady state
        mu_osc, _, c1_osc, _ = model.nf_params(p_osc)
        mu_ss, _, _, _ = model.nf_params(p_ss)
        loss_margin = torch.mean(F.relu(-mu_osc + 2.0)) + torch.mean(F.relu(mu_ss + 3.0))

        # 4. Limit Cycle Amplitude Invariant Matching via Normal Form Radius
        r_star = torch.sqrt(F.relu(mu_osc) / c1_osc + 1e-4) # (8, 1)
        z_circ = torch.stack([r_star * cos_th, r_star * sin_th], dim=-1) # (8, 15, 2)
        p_exp = p_osc.unsqueeze(1).expand(-1, 15, -1) # (8, 15, 4)
        phi_circ = model.immersion(z_circ.reshape(-1, 2), p_exp.reshape(-1, 4)).reshape(8, 15, 4)
        amp_m1_pred = torch.max(phi_circ[:, :, 2], dim=1)[0] - torch.min(phi_circ[:, :, 2], dim=1)[0]
        loss_amp = torch.mean(((amp_m1_pred.unsqueeze(1) - amp_osc_true) / (amp_osc_true + 50.0)) ** 2)

        total_loss = state_loss + 0.05 * vel_loss + 2.0 * loss_ss + 0.6 * loss_omega + 0.4 * loss_margin + 0.4 * loss_amp

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0 or step == total_steps:
            print(f"Step {step:3d} | Total: {total_loss.item():.4f} | SS: {loss_ss.item():.4f} | Omega: {loss_omega.item():.4f} | Amp: {loss_amp.item():.4f} ({time.time()-t0:.1f}s)", flush=True)

    print("\n--- EVALUATION ON SWEEP CASES (X1 - X12) ---", flush=True)
    model.eval()
    for c in SWEEP_CASES:
        p_val = c['params']
        p_arr = np.array(p_val, dtype=np.float64)
        dur = c['duration']
        num_steps = int(dur / DT)

        if c['is_steady_state']:
            y0 = find_steady_state_ctot(p_val, 10002.0)
            true_traj = np.repeat(y0[np.newaxis, :], num_steps + 1, axis=0)
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c.get('warmup', 6.0) / DT)):
                y = rk4_step(y, DT, p_arr)
            y0 = y.copy()
            true_traj = simulate_trajectory(y0, num_steps, DT, p_arr)

        with torch.no_grad():
            s_t = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
            p_t = torch.tensor(p_val, dtype=torch.float32).unsqueeze(0)
            c_t0 = s_t[:, 0:1] + s_t[:, 1:2] + s_t[:, 2:3]
            pred_traj_t = model.forward_rollout(s_t, p_t, num_steps=num_steps, target_ctot=c_t0)
            pred_traj = pred_traj_t[0].cpu().numpy()
            pred_traj = np.vstack([y0, pred_traj])

        m1_true = true_traj[:, 2]
        m1_pred = pred_traj[:, 2]

        with torch.no_grad():
            mu_v, omega_v, c1_v, _ = model.nf_params(p_t)
            ss_v = model.ss_net(p_t)

        if c['is_steady_state']:
            t_amp = 0.0
            p_amp = float(np.ptp(m1_pred[-int(0.3 * len(m1_pred)):]))
            err_ss = abs(ss_v[0, 2].item() - y0[2])
            print(f"[{c['id']:3s}] SS  | True SS m1*={y0[2]:6.1f} | Pred SS m1*={ss_v[0, 2].item():6.1f} (err={err_ss:4.1f}) | Pred Amp={p_amp:6.4f} mM | mu={mu_v.item():5.2f}", flush=True)
        else:
            res_true = analyze_signal(m1_true, prominence=c['prominence'], min_dist=c['min_dist'], dt=DT)
            res_pred = analyze_signal(m1_pred, prominence=c['prominence'], min_dist=c['min_dist'], dt=DT)
            t_amp = float(res_true['steady_amp'])
            t_per = float(res_true['period'])
            p_amp = float(res_pred['steady_amp'])
            p_per = float(res_pred['period'])
            amp_err = abs(p_amp - t_amp) / t_amp * 100.0
            per_err = abs(p_per - t_per) / t_per * 100.0 if t_per > 0 and p_per > 0 else 0.0
            print(f"[{c['id']:3s}] OSC | True Per={t_per:.3f}s | Pred Per={p_per:.3f}s (err={per_err:4.1f}%) | True Amp={t_amp:6.1f} | Pred Amp={p_amp:6.1f} (err={amp_err:4.1f}%) | mu={mu_v.item():5.2f}", flush=True)

if __name__ == "__main__":
    run()
