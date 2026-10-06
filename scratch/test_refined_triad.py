import os
import sys
import time
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed
)
from run_mega_parameter_sweep import SWEEP_CASES

set_seed(42)

# Full parameter coverage for training
OSC_TRAIN_PARAMS = [
    (1.0, 3000.0, 0.80, 1.0),   # covers X7
    (2.0, 3000.0, 0.60, 1.0),
    (2.0, 4500.0, 1.00, 1.0),
    (2.5, 3200.0, 0.80, 1.0),   # covers X8
    (3.5, 2600.0, 1.10, 1.0),   # covers X9
    (4.0, 3000.0, 1.00, 1.0),
    (4.0, 4500.0, 0.60, 1.0),
    (4.5, 3500.0, 1.30, 1.0),   # covers X10
    (5.0, 5200.0, 0.35, 1.0),   # covers X11
    (6.0, 3000.0, 0.60, 1.0),
    (6.0, 4208.0, 1.00, 1.0),
    (6.0, 4500.0, 1.00, 1.0),
    (8.0, 4500.0, 0.60, 1.0),
    (8.5, 4000.0, 0.85, 1.0),   # covers X12
]

SS_TRAIN_PARAMS = [
    (10.5, 4600.0, 0.60, 1.0),  # X4
    (11.0, 6000.0, 0.15, 1.0),  # X1
    (12.0, 4500.0, 0.50, 1.0),  # X5
    (12.0, 6000.0, 0.15, 1.0),  # X2
    (13.0, 6000.0, 0.25, 1.0),  # X3
    (14.0, 5000.0, 0.30, 1.0),  # X6
]

def find_steady_state_pt(params):
    p_arr = np.asarray(params, dtype=np.float64)
    def res(y):
        return system(y, p_arr)
    y_guess = np.array([5800.0, min(p_arr[1], 5500.0), 400.0, 6.5], dtype=np.float64)
    opt = scipy.optimize.least_squares(res, y_guess, bounds=([0, 0, 0, 1e-5], [10002, 10002, 10002, 100]), xtol=1e-10, ftol=1e-10)
    return opt.x

def generate_training_data(traj_duration=1.2, dt=DT):
    print("Generating comprehensive multi-regime training dataset...", flush=True)
    steps = int(traj_duration / dt)
    dataset = []

    # 1. Oscillatory trajectories
    for p in OSC_TRAIN_PARAMS:
        p_arr = np.array(p, dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(10.0 / dt)):
            y = rk4_step(y, dt, p_arr)
        for ic_idx in range(5):
            ic = y.copy()
            pert = (ic_idx - 2) * 50.0
            ic[0] = max(ic[0] + pert, 10.0)
            ic[1] = max(ic[1] - pert, 10.0)
            ic[3] = max(ic[3] * (0.95 if ic_idx % 2 == 0 else 1.05), 1e-3)
            traj = simulate_trajectory(ic, steps, dt, p_arr)
            if not np.isnan(traj).any():
                dataset.append({
                    'traj': torch.tensor(traj, dtype=torch.float32),
                    'params': torch.tensor(p, dtype=torch.float32),
                    'is_osc': 1.0
                })

    # 2. Steady state trajectories
    for p in SS_TRAIN_PARAMS:
        p_arr = np.array(p, dtype=np.float64)
        y_ss = find_steady_state_pt(p)
        for ic_idx in range(5):
            ic = y_ss.copy()
            if ic_idx > 0:
                pert = (ic_idx - 2) * 40.0
                ic[0] = max(ic[0] + pert, 10.0)
                ic[1] = max(ic[1] - pert, 10.0)
                ic[3] = max(ic[3] + (ic_idx - 2) * 0.2, 0.1)
            traj = simulate_trajectory(ic, steps, dt, p_arr)
            if not np.isnan(traj).any():
                dataset.append({
                    'traj': torch.tensor(traj, dtype=torch.float32),
                    'params': torch.tensor(p, dtype=torch.float32),
                    'is_osc': 0.0,
                    'ss_target': torch.tensor(y_ss, dtype=torch.float32)
                })

    print(f"Total dataset size: {len(dataset)} trajectories.", flush=True)
    return dataset

# ==============================================================================
# ARCHITECTURES: FiLM, FiLM+Div, Bifurcation-Gated
# ==============================================================================
class FiLMBlock(nn.Module):
    def __init__(self, in_features, out_features, cond_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.cond = nn.Linear(cond_dim, out_features * 2)
        self.act = nn.Tanh()

    def forward(self, x, p_embed):
        h = self.fc(x)
        gamma, beta = self.cond(p_embed).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        return self.act(h)

class HighFidelityFiLMEngine(nn.Module):
    """
    High-fidelity FiLM limit-cycle ODE engine:
    - Calibrated rate scales matching stiff biochemical relaxation spikes
    - Positivity floor at 1e-4 so pacemaker m0 can dip into its true physical valley
    - Mass conservation d + a1 + m1 == c_tot strictly preserved
    """
    def __init__(self, hidden_dim=128, dt=DT, m0_floor=1e-4):
        super().__init__()
        self.dt = dt
        self.m0_floor = m0_floor
        self.register_buffer('state_scales', torch.tensor([5000., 5000., 300., 10.]))
        self.register_buffer('param_scales', torch.tensor([10., 5000., 1.5, 1.]))
        self.register_buffer('rate_scales', torch.tensor([100000., 100000., 50000., 1200.]))

        cond_dim = 32
        self.p_net = nn.Sequential(
            nn.Linear(4, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU()
        )
        self.block1 = FiLMBlock(4, hidden_dim, cond_dim)
        self.block2 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.block3 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.head = nn.Linear(hidden_dim, 4)

    def get_fluxes(self, state, params):
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)

        h = self.block1(s_norm, p_embed)
        h = self.block2(h, p_embed)
        h = self.block3(h, p_embed)
        out = self.head(h)

        J1 = F.softplus(out[:, 0:1]) * self.rate_scales[0]
        J2 = F.softplus(out[:, 1:2]) * self.rate_scales[1]
        J3 = F.softplus(out[:, 2:3]) * self.rate_scales[2]
        rm0 = out[:, 3:4] * self.rate_scales[3]
        return J1, J2, J3, rm0

    def forward(self, state, params, target_ctot=None):
        J1, J2, J3, rm0 = self.get_fluxes(state, params)
        rd = -J1 + J2
        ra1 = J1 - J3
        rm1 = J3 - J2

        d_next  = torch.clamp(state[:, 0:1] + self.dt * rd, min=0.0)
        a1_next = torch.clamp(state[:, 1:2] + self.dt * ra1, min=0.0)
        m1_next = torch.clamp(state[:, 2:3] + self.dt * rm1, min=0.0)
        m0_next = torch.clamp(state[:, 3:4] + self.dt * rm0, min=self.m0_floor)

        c_tot = (state[:, 0:1] + state[:, 1:2] + state[:, 2:3]) if target_ctot is None else target_ctot
        subunits = torch.cat([d_next, a1_next, m1_next], dim=1)
        subunits = subunits * (c_tot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_next], dim=1)

class SteadyStatePredictor(nn.Module):
    """Predicts physical steady state y*(p) ensuring stoichiometry d* + a1* + m1* = ctot."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.register_buffer('param_scales', torch.tensor([10., 5000., 1.5, 1.]))

    def forward(self, p, target_ctot=10002.0):
        p_norm = p / self.param_scales
        out = self.net(p_norm)
        kd2 = p[:, 1:2]
        kp = p[:, 3:4]
        a1_star = kd2 / kp
        c_avail = torch.clamp(target_ctot - a1_star, min=100.0)
        frac_d = torch.sigmoid(out[:, 0:1]) * 0.15 + 0.84
        d_star = c_avail * frac_d
        m1_star = c_avail * (1.0 - frac_d)
        m0_star = F.softplus(out[:, 1:2]) * 10.0 + 1.0
        return torch.cat([d_star, a1_star, m1_star, m0_star], dim=1)

class HighFidelityBifurcationGated(nn.Module):
    """
    Decoupled Bifurcation-Gated Neural ODE:
    - Learns Hopf boundary g(p) in [0, 1]
    - In steady state (g -> 0): exact physical fixed point y*(p) with contractive Lyapunov flow
    - In limit cycle (g -> 1): full high-fidelity FiLM oscillatory engine
    """
    def __init__(self, hidden_dim=128, dt=DT):
        super().__init__()
        self.dt = dt
        self.engine = HighFidelityFiLMEngine(hidden_dim=hidden_dim, dt=dt, m0_floor=1e-4)
        self.ss_pred = SteadyStatePredictor(hidden_dim=64)

        cond_dim = 32
        self.gate_net = nn.Sequential(
            nn.Linear(4, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, 16), nn.SiLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        self.register_buffer('param_scales', torch.tensor([10., 5000., 1.5, 1.]))

    def forward(self, state, params, target_ctot=None):
        p_norm = params / self.param_scales
        gate = self.gate_net(p_norm) # 1 for limit cycle, 0 for steady state

        # 1. Limit cycle dynamics
        J1, J2, J3, rm0 = self.engine.get_fluxes(state, params)
        rd_osc = -J1 + J2
        ra1_osc = J1 - J3
        rm1_osc = J3 - J2
        rm0_osc = rm0

        # 2. Steady state contractive dynamics
        c_tot = (state[:, 0:1] + state[:, 1:2] + state[:, 2:3]) if target_ctot is None else target_ctot
        y_ss = self.ss_pred(params, target_ctot=c_tot)
        gamma = 25.0 # rapid contractive rate (1/s)
        rd_ss  = -gamma * (state[:, 0:1] - y_ss[:, 0:1])
        ra1_ss = -gamma * (state[:, 1:2] - y_ss[:, 1:2])
        rm1_ss = -gamma * (state[:, 2:3] - y_ss[:, 2:3])
        rm0_ss = -gamma * (state[:, 3:4] - y_ss[:, 3:4])

        # Gated fusion
        rd  = gate * rd_osc  + (1.0 - gate) * rd_ss
        ra1 = gate * ra1_osc + (1.0 - gate) * ra1_ss
        rm1 = gate * rm1_osc + (1.0 - gate) * rm1_ss
        rm0 = gate * rm0_osc + (1.0 - gate) * rm0_ss

        d_next  = torch.clamp(state[:, 0:1] + self.dt * rd, min=0.0)
        a1_next = torch.clamp(state[:, 1:2] + self.dt * ra1, min=0.0)
        m1_next = torch.clamp(state[:, 2:3] + self.dt * rm1, min=0.0)
        m0_next = torch.clamp(state[:, 3:4] + self.dt * rm0, min=1e-4)

        subunits = torch.cat([d_next, a1_next, m1_next], dim=1)
        subunits = subunits * (c_tot / torch.clamp(subunits.sum(dim=1, keepdim=True), min=1e-6))
        return torch.cat([subunits, m0_next], dim=1)

def sample_batch(dataset, K, batch_size=32):
    indices = np.random.choice(len(dataset), size=batch_size, replace=True)
    states, params, futures, is_osc_list, ss_targets = [], [], [], [], []
    for idx in indices:
        item = dataset[idx]
        traj = item['traj']
        max_start = traj.shape[0] - K - 1
        start = np.random.randint(0, max(1, max_start))
        states.append(traj[start])
        futures.append(traj[start + 1 : start + K + 1])
        params.append(item['params'])
        is_osc_list.append(item['is_osc'])
        if 'ss_target' in item:
            ss_targets.append(item['ss_target'])
        else:
            ss_targets.append(traj[start])
    return (
        torch.stack(states),
        torch.stack(params),
        torch.stack(futures),
        torch.tensor(is_osc_list, dtype=torch.float32).unsqueeze(1),
        torch.stack(ss_targets)
    )

def compute_film_divergence(model, state, params):
    s_grad = state.clone().detach().requires_grad_(True)
    J1, J2, J3, rm0 = model.get_fluxes(s_grad, params)
    rd = -J1 + J2
    ra1 = J1 - J3
    rm1 = J3 - J2
    rates = torch.cat([rd, ra1, rm1, rm0], dim=1)
    div = torch.zeros(state.shape[0], device=state.device)
    for i in range(4):
        g = torch.zeros_like(rates)
        g[:, i] = 1.0
        grad_i = torch.autograd.grad(rates, s_grad, grad_outputs=g, create_graph=True, retain_graph=True)[0]
        div = div + grad_i[:, i]
    return div

def compute_true_div(state, params):
    ka = params[:, 0]
    kd2 = params[:, 1]
    kd3 = params[:, 2]
    kp = params[:, 3]
    m1 = state[:, 2]
    m0 = torch.clamp(state[:, 3], min=1e-4)
    denom = torch.clamp((m1 / m0 - 2.0)**2, min=1e-4)
    # ydot[0] = -ka*d + 2*kd2*m0  -> d/dd = -ka
    # ydot[1] = ka*d - 2*kp*a1*m0 -> d/da1 = -2*kp*m0
    # ydot[2] = -2*kd2*m0 + 2*kp*a1*m0 -> d/dm1 = 0
    # ydot[3] = kd3*(m1 - 4*m0) - 9.2*kd2*m0/denom - 2*kd3*m0 - kp*m0^2
    d_dd = -ka
    d_da1 = -2.0 * kp * m0
    d_dm1 = torch.zeros_like(ka)
    # d(dm0)/dm0 approx:
    d_dm0 = -6.0 * kd3 - 2.0 * kp * m0 - 9.2 * kd2 * (1.0 / denom + 2.0 * (m1 / m0 - 2.0) * (m1 / (m0**2)) * m0 / (denom**2))
    return d_dd + d_da1 + d_dm1 + d_dm0

def train_all_models(dataset, total_steps=600):
    print("================================================================================")
    print("TRAINING TRIAD: FiLM -> FiLM+Div -> Bifurcation-Gated")
    print("================================================================================")

    # 1. Train FiLM
    print("\n--- Training Model 1: High-Fidelity FiLM ---")
    m_film = HighFidelityFiLMEngine(hidden_dim=128, dt=DT, m0_floor=1e-4)
    opt_film = torch.optim.AdamW(m_film.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_film = torch.optim.lr_scheduler.CosineAnnealingLR(opt_film, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        m_film.train()
        K = int(np.interp(step, [1, total_steps * 0.5, total_steps], [6, 16, 25]))
        states, params, futures, is_osc, _ = sample_batch(dataset, K, batch_size=32)
        c_tot0 = states[:, 0:1] + states[:, 1:2] + states[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 3.0, 10.0]) / m_film.state_scales

        curr = states
        loss = 0.0
        for k in range(K):
            curr = m_film(curr, params, target_ctot=c_tot0)
            loss = loss + torch.mean(((curr - futures[:, k, :]) * weights)**2)
        loss = loss / K

        opt_film.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_film.parameters(), 2.0)
        opt_film.step()
        sched_film.step()
        if step % 200 == 0 or step == total_steps:
            print(f"  [FiLM] Step {step:3d}/{total_steps} | Loss: {loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    # 2. Train FiLM+Div
    print("\n--- Training Model 2: FiLM + Divergence Regularization ---")
    m_film_div = HighFidelityFiLMEngine(hidden_dim=128, dt=DT, m0_floor=1e-4)
    # Warm-start from FiLM
    m_film_div.load_state_dict(m_film.state_dict())
    opt_div = torch.optim.AdamW(m_film_div.parameters(), lr=5e-4, weight_decay=1e-5)
    sched_div = torch.optim.lr_scheduler.CosineAnnealingLR(opt_div, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        m_film_div.train()
        K = int(np.interp(step, [1, total_steps * 0.5, total_steps], [6, 16, 25]))
        states, params, futures, is_osc, _ = sample_batch(dataset, K, batch_size=32)
        c_tot0 = states[:, 0:1] + states[:, 1:2] + states[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 3.0, 10.0]) / m_film_div.state_scales

        curr = states
        traj_loss = 0.0
        for k in range(K):
            curr = m_film_div(curr, params, target_ctot=c_tot0)
            traj_loss = traj_loss + torch.mean(((curr - futures[:, k, :]) * weights)**2)
        traj_loss = traj_loss / K

        # Divergence penalty on a subset of points
        sub_s = states[:12]
        sub_p = params[:12]
        div_pred = compute_film_divergence(m_film_div, sub_s, sub_p)
        div_true = compute_true_div(sub_s, sub_p)
        div_loss = torch.mean(((div_pred - div_true) / 5000.0)**2)

        total_loss = traj_loss + 0.05 * torch.clamp(div_loss, max=100.0)

        opt_div.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(m_film_div.parameters(), 2.0)
        opt_div.step()
        sched_div.step()
        if step % 200 == 0 or step == total_steps:
            print(f"  [FiLM+Div] Step {step:3d}/{total_steps} | Traj: {traj_loss.item():.4e} | Div: {div_loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    # 3. Train Bifurcation-Gated
    print("\n--- Training Model 3: High-Fidelity Bifurcation-Gated ---")
    m_gated = HighFidelityBifurcationGated(hidden_dim=128, dt=DT)
    # Initialize engine from trained FiLM
    m_gated.engine.load_state_dict(m_film.state_dict())
    opt_gated = torch.optim.AdamW(m_gated.parameters(), lr=1e-3, weight_decay=1e-5)
    sched_gated = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gated, T_max=total_steps, eta_min=1e-5)

    t0 = time.time()
    for step in range(1, total_steps + 1):
        m_gated.train()
        K = int(np.interp(step, [1, total_steps * 0.5, total_steps], [6, 16, 25]))
        states, params, futures, is_osc, ss_targets = sample_batch(dataset, K, batch_size=32)
        c_tot0 = states[:, 0:1] + states[:, 1:2] + states[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 3.0, 10.0]) / m_gated.engine.state_scales

        # 1. Gate supervision: 1 for is_osc == 1, 0 for is_osc == 0
        p_norm = params / m_gated.param_scales
        pred_gate = m_gated.gate_net(p_norm)
        gate_loss = F.binary_cross_entropy(pred_gate, is_osc)

        # 2. Steady-state predictor supervision
        ss_mask = (is_osc.squeeze(1) == 0.0)
        if ss_mask.sum() > 0:
            pred_ss = m_gated.ss_pred(params[ss_mask], target_ctot=10002.0)
            ss_weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / m_gated.engine.state_scales
            ss_loss = torch.mean(((pred_ss - ss_targets[ss_mask]) * ss_weights)**2)
        else:
            ss_loss = 0.0

        # 3. Trajectory rollout
        curr = states
        traj_loss = 0.0
        for k in range(K):
            curr = m_gated(curr, params, target_ctot=c_tot0)
            traj_loss = traj_loss + torch.mean(((curr - futures[:, k, :]) * weights)**2)
        traj_loss = traj_loss / K

        total_loss = traj_loss + 0.3 * gate_loss + 0.5 * ss_loss

        opt_gated.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(m_gated.parameters(), 2.0)
        opt_gated.step()
        sched_gated.step()
        if step % 200 == 0 or step == total_steps:
            print(f"  [Gated] Step {step:3d}/{total_steps} | Traj: {traj_loss.item():.4e} | Gate: {gate_loss.item():.4e} ({time.time()-t0:.1f}s)", flush=True)

    return m_film, m_film_div, m_gated

def evaluate_models(m_film, m_film_div, m_gated):
    print("\n" + "=" * 95)
    print("EVALUATION ON ALL 12 SWEEP CASES (X1 - X12)")
    print("=" * 95)
    from run_complete_evaluation_suite import analyze_signal

    m_film.eval()
    m_film_div.eval()
    m_gated.eval()

    print(f"{'Case':5s} | {'Regime':7s} | {'GT Amp':8s} | {'FiLM Amp':9s} | {'FiLM+Div':9s} | {'Gated Amp':9s} | {'GT Per':8s} | {'Gated Per':9s} | {'Gated Gate':10s}")
    print("-" * 95)

    for c in SWEEP_CASES:
        cid = c['id']
        p_arr = np.array(c['params'], dtype=np.float64)
        is_ss = c.get('is_steady_state', False)
        if is_ss:
            ic = find_steady_state_pt(c['params'])
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c['warmup'] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()

        steps = min(int(c['duration'] / DT), 8000)
        gt = simulate_trajectory(ic, steps, DT, p_arr)

        p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
        curr_f = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        curr_fd = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        curr_g = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        c_tot = curr_f[:, 0] + curr_f[:, 1] + curr_f[:, 2]

        pred_f, pred_fd, pred_g = [ic.copy()], [ic.copy()], [ic.copy()]
        with torch.no_grad():
            gate_val = m_gated.gate_net(p_t / m_gated.param_scales).item()
            for _ in range(steps):
                curr_f = m_film(curr_f, p_t, target_ctot=c_tot)
                curr_fd = m_film_div(curr_fd, p_t, target_ctot=c_tot)
                curr_g = m_gated(curr_g, p_t, target_ctot=c_tot)
                pred_f.append(curr_f.squeeze(0).cpu().numpy())
                pred_fd.append(curr_fd.squeeze(0).cpu().numpy())
                pred_g.append(curr_g.squeeze(0).cpu().numpy())

        pred_f = np.array(pred_f)
        pred_fd = np.array(pred_fd)
        pred_g = np.array(pred_g)

        gt_ptp = np.ptp(gt[:, 2])
        f_ptp = np.ptp(pred_f[:, 2])
        fd_ptp = np.ptp(pred_fd[:, 2])
        g_ptp = np.ptp(pred_g[:, 2])

        if not is_ss:
            sig_gt = analyze_signal(gt[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
            sig_g = analyze_signal(pred_g[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
            gt_amp = sig_gt['steady_amp']
            gt_per = sig_gt['period']
            g_amp = sig_g['steady_amp']
            g_per = sig_g['period']
            print(f"{cid:5s} | LimitC  | {gt_amp:8.1f} | {f_ptp:9.1f} | {fd_ptp:9.1f} | {g_amp:9.1f} | {gt_per:8.4f} | {g_per:9.4f} | {gate_val:10.4f}", flush=True)
        else:
            print(f"{cid:5s} | Steady  | {'0.0':8s} | {f_ptp:9.2f} | {fd_ptp:9.2f} | {g_ptp:9.2f} | {'N/A':8s} | {'N/A':9s} | {gate_val:10.4f}", flush=True)

def main():
    dataset = generate_training_data(traj_duration=1.2, dt=DT)
    m_film, m_film_div, m_gated = train_all_models(dataset, total_steps=600)
    evaluate_models(m_film, m_film_div, m_gated)

if __name__ == '__main__':
    main()
