"""
======================================================================================
RESEARCH-GRADE SCIENTIFIC MACHINE LEARNING FRAMEWORK:
Van der Pol Relaxation Oscillator & Hopf Bifurcation Analysis
======================================================================================
Governing Equations:
  dx/dt = v
  dv/dt = mu * (1 - x^2) * v - x

Analytical Bifurcation Ground Truth:
  Fixed point: (x*, v*) = (0, 0)
  Jacobian: J = [[0, 1], [-1, mu]]
  Eigenvalues: lambda = (mu +- sqrt(mu^2 - 4)) / 2
  Supercritical Hopf Bifurcation at mu_c = 0.0:
    - mu < 0: Stable focus / node at origin
    - mu = 0: Center (Hopf point)
    - mu > 0: Unstable origin with stable limit cycle
======================================================================================
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from numba import njit
from scipy.signal import find_peaks

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DT = 0.05
BENCHMARK_PARAMS = [1.0]  # mu = 1.0 canonical benchmark

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================
# 1. NUMBA-JIT STIFF DYNAMICAL INTEGRATOR
# ============================================================
@njit(cache=True, fastmath=True)
def vdp_deriv(x, v, mu):
    dx = v
    dv = mu * (1.0 - x * x) * v - x
    return dx, dv

@njit(cache=True, fastmath=True)
def rk4_step(y, dt, mu):
    x = y[0]
    v = y[1]
    k1x, k1v = vdp_deriv(x, v, mu)
    k2x, k2v = vdp_deriv(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, mu)
    k3x, k3v = vdp_deriv(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, mu)
    k4x, k4v = vdp_deriv(x + dt * k3x, v + dt * k3v, mu)

    x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return np.array([x_next, v_next], dtype=np.float64)

@njit(cache=True, fastmath=True)
def simulate_trajectory(y0, steps, dt, mu):
    traj = np.empty((steps + 1, 2), dtype=np.float64)
    traj[0] = y0
    curr = y0.copy()
    for i in range(steps):
        curr = rk4_step(curr, dt, mu)
        traj[i + 1] = curr
    return traj

# ============================================================
# 2. ANALYTICAL BIFURCATION & MEASUREMENT UTILITIES
# ============================================================
def analytical_hopf_threshold():
    return 0.0

def measure_period_and_amplitude(traj_x, dt=DT, min_amp=0.5):
    n = len(traj_x)
    asympt = traj_x[n // 2:]
    amp = float(np.ptp(asympt))
    if amp < min_amp or np.isnan(amp):
        return 0.0, amp

    min_dist = max(int(2.0 / dt), 10)
    prom = max(0.25 * amp, 0.2)
    peaks, _ = find_peaks(asympt, distance=min_dist, prominence=prom)
    if len(peaks) > 1:
        period = float(np.mean(np.diff(peaks)) * dt)
    else:
        period = 0.0
    return period, amp

# ============================================================
# 3. SCIENTIFIC MACHINE LEARNING ARCHITECTURES
# ============================================================
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

class ProposedFiLMVDPMLP(nn.Module):
    """
    Proposed FiLM-Conditioned SciML Architecture for Van der Pol:
    - Conditions acceleration field dv/dt on parameter mu.
    - Preserves exact kinematic relation dx/dt = v.
    - Learns vector field from scratch (or optionally residual).
    """
    def __init__(self, hidden_dim=64, dt=DT, embed_prior=False):
        super().__init__()
        self.dt = dt
        self.embed_prior = embed_prior
        
        # State scale [x ~ 3.0, v ~ 3.0]
        self.register_buffer('state_scales', torch.tensor([3.0, 3.0], dtype=torch.float32))
        # Parameter scale [mu ~ 2.0]
        self.register_buffer('param_scales', torch.tensor([2.0], dtype=torch.float32))

        cond_dim = 16
        self.p_net = nn.Sequential(
            nn.Linear(1, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU()
        )
        self.block1 = FiLMBlock(2, hidden_dim, cond_dim)
        self.block2 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.block3 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def predict_derivatives(self, state, params):
        x = state[:, 0:1]
        v = state[:, 1:2]
        mu = params[:, 0:1]

        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)
        
        h = self.block1(s_norm, p_embed)
        h = self.block2(h, p_embed)
        h = self.block3(h, p_embed)
        neural_accel = self.head(h) * 3.0  # scale to match dv/dt magnitude

        dx = v
        if self.embed_prior:
            phys_dv = mu * (1.0 - x * x) * v - x
            dv = phys_dv + 0.1 * neural_accel
        else:
            dv = neural_accel

        return torch.cat([dx, dv], dim=1)

    def forward(self, state, params):
        # Euler step for fast BPTT unrolling
        derivs = self.predict_derivatives(state, params)
        return state + self.dt * derivs

    def rk4_step_forward(self, state, params):
        # 4th-order Runge-Kutta for precision inference
        dt = self.dt
        k1 = self.predict_derivatives(state, params)
        k2 = self.predict_derivatives(state + 0.5 * dt * k1, params)
        k3 = self.predict_derivatives(state + 0.5 * dt * k2, params)
        k4 = self.predict_derivatives(state + dt * k3, params)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class UnconstrainedBaselineVDPMLP(nn.Module):
    """
    Standard Black-Box MLP Baseline without kinematic constraint:
    Maps [x, v, mu] -> [dx/dt, dv/dt] directly.
    """
    def __init__(self, hidden_dim=64, dt=DT):
        super().__init__()
        self.dt = dt
        self.register_buffer('state_scales', torch.tensor([3.0, 3.0], dtype=torch.float32))
        self.register_buffer('param_scales', torch.tensor([2.0], dtype=torch.float32))
        self.register_buffer('deriv_scales', torch.tensor([3.0, 3.0], dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, state, params):
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        inp = torch.cat([s_norm, p_norm], dim=1)
        derivs = self.net(inp) * self.deriv_scales
        return state + self.dt * derivs

    def rk4_step_forward(self, state, params):
        dt = self.dt
        def f(s):
            s_norm = s / self.state_scales
            p_norm = params / self.param_scales
            return self.net(torch.cat([s_norm, p_norm], dim=1)) * self.deriv_scales
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# 4. DATASET GENERATION
# ============================================================
def build_vdp_dataset(parameter_grid, traj_duration=40.0, dt=DT, num_ics_per_param=4):
    print(f"Generating VDP dataset across {len(parameter_grid)} parameter combinations...", flush=True)
    steps = int(traj_duration / dt)
    dataset = []

    for mu in parameter_grid:
        # Initial conditions: sample across phase space
        ics = [
            np.array([2.0, 0.0], dtype=np.float64),
            np.array([0.5, 1.5], dtype=np.float64),
            np.array([-1.5, -0.5], dtype=np.float64),
            np.array([0.1, 0.1], dtype=np.float64),
        ]
        if num_ics_per_param > 4:
            for _ in range(num_ics_per_param - 4):
                ics.append(np.random.uniform(low=[-2.5, -2.5], high=[2.5, 2.5]))

        for y0 in ics[:num_ics_per_param]:
            traj = simulate_trajectory(y0, steps, dt, mu)
            dataset.append({
                "traj": torch.tensor(traj, dtype=torch.float32),
                "params": torch.tensor([mu], dtype=torch.float32),
                "mu": mu
            })

    print(f"Generated {len(dataset)} trajectories ({steps} steps each).", flush=True)
    return dataset


# ============================================================
# 5. TRAINING ROUTINE
# ============================================================
def train_model(model, dataset, epochs=50, batch_size=32, lr=2e-3, seed=42):
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    k_unroll = 10
    total_samples = len(dataset)
    num_steps_per_epoch = 150

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(num_steps_per_epoch):
            batch_traj = []
            batch_params = []
            for _ in range(batch_size):
                idx = np.random.randint(total_samples)
                item = dataset[idx]
                traj = item["traj"]
                max_start = len(traj) - k_unroll - 1
                start = np.random.randint(max_start)
                batch_traj.append(traj[start:start + k_unroll + 1])
                batch_params.append(item["params"])

            batch_traj = torch.stack(batch_traj, dim=0)    # [B, K+1, 2]
            batch_params = torch.stack(batch_params, dim=0) # [B, 1]

            pred_states = []
            curr = batch_traj[:, 0, :]
            for step in range(k_unroll):
                curr = model(curr, batch_params)
                pred_states.append(curr)

            pred_states = torch.stack(pred_states, dim=1)   # [B, K, 2]
            target_states = batch_traj[:, 1:, :]           # [B, K, 2]

            loss = criterion(pred_states, target_states)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs:
            avg_loss = epoch_loss / num_steps_per_epoch
            print(f"  [Epoch {epoch:2d}/{epochs:2d}] Loss: {avg_loss:.6e}", flush=True)

    return model
