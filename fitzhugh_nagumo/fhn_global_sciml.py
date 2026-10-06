"""
==============================================================================
RESEARCH-GRADE SCIENTIFIC MACHINE LEARNING FRAMEWORK:
Global Dynamics, Limit-Cycle Attractors & Hopf Bifurcations of FitzHugh-Nagumo
==============================================================================
System Equations:
  dv/dt = v - (v^3)/3 - w + I
  dw/dt = eps * (v + a - b * w)

Parameters:
  I   : Injected / stimulus current (primary bifurcation parameter)
  a   : Activation / nullcline threshold
  b   : Recovery damping / nullcline slope
  eps : Time-scale ratio (fast activator vs. slow recovery, default eps = 0.08)

Analytical Hopf Bifurcation:
  Fixed point (v*, w*) satisfies: (v*^3)/3 + (1/b - 1)*v* + (a/b - I) = 0
  Tr(J) = 1 - (v*)^2 - eps * b = 0  ==>  |v*| = sqrt(1 - eps * b)
  I_Hopf,1 (lower onset) = - (1-eps*b)^(3/2)/3 - (1/b - 1)*sqrt(1-eps*b) + a/b
  I_Hopf,2 (upper block) = + (1-eps*b)^(3/2)/3 + (1/b - 1)*sqrt(1-eps*b) + a/b
==============================================================================
"""

import os
import sys

# Prevent OpenBLAS/OMP thread exhaustion and memory errors on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import random
import numpy as np
import torch
import torch.nn as nn
from numba import njit
from scipy.signal import find_peaks
from scipy.optimize import root_scalar

torch.set_num_threads(1)

# Default parameter configurations
DEFAULT_PARAMS = np.array([0.8, 0.7, 0.8, 0.08], dtype=np.float64) # [I, a, b, eps]
DEFAULT_IC = np.array([-1.0, -0.5], dtype=np.float64)
DT = 0.05
VAR_NAMES = ["v", "w"]


def set_seed(seed=42):
    """Guarantees bit-for-bit reproducibility across all libraries and hardware."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 1. NUMBA-JIT STIFF DYNAMICAL INTEGRATOR
# ============================================================
@njit(cache=True, fastmath=True)
def fhn_deriv(v, w, I, a, b, eps):
    dv = v - (v * v * v) / 3.0 - w + I
    dw = eps * (v + a - b * w)
    return dv, dw

@njit(cache=True, fastmath=True)
def rk4_step(y, dt, params):
    v = y[0]
    w = y[1]
    I = params[0]
    a = params[1]
    b = params[2]
    eps = params[3]

    k1v, k1w = fhn_deriv(v, w, I, a, b, eps)
    k2v, k2w = fhn_deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, I, a, b, eps)
    k3v, k3w = fhn_deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, I, a, b, eps)
    k4v, k4w = fhn_deriv(v + dt * k3v, w + dt * k3w, I, a, b, eps)

    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    w_next = w + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
    return np.array([v_next, w_next], dtype=np.float64)

@njit(cache=True, fastmath=True)
def simulate_trajectory(y0, steps, dt, params):
    traj = np.empty((steps + 1, 2), dtype=np.float64)
    traj[0] = y0
    curr = y0.copy()
    for i in range(steps):
        curr = rk4_step(curr, dt, params)
        traj[i + 1] = curr
    return traj


# ============================================================
# 2. ANALYTICAL BIFURCATION & STEADY-STATE SOLVER
# ============================================================
def analytical_fixed_point(I, a, b, eps):
    """
    Finds real fixed point v* by solving the cubic:
    f(v) = v^3 / 3 + (1/b - 1)*v + (a/b - I) = 0
    w* = (v* + a) / b
    """
    def cubic(v):
        return (v**3) / 3.0 + (1.0 / b - 1.0) * v + (a / b - I)
    
    # Bracket search: cubic has positive derivative for large |v|
    res = root_scalar(cubic, bracket=[-10.0, 10.0], method='brentq')
    v_star = res.root
    w_star = (v_star + a) / b
    return v_star, w_star

def analytical_hopf_currents(a, b, eps):
    """
    Computes exact analytical Hopf bifurcation thresholds (I_H1, I_H2)
    where Tr(J) = 1 - (v*)^2 - eps*b = 0.
    """
    rad = 1.0 - eps * b
    if rad <= 0.0:
        return None, None
    v_h = np.sqrt(rad)
    v1 = -v_h
    v2 = +v_h
    I1 = (v1**3) / 3.0 + (1.0 / b - 1.0) * v1 + a / b
    I2 = (v2**3) / 3.0 + (1.0 / b - 1.0) * v2 + a / b
    return I1, I2


def measure_period_and_amplitude(traj_v, dt=DT, min_amp=1.0):
    """
    Measures limit-cycle period and peak-to-peak amplitude.
    Uses prominence-based peak detection on the asymptotic section.
    """
    n = len(traj_v)
    asympt = traj_v[n // 2:]
    amp = float(np.ptp(asympt))
    if amp < min_amp or np.isnan(amp):
        return 0.0, amp

    min_dist = max(int(10.0 / dt), 10)
    prom = max(0.25 * amp, 0.5)
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
    """Feature-wise Linear Modulation block conditioned on kinetic/stimulus parameters."""
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

class ProposedFiLMFHNMLP(nn.Module):
    """
    Proposed FiLM-Conditioned Physics-Informed SciML Architecture:
    - Layer-wise FiLM modulation explicitly conditions the neural vector field on [I, a, b, eps].
    - Normalization buffers for stable learning across wide parameter regimes.
    - Residual Euler/RK integration: y_{t+dt} = y_t + dt * f_theta(y_t, p).
    """
    def __init__(self, hidden_dim=128, dt=DT):
        super().__init__()
        self.dt = dt
        # State scale [v ~ 4.0, w ~ 2.0]
        self.register_buffer('state_scales', torch.tensor([4.0, 2.0], dtype=torch.float32))
        # Parameter scale [I ~ 2.0, a ~ 1.0, b ~ 1.5, eps ~ 0.1]
        self.register_buffer('param_scales', torch.tensor([2.0, 1.0, 1.5, 0.1], dtype=torch.float32))
        # Derivative scale [dv/dt ~ 5.0, dw/dt ~ 0.5]
        self.register_buffer('deriv_scales', torch.tensor([5.0, 0.5], dtype=torch.float32))

        cond_dim = 32
        self.p_net = nn.Sequential(
            nn.Linear(4, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU()
        )
        self.block1 = FiLMBlock(2, hidden_dim, cond_dim)
        self.block2 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.block3 = FiLMBlock(hidden_dim, hidden_dim, cond_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def predict_derivatives(self, state, params):
        v = state[:, 0:1]
        w = state[:, 1:2]
        I = params[:, 0:1]
        a = params[:, 1:2]
        b = params[:, 2:3]
        eps = params[:, 3:4]

        # Prior Physical Vector Field
        phys_dv = v - (v * v * v) / 3.0 - w + I
        phys_dw = eps * (v + a - b * w)

        # Parameter-Conditioned Residual Correction
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)
        h = self.block1(s_norm, p_embed)
        h = self.block2(h, p_embed)
        h = self.block3(h, p_embed)
        res = self.head(h)

        dv = phys_dv + 0.1 * res[:, 0:1]
        dw = phys_dw + 0.05 * res[:, 1:2]
        return torch.cat([dv, dw], dim=1)

    def forward(self, state, params):
        # Fast Euler integration step for BPTT training
        derivs = self.predict_derivatives(state, params)
        return state + self.dt * derivs

    def rk4_step_forward(self, state, params):
        # 4th-order Runge-Kutta step for high-precision inference and rollout
        dt = self.dt
        k1 = self.predict_derivatives(state, params)
        k2 = self.predict_derivatives(state + 0.5 * dt * k1, params)
        k3 = self.predict_derivatives(state + 0.5 * dt * k2, params)
        k4 = self.predict_derivatives(state + dt * k3, params)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)



class UnconstrainedBaselineFHNMLP(nn.Module):
    """
    Standard Black-Box Unconstrained MLP Baseline:
    Directly concatenates [v, w, I, a, b, eps] without FiLM modulation.
    """
    def __init__(self, hidden_dim=128, dt=DT):
        super().__init__()
        self.dt = dt
        self.register_buffer('state_scales', torch.tensor([4.0, 2.0], dtype=torch.float32))
        self.register_buffer('param_scales', torch.tensor([2.0, 1.0, 1.5, 0.1], dtype=torch.float32))
        self.register_buffer('deriv_scales', torch.tensor([5.0, 0.5], dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, state, params):
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        x = torch.cat([s_norm, p_norm], dim=1)
        derivs = self.net(x) * self.deriv_scales
        return state + self.dt * derivs


# ============================================================
# 4. BASIN-WIDE DATASET GENERATION
# ============================================================
def build_fhn_dataset(parameter_grid, traj_duration=150.0, dt=DT, num_ics_per_param=3):
    """
    Generates training trajectories across the parameter grid.
    Perturbs initial conditions across the phase plane to learn the attractor basin.
    """
    print(f"Generating FHN dataset across {len(parameter_grid)} parameter combinations...", flush=True)
    steps = int(traj_duration / dt)
    dataset = []

    for p in parameter_grid:
        p_arr = np.array(p, dtype=np.float64)
        I, a, b, eps = p[0], p[1], p[2], p[3]
        v_star, w_star = analytical_fixed_point(I, a, b, eps)

        # Baseline asymptotic point via warmup
        y_warm = np.array([v_star + 0.1, w_star + 0.1], dtype=np.float64)
        for _ in range(int(100.0 / dt)):
            y_warm = rk4_step(y_warm, dt, p_arr)

        # Multiple initial conditions: on-attractor and basin perturbations
        for ic_idx in range(num_ics_per_param):
            if ic_idx == 0:
                y0 = y_warm.copy()
            else:
                scale_v = 0.5 * ic_idx
                scale_w = 0.3 * ic_idx
                y0 = np.array([y_warm[0] + scale_v * np.sin(ic_idx * 1.5),
                               y_warm[1] + scale_w * np.cos(ic_idx * 1.5)], dtype=np.float64)
            
            traj = simulate_trajectory(y0, steps, dt, p_arr)
            dataset.append({
                "params": p_arr,
                "traj": traj
            })

    print(f"  Dataset complete: {len(dataset)} trajectories ({steps} steps each).", flush=True)
    return dataset


# ============================================================
# 5. CURRICULAR MULTI-STEP BPTT TRAINING ENGINE
# ============================================================
def train_model(model, train_dataset, val_dataset, total_steps=2500, batch_size=64, lr=2e-3, seed=42):
    """
    Curricular training engine:
    Ramps unroll horizon K from 4 to 20 steps over total_steps.
    """
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    def sample_batch(ds, K):
        states, params, future = [], [], []
        indices = np.random.randint(0, len(ds), size=batch_size)
        for idx in indices:
            item = ds[idx]
            traj = item["traj"]
            max_t = len(traj) - K - 1
            t0 = np.random.randint(0, max_t)
            states.append(traj[t0])
            params.append(item["params"])
            future.append(traj[t0 + 1 : t0 + 1 + K])
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(params), dtype=torch.float32),
                torch.tensor(np.array(future), dtype=torch.float32))

    best_val_loss = float("inf")
    best_weights = None
    start_time = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        # Curriculum: K ramps from 4 to 20
        progress = min(1.0, step / float(total_steps * 0.75))
        K = int(4 + progress * 16)

        states, params, future = sample_batch(train_dataset, K)
        pred_traj = []
        curr = states
        for k in range(K):
            curr = model(curr, params)
            pred_traj.append(curr.unsqueeze(1))
        pred_traj = torch.cat(pred_traj, dim=1)

        loss = nn.functional.mse_loss(pred_traj, future)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation check every 250 steps
        if step % 250 == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                val_states, val_params, val_future = sample_batch(val_dataset, K=15)
                val_pred = []
                curr = val_states
                for _ in range(15):
                    curr = model(curr, val_params)
                    val_pred.append(curr.unsqueeze(1))
                val_pred = torch.cat(val_pred, dim=1)
                val_loss = float(nn.functional.mse_loss(val_pred, val_future))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

            elapsed = time.time() - start_time
            print(f"  Step {step:4d}/{total_steps} (K={K:2d}) | Train Loss: {loss.item():.6e} | Val Loss: {val_loss:.6e} | {elapsed:.1f}s", flush=True)

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"Training completed. Best Val Loss: {best_val_loss:.6e}", flush=True)
    return model
