"""
Tests the accuracy improvement for FitzHugh-Nagumo SciML using:
1. Residual Physical Formulation: embeds exact linear coupling dw/dt = eps*(v+a-bw)
2. Boundary-Focused Sampling: fine-grained sampling in the bifurcation corridor [0.28, 0.38]
3. Neural RK4 multi-step rollout
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fhn_global_sciml import (
    ProposedFiLMFHNMLP,
    simulate_trajectory,
    rk4_step,
    analytical_fixed_point,
    analytical_hopf_currents,
    measure_period_and_amplitude,
    set_seed,
    DT
)

class EnhancedPhysicalFiLMMLP(nn.Module):
    """
    Enhanced Physics-Residual FiLM Architecture:
    - Embeds known linear recovery dynamics: dw/dt = eps*(v + a - b*w) exactly.
    - Neural network only learns the nonlinear activation vector field: dv/dt = v - v^3/3 - w + I + delta_v.
    """
    def __init__(self, hidden_dim=128, dt=DT):
        super().__init__()
        self.dt = dt
        self.register_buffer('state_scales', torch.tensor([4.0, 2.0], dtype=torch.float32))
        self.register_buffer('param_scales', torch.tensor([2.0, 1.0, 1.5, 0.1], dtype=torch.float32))

        cond_dim = 32
        self.p_net = nn.Sequential(
            nn.Linear(4, cond_dim), nn.Tanh(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU()
        )
        self.fc1 = nn.Linear(2, hidden_dim)
        self.cond1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cond2 = nn.Linear(cond_dim, hidden_dim * 2)
        self.head = nn.Linear(hidden_dim, 2)
        self.act = nn.Tanh()

    def predict_derivatives(self, state, params):
        v = state[:, 0:1]
        w = state[:, 1:2]
        I = params[:, 0:1]
        a = params[:, 1:2]
        b = params[:, 2:3]
        eps = params[:, 3:4]

        # Prior physics
        phys_dv = v - (v**3)/3.0 - w + I
        phys_dw = eps * (v + a - b * w)

        # Learned residual correction
        s_norm = state / self.state_scales
        p_norm = params / self.param_scales
        p_embed = self.p_net(p_norm)

        h1 = self.fc1(s_norm)
        g1, b1 = self.cond1(p_embed).chunk(2, dim=-1)
        h1 = self.act(h1 * (1.0 + g1) + b1)

        h2 = self.fc2(h1)
        g2, b2 = self.cond2(p_embed).chunk(2, dim=-1)
        h2 = self.act(h2 * (1.0 + g2) + b2)

        res = self.head(h2)
        return torch.cat([phys_dv + 0.1 * res[:, 0:1], phys_dw + 0.05 * res[:, 1:2]], dim=1)

    def forward(self, state, params):
        return state + self.dt * self.predict_derivatives(state, params)

    def rk4_step_forward(self, state, params):
        dt = self.dt
        k1 = self.predict_derivatives(state, params)
        k2 = self.predict_derivatives(state + 0.5 * dt * k1, params)
        k3 = self.predict_derivatives(state + 0.5 * dt * k2, params)
        k4 = self.predict_derivatives(state + dt * k3, params)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)



def main():
    set_seed(42)
    print("=" * 80)
    print("EVALUATING ENHANCED ACCURACY METHODOLOGY")
    print("=" * 80)

    # Generate Boundary-Focused Training Set
    # Includes fine resolution around the Hopf threshold [0.28, 0.38]
    train_I = [0.2, 0.28, 0.32, 0.35, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    train_a = [0.5, 0.7, 0.9]
    train_b = [0.6, 0.8, 1.0]
    eps = 0.08

    train_params = []
    for I in train_I:
        for a in train_a:
            for b in train_b:
                train_params.append([I, a, b, eps])

    from fhn_global_sciml import build_fhn_dataset, train_model
    train_ds = build_fhn_dataset(train_params, traj_duration=100.0, dt=DT, num_ics_per_param=2)

    val_params = [[0.33, 0.7, 0.8, 0.08], [0.8, 0.7, 0.8, 0.08], [0.35, 0.6, 0.7, 0.08]]
    val_ds = build_fhn_dataset(val_params, traj_duration=100.0, dt=DT, num_ics_per_param=2)

    print("\nTraining Enhanced Physical FiLM SciML (RK4 + Boundary Focus)...", flush=True)
    model = EnhancedPhysicalFiLMMLP(hidden_dim=128, dt=DT)
    model = train_model(model, train_ds, val_ds, total_steps=300, batch_size=64, lr=1e-3, seed=42)


    # Benchmark Evaluation
    bench_p = np.array([0.8, 0.7, 0.8, 0.08])
    y_warm = np.array([-1.0, -0.5])
    for _ in range(2000):
        y_warm = rk4_step(y_warm, DT, bench_p)

    gt_traj = simulate_trajectory(y_warm, 3000, DT, bench_p)
    p_gt, a_gt = measure_period_and_amplitude(gt_traj[:, 0])

    # Model rollout
    c = torch.tensor([y_warm], dtype=torch.float32)
    pt = torch.tensor([bench_p], dtype=torch.float32)
    pred_v = []
    with torch.no_grad():
        for _ in range(3000):
            c = model.rk4_step_forward(c, pt)
            pred_v.append(c[0, 0].item())

    p_pred, a_pred = measure_period_and_amplitude(np.array(pred_v))

    print("\n" + "=" * 80)
    print("ACCURACY COMPARISON RESULTS:")
    print("=" * 80)
    print(f"Ground Truth RK4  : Period = {p_gt:.4f} s | Amplitude = {a_gt:.4f}")
    print(f"Original SciML S42: Period = 37.4000 s (+2.42% error) | Amplitude = 3.8516 (+0.19% error)")
    print(f"Enhanced SciML    : Period = {p_pred:.4f} s ({abs(p_pred - p_gt)/p_gt*100:.2f}% error) | Amplitude = {a_pred:.4f} ({abs(a_pred - a_gt)/a_gt*100:.2f}% error)")
    print("=" * 80)

if __name__ == "__main__":
    main()
