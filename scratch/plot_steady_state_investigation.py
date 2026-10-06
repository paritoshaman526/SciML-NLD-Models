import sys, os
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
import numpy as np
import torch
import matplotlib.pyplot as plt

from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DT, DEFAULT_GUESS
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state
from train_and_benchmark_film_models import autonomous_solve_ss
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML

# Load model
model = BifurcationGatedFiLMSciML(hidden_dim=128, dt=DT, m0_floor=0.5)
ckpt = os.path.join(r"c:\Users\satya\Downloads\Van Der Pol", "film_bifurcation_gated_s2026.pt")
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, c in enumerate([c for c in SWEEP_CASES if c['is_steady_state']]):
    ax = axes[idx]
    p = np.array(c['params'])
    p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
    dur = 1.0
    steps = int(dur / DT)
    t = np.linspace(0, dur, steps)

    # 1. Physical RK4 starting from DEFAULT_GUESS after 10s warmup (True physical stationary state)
    y_warm = DEFAULT_GUESS.copy()
    for _ in range(int(10.0 / DT)):
        y_warm = rk4_step(y_warm, DT, p)
    traj_rk4_warm = simulate_trajectory(y_warm, steps - 1, DT, p)

    # 2. Old find_steady_state (unconstrained c_tot)
    ss_old = find_steady_state(c['params'])
    traj_rk4_old = simulate_trajectory(ss_old, steps - 1, DT, p)

    # 3. Model rollout starting from y_warm
    curr = torch.tensor(y_warm, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0:1] + curr[:, 1:2] + curr[:, 2:3]
    m_pred_warm = [y_warm.copy()]
    with torch.no_grad():
        for _ in range(steps - 1):
            curr = model(curr, p_t, target_ctot=c_tot, steady_state_target=None)
            m_pred_warm.append(curr.squeeze(0).numpy())
    m_pred_warm = np.array(m_pred_warm)

    # 4. Model rollout starting from ss_old
    curr = torch.tensor(ss_old, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0:1] + curr[:, 1:2] + curr[:, 2:3]
    m_pred_old = [ss_old.copy()]
    with torch.no_grad():
        for _ in range(steps - 1):
            curr = model(curr, p_t, target_ctot=c_tot, steady_state_target=None)
            m_pred_old.append(curr.squeeze(0).numpy())
    m_pred_old = np.array(m_pred_old)

    ax.plot(t, traj_rk4_warm[:, 2], 'k-', lw=2.5, label=f'True RK4 (c_tot=10002, m1={y_warm[2]:.1f})')
    ax.plot(t, m_pred_warm[:, 2], 'r--', lw=2.0, label=f'Model from y_warm (m1={m_pred_warm[-1, 2]:.1f})')
    ax.plot(t, traj_rk4_old[:, 2], 'g:', lw=2.0, label=f'Old RK4 find_ss (c_tot={ss_old[:3].sum():.0f}, m1={ss_old[2]:.1f})')
    ax.plot(t, m_pred_old[:, 2], 'm-.', lw=1.5, label=f'Model from old_ss (m1={m_pred_old[-1, 2]:.1f})')

    ax.set_title(f"{c['id']}: ka={p[0]}, kd2={p[1]}, kd3={p[2]}", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Polymer m1 (mM)")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=9, loc='center right')

plt.tight_layout()
out_fig = os.path.join(r"c:\Users\satya\Downloads\Van Der Pol", "steady_state_rk4_investigation.png")
plt.savefig(out_fig, dpi=150)
print(f"Saved diagnostic figure to {out_fig}")
