import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\4fd50b55-98fd-4b91-92cf-1a428e462f37"

from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from run_mega_parameter_sweep import SWEEP_CASES
from train_perfect_film_triad import BifurcationGatedFiLMSciML

# Load the trained Bifurcation-Gated model
model = BifurcationGatedFiLMSciML()
ckpt_path = os.path.join(CURRENT_DIR, "bifurcation_gated_v2.pt")
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()

# Cases of interest: X9 and X11
case_x9 = next(c for c in SWEEP_CASES if c['id'] == 'X9')
case_x11 = next(c for c in SWEEP_CASES if c['id'] == 'X11')

steps = 8000
t_arr = np.linspace(0, steps * DT, steps)

results = {}
for case in [case_x9, case_x11]:
    cid = case['id']
    p_arr = np.array(case['params'], dtype=np.float64)
    p_t = torch.tensor(case['params'], dtype=torch.float32).unsqueeze(0)

    # Initial condition after warmup
    y = DEFAULT_GUESS.copy()
    for _ in range(int(case['warmup'] / DT)):
        y = rk4_step(y, DT, p_arr)
    ic = y.copy()

    # Ground truth trajectory
    gt_traj = simulate_trajectory(ic, steps, DT, p_arr)

    # Model rollout tracking pred_gate at every step
    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]

    pred_states = [ic.copy()]
    gate_values = []

    with torch.no_grad():
        for step_idx in range(steps):
            # Compute gate at this time step
            p_norm = p_t / model.param_scales
            gate_val = model.gate_net(p_norm).item()
            gate_values.append(gate_val)

            # Step forward
            curr = model(curr, p_t, target_ctot=c_tot)
            pred_states.append(curr.squeeze(0).cpu().numpy())

    pred_states = np.array(pred_states)[:steps]
    gate_values = np.array(gate_values)

    results[cid] = {
        'case': case,
        'gt_traj': gt_traj,
        'pred_states': pred_states,
        'gate_values': gate_values
    }

# Create 2x2 diagnostic figure
fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=220)
plt.subplots_adjust(hspace=0.35, wspace=0.22)

# Column 0: X9
ax_traj_x9 = axes[0, 0]
ax_gate_x9 = axes[1, 0]

# Column 1: X11
ax_traj_x11 = axes[0, 1]
ax_gate_x11 = axes[1, 1]

# Plot X9 Trajectory
res_x9 = results['X9']
ax_traj_x9.plot(t_arr, res_x9['gt_traj'][:steps, 2], 'k-', lw=2.2, label="Ground Truth RK4")
ax_traj_x9.plot(t_arr, res_x9['pred_states'][:, 2], color='#d62728', lw=1.8, label="Bifurcation-Gated SciML")
ax_traj_x9.set_title(f"X9: $k_a=3.5, k_{{d2}}=2600, k_{{d3}}=1.1$\nPolymer $m_1$ Trajectory", fontsize=11, fontweight="bold")
ax_traj_x9.set_xlabel("Time (s)", fontsize=10)
ax_traj_x9.set_ylabel(r"$m_1$ (mM)", fontsize=10)
ax_traj_x9.grid(True, linestyle=":", alpha=0.5)
ax_traj_x9.legend(fontsize=9.5)

# Plot X9 Gate over time
ax_gate_x9.plot(t_arr, res_x9['gate_values'], color='#1f77b4', lw=2.5, label=f"pred_gate(t) = {res_x9['gate_values'][0]:.6f}")
ax_gate_x9.set_ylim(-0.05, 1.05)
ax_gate_x9.set_title("X9: Learned Bifurcation Gate $g(\\mathbf{p})$ over Time", fontsize=11, fontweight="bold")
ax_gate_x9.set_xlabel("Time (s)", fontsize=10)
ax_gate_x9.set_ylabel(r"pred_gate $g(\mathbf{p}) \in [0, 1]$", fontsize=10)
ax_gate_x9.grid(True, linestyle=":", alpha=0.5)
ax_gate_x9.legend(fontsize=9.5, loc="lower right")

# Plot X11 Trajectory
res_x11 = results['X11']
ax_traj_x11.plot(t_arr, res_x11['gt_traj'][:steps, 2], 'k-', lw=2.2, label="Ground Truth RK4")
ax_traj_x11.plot(t_arr, res_x11['pred_states'][:, 2], color='#d62728', lw=1.8, label="Bifurcation-Gated SciML")
ax_traj_x11.set_title(f"X11: $k_a=5.0, k_{{d2}}=5200, k_{{d3}}=0.35$\nPolymer $m_1$ Trajectory", fontsize=11, fontweight="bold")
ax_traj_x11.set_xlabel("Time (s)", fontsize=10)
ax_traj_x11.set_ylabel(r"$m_1$ (mM)", fontsize=10)
ax_traj_x11.grid(True, linestyle=":", alpha=0.5)
ax_traj_x11.legend(fontsize=9.5)

# Plot X11 Gate over time
ax_gate_x11.plot(t_arr, res_x11['gate_values'], color='#1f77b4', lw=2.5, label=f"pred_gate(t) = {res_x11['gate_values'][0]:.6f}")
ax_gate_x11.set_ylim(-0.05, 1.05)
ax_gate_x11.set_title("X11: Learned Bifurcation Gate $g(\\mathbf{p})$ over Time", fontsize=11, fontweight="bold")
ax_gate_x11.set_xlabel("Time (s)", fontsize=10)
ax_gate_x11.set_ylabel(r"pred_gate $g(\mathbf{p}) \in [0, 1]$", fontsize=10)
ax_gate_x11.grid(True, linestyle=":", alpha=0.5)
ax_gate_x11.legend(fontsize=9.5, loc="lower right")

fig.suptitle("Diagnostic Investigation: pred_gate over Time for X9 and X11", fontsize=13, fontweight="bold", y=0.995)

out_png1 = os.path.join(CURRENT_DIR, "gate_over_time_x9_x11_diagnostic.png")
out_png2 = os.path.join(BRAIN_DIR, "gate_over_time_x9_x11_diagnostic.png")
plt.savefig(out_png1, bbox_inches="tight", dpi=220)
plt.savefig(out_png2, bbox_inches="tight", dpi=220)
plt.close()

print(f"Diagnostic plot saved to:")
print(f"  - {out_png1}")
print(f"  - {out_png2}")
print(f"X9:  gate[0]={res_x9['gate_values'][0]:.6f}, min={np.min(res_x9['gate_values']):.6f}, max={np.max(res_x9['gate_values']):.6f}")
print(f"X11: gate[0]={res_x11['gate_values'][0]:.6f}, min={np.min(res_x11['gate_values']):.6f}, max={np.max(res_x11['gate_values']):.6f}")
