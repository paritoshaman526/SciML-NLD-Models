import os
import sys
import time
import numpy as np
import scipy.signal
import torch

print("Starting batched settling script...", flush=True)

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import (
    ProposedFiLMVDPMLP, UnconstrainedBaselineVDPMLP,
    simulate_trajectory, rk4_step, DT
)

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

models_a = []
for s in SEEDS:
    ckpt = os.path.join(SCRATCH_DIR, f"vdp_sciml_fromscratch_s{s}.pt")
    m = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    models_a.append((s, m))
print(f"Loaded {len(models_a)} Model A seeds.", flush=True)

m_b = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=True)
m_b.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_sciml_residual_s42.pt"), map_location="cpu", weights_only=True))
m_b.eval()

m_base = UnconstrainedBaselineVDPMLP(hidden_dim=64, dt=DT)
m_base.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_baseline_s42.pt"), map_location="cpu", weights_only=True))
m_base.eval()
print("Loaded Model B and Baseline.", flush=True)

# 41 points is plenty for smooth curves across [-1.0, 2.5]
mu_sweep = np.linspace(-1.0, 2.5, 36)
N = len(mu_sweep)
print(f"Sweeping {N} mu points...", flush=True)

# 1. Ground Truth RK4
gt_upper = np.zeros(N)
gt_lower = np.zeros(N)
gt_period = np.zeros(N)

for i, mu in enumerate(mu_sweep):
    if mu <= 0.0:
        gt_upper[i] = 0.0
        gt_lower[i] = 0.0
        gt_period[i] = 0.0
    else:
        tau = 2.0 / max(mu, 0.05)
        warmup_s = min(max(5.0 * tau, 30.0), 80.0)
        curr = np.array([1.5, 0.5], dtype=np.float64)
        for _ in range(int(warmup_s / DT)):
            curr = rk4_step(curr, DT, mu)
        
        traj = simulate_trajectory(curr, int(40.0 / DT), DT, mu)[:, 0]
        pks, _ = scipy.signal.find_peaks(traj, distance=int(3.0/DT), prominence=0.2)
        troughs, _ = scipy.signal.find_peaks(-traj, distance=int(3.0/DT), prominence=0.2)
        if len(pks) > 1 and len(troughs) > 1:
            gt_upper[i] = float(np.mean(traj[pks]))
            gt_lower[i] = float(np.mean(traj[troughs]))
            refined_t = []
            for p in pks:
                if 1 <= p < len(traj) - 1:
                    alpha, beta, gamma = traj[p-1], traj[p], traj[p+1]
                    denom = alpha - 2.0 * beta + gamma
                    delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
                    refined_t.append((p + delta) * DT)
            gt_period[i] = float(np.mean(np.diff(refined_t)))

print("GT RK4 completed.", flush=True)

def simulate_model_batched(model, mu_vals):
    n_mu = len(mu_vals)
    warmup_steps = int(80.0 / DT)
    measure_steps = int(40.0 / DT)
    
    y0 = torch.tensor([[1.5, 0.5]], dtype=torch.float32).repeat(n_mu, 1)
    params = torch.tensor(mu_vals, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        curr = y0
        for _ in range(warmup_steps):
            curr = model.rk4_step_forward(curr, params)
            
        trajs = [curr[:, 0:1]]
        for _ in range(measure_steps):
            curr = model.rk4_step_forward(curr, params)
            trajs.append(curr[:, 0:1])
        trajs = torch.cat(trajs, dim=1).cpu().numpy()
        
    uppers = np.zeros(n_mu)
    lowers = np.zeros(n_mu)
    periods = np.zeros(n_mu)
    
    for i in range(n_mu):
        mu = mu_vals[i]
        tr = trajs[i]
        ptp = np.ptp(tr)
        if ptp < 0.05 or (mu <= 0.0 and ptp < 0.2):
            uppers[i] = 0.0
            lowers[i] = 0.0
            periods[i] = 0.0
            continue
            
        pks, _ = scipy.signal.find_peaks(tr, distance=int(3.0/DT), prominence=0.2)
        troughs, _ = scipy.signal.find_peaks(-tr, distance=int(3.0/DT), prominence=0.2)
        if len(pks) > 1 and len(troughs) > 1:
            uppers[i] = float(np.mean(tr[pks]))
            lowers[i] = float(np.mean(tr[troughs]))
            refined_t = []
            for p in pks:
                if 1 <= p < len(tr) - 1:
                    alpha, beta, gamma = tr[p-1], tr[p], tr[p+1]
                    denom = alpha - 2.0 * beta + gamma
                    delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
                    refined_t.append((p + delta) * DT)
            periods[i] = float(np.mean(np.diff(refined_t)))
        else:
            uppers[i] = 0.0
            lowers[i] = 0.0
            periods[i] = 0.0
            
    return uppers, lowers, periods

a_uppers = []
a_lowers = []
a_periods = []
for s, m in models_a:
    u, l, p = simulate_model_batched(m, mu_sweep)
    a_uppers.append(u)
    a_lowers.append(l)
    a_periods.append(p)
print("Model A (7 seeds) completed.", flush=True)

b_u, b_l, b_p = simulate_model_batched(m_b, mu_sweep)
base_u, base_l, base_p = simulate_model_batched(m_base, mu_sweep)
print("Model B and Baseline completed.", flush=True)

a_u_mean = np.mean(a_uppers, axis=0)
a_l_mean = np.mean(a_lowers, axis=0)
a_p_mean = np.mean(a_periods, axis=0)

print("\n" + "=" * 90, flush=True)
print(f"{'mu':<6} | {'GT x_max':<10} | {'Model A x_max':<14} | {'GT Period':<10} | {'Model A Period':<14} | {'Delta T':<10}", flush=True)
print("-" * 90, flush=True)
for i in range(0, N, 3):
    mu = mu_sweep[i]
    dt_val = a_p_mean[i] - gt_period[i] if (gt_period[i] > 0 and a_p_mean[i] > 0) else 0.0
    print(f"{mu:<+6.2f} | {gt_upper[i]:<+10.4f} | {a_u_mean[i]:<+14.4f} | {gt_period[i]:<10.4f} | {a_p_mean[i]:<14.4f} | {dt_val:<+10.4f}", flush=True)
print("=" * 90, flush=True)

np.savez(
    os.path.join(SCRATCH_DIR, "vdp_settled_bifurcation_data.npz"),
    mu_sweep=mu_sweep,
    gt_upper=gt_upper, gt_lower=gt_lower, gt_period=gt_period,
    a_uppers=np.array(a_uppers), a_lowers=np.array(a_lowers), a_periods=np.array(a_periods),
    a_u_mean=a_u_mean, a_l_mean=a_l_mean, a_p_mean=a_p_mean,
    b_u=b_u, b_l=b_l, b_p=b_p,
    base_u=base_u, base_l=base_l, base_p=base_p
)
print("Saved vdp_settled_bifurcation_data.npz successfully.", flush=True)
