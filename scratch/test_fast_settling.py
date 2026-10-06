import os
import sys
import time
import numpy as np
import scipy.signal
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import ProposedFiLMVDPMLP, simulate_trajectory, rk4_step, DT

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

models_a = []
for s in SEEDS:
    ckpt = os.path.join(SCRATCH_DIR, f"vdp_sciml_fromscratch_s{s}.pt")
    m = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    models_a.append(m)

print(f"Loaded {len(models_a)} seeds.")

def get_asymptotic_measurements(mu, steps_warmup, steps_measure):
    """
    Simulates GT and all 7 seeds of Model A with proper warmup.
    """
    y0 = np.array([1.5, 0.5], dtype=np.float64)
    
    # 1. Ground Truth
    curr_gt = y0.copy()
    for _ in range(steps_warmup):
        curr_gt = rk4_step(curr_gt, DT, mu)
    
    if mu <= 0.0 and np.linalg.norm(curr_gt) < 1e-3:
        gt_res = (0.0, 0.0, 0.0)
    else:
        traj_gt = simulate_trajectory(curr_gt, steps_measure, DT, mu)[:, 0]
        ptp = np.ptp(traj_gt)
        if ptp < 0.05 or (mu <= 0.0 and ptp < 0.2):
            gt_res = (0.0, 0.0, 0.0)
        else:
            pks, _ = scipy.signal.find_peaks(traj_gt, distance=int(3.0/DT), prominence=0.2)
            troughs, _ = scipy.signal.find_peaks(-traj_gt, distance=int(3.0/DT), prominence=0.2)
            if len(pks) > 1 and len(troughs) > 1:
                # period
                refined_t = []
                for p in pks:
                    if 1 <= p < len(traj_gt) - 1:
                        alpha, beta, gamma = traj_gt[p-1], traj_gt[p], traj_gt[p+1]
                        denom = alpha - 2.0 * beta + gamma
                        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
                        refined_t.append((p + delta) * DT)
                per = float(np.mean(np.diff(refined_t)))
                gt_res = (float(np.mean(traj_gt[pks])), float(np.mean(traj_gt[troughs])), per)
            else:
                gt_res = (0.0, 0.0, 0.0)
                
    # 2. Model A (7 seeds batched)
    pt = torch.tensor([[mu]], dtype=torch.float32) # (1, 1)
    # Run each model
    seed_res = []
    with torch.no_grad():
        for m in models_a:
            curr = torch.tensor([y0], dtype=torch.float32)
            for _ in range(steps_warmup):
                curr = m.rk4_step_forward(curr, pt)
            if mu <= 0.0 and torch.norm(curr).item() < 1e-3:
                seed_res.append((0.0, 0.0, 0.0))
                continue
            pred = [curr[0, 0].item()]
            for _ in range(steps_measure):
                curr = m.rk4_step_forward(curr, pt)
                pred.append(curr[0, 0].item())
            traj = np.array(pred)
            ptp = np.ptp(traj)
            if ptp < 0.05 or (mu <= 0.0 and ptp < 0.2):
                seed_res.append((0.0, 0.0, 0.0))
                continue
            pks, _ = scipy.signal.find_peaks(traj, distance=int(3.0/DT), prominence=0.2)
            troughs, _ = scipy.signal.find_peaks(-traj, distance=int(3.0/DT), prominence=0.2)
            if len(pks) > 1 and len(troughs) > 1:
                refined_t = []
                for p in pks:
                    if 1 <= p < len(traj) - 1:
                        alpha, beta, gamma = traj[p-1], traj[p], traj[p+1]
                        denom = alpha - 2.0 * beta + gamma
                        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
                        refined_t.append((p + delta) * DT)
                per = float(np.mean(np.diff(refined_t)))
                seed_res.append((float(np.mean(traj[pks])), float(np.mean(traj[troughs])), per))
            else:
                seed_res.append((0.0, 0.0, 0.0))
                
    return gt_res, seed_res

# Test sweep across mu in [-0.5, 1.0]
t0 = time.time()
test_mus = [-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
print(f"{'mu':<6} | {'GT x_max':<10} | {'GT Period':<10} | {'Model A Mean x_max':<20} | {'Model A Mean Period':<20}")
print("-" * 75)
for mu in test_mus:
    tau = 2.0 / max(abs(mu), 0.04)
    warmup_s = min(max(5.0 * tau, 50.0), 200.0)
    w_steps = int(warmup_s / DT)
    m_steps = int(60.0 / DT)
    gt_res, s_res = get_asymptotic_measurements(mu, w_steps, m_steps)
    a_max = np.mean([x[0] for x in s_res])
    a_per = np.mean([x[2] for x in s_res])
    print(f"{mu:<+6.2f} | {gt_res[0]:<+10.4f} | {gt_res[2]:<10.4f} | {a_max:<+20.4f} | {a_per:<20.4f}")
print(f"Total time: {time.time() - t0:.2f} s")
