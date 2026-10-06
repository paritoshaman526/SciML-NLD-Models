import numpy as np
import scipy.signal
import torch
import sys
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import ProposedFiLMVDPMLP, simulate_trajectory, rk4_step, DT

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"

m42 = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
m42.load_state_dict(torch.load(SCRATCH_DIR + "/vdp_sciml_fromscratch_s42.pt", map_location="cpu", weights_only=True))
m42.eval()

def measure_true_asymptotic(mu, is_model=False, model=None):
    """
    Properly settles the trajectory onto the true invariant attractor:
    1. Warmup time scales with critical slowing down: tau = 2 / max(|mu|, 0.05)
       Warmup duration = min(max(5 * tau, 60.0), 300.0) seconds.
    2. Measurement window = 60.0 seconds (at least 6-10 full cycles).
    3. If mu <= 0 (subcritical): check if contracting towards origin.
    """
    tau = 2.0 / max(abs(mu), 0.04)
    warmup_s = min(max(5.0 * tau, 60.0), 250.0)
    measure_s = 60.0
    
    warmup_steps = int(warmup_s / DT)
    measure_steps = int(measure_s / DT)
    
    y0 = np.array([1.5, 0.5], dtype=np.float64)
    
    if not is_model:
        # Ground Truth RK4
        curr = y0.copy()
        for _ in range(warmup_steps):
            curr = rk4_step(curr, DT, mu)
        # Check if already collapsed to origin
        if np.linalg.norm(curr) < 1e-4:
            return 0.0, 0.0, 0.0 # x_max, x_min, period
        
        traj = simulate_trajectory(curr, measure_steps, DT, mu)[:, 0]
    else:
        # Neural Model
        pt = torch.tensor([[mu]], dtype=torch.float32)
        curr_t = torch.tensor([y0], dtype=torch.float32)
        with torch.no_grad():
            for _ in range(warmup_steps):
                curr_t = model.rk4_step_forward(curr_t, pt)
            if torch.norm(curr_t).item() < 1e-4:
                return 0.0, 0.0, 0.0
            
            pred = [curr_t[0, 0].item()]
            for _ in range(measure_steps):
                curr_t = model.rk4_step_forward(curr_t, pt)
                pred.append(curr_t[0, 0].item())
            traj = np.array(pred)
    
    ptp = np.ptp(traj)
    if ptp < 0.02 or (mu <= 0.0 and ptp < 0.1):
        return 0.0, 0.0, 0.0
    
    pks, _ = scipy.signal.find_peaks(traj, distance=int(3.0/DT), prominence=0.2)
    troughs, _ = scipy.signal.find_peaks(-traj, distance=int(3.0/DT), prominence=0.2)
    
    if len(pks) > 1 and len(troughs) > 1:
        x_max = float(np.mean(traj[pks]))
        x_min = float(np.mean(traj[troughs]))
        # Parabolic peak interpolation for period
        refined_t = []
        for p in pks:
            if 1 <= p < len(traj) - 1:
                alpha = traj[p - 1]
                beta = traj[p]
                gamma = traj[p + 1]
                denom = alpha - 2.0 * beta + gamma
                delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
                refined_t.append((p + delta) * DT)
        per = float(np.mean(np.diff(refined_t)))
        return x_max, x_min, per
    else:
        return float(np.max(traj)), float(np.min(traj)), 0.0

print("=" * 85)
print("VERIFYING ASYMPTOTIC SETTLING ACROSS MU IN [-0.5, 0.5]")
print("=" * 85)
test_mus = [-0.5, -0.3, -0.2, -0.1, -0.05, 0.0, +0.05, +0.1, +0.2, +0.5, +1.0]
print(f"{'mu':<6} | {'GT x_max':<10} | {'GT x_min':<10} | {'GT Period':<10} | {'Model A x_max':<14} | {'Model A Period':<14}")
print("-" * 85)

for mu in test_mus:
    gt_max, gt_min, gt_per = measure_true_asymptotic(mu, is_model=False)
    a_max, a_min, a_per = measure_true_asymptotic(mu, is_model=True, model=m42)
    print(f"{mu:<+6.2f} | {gt_max:<+10.4f} | {gt_min:<+10.4f} | {gt_per:<10.4f} | {a_max:<+14.4f} | {a_per:<14.4f}")
