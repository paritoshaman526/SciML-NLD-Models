import torch
import numpy as np
import scipy.signal
import sys
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import ProposedFiLMVDPMLP, simulate_trajectory, DT

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"

m42 = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
m42.load_state_dict(torch.load(SCRATCH_DIR + "/vdp_sciml_fromscratch_s42.pt", map_location="cpu", weights_only=True))
m42.eval()

mu_sweep = np.linspace(-1.0, 2.5, 36)
steps = int(40.0 / DT)

print(f"{'mu':<6} | {'GT ptp':<8} | {'Model A ptp':<12} | {'GT Period':<10} | {'Model A Period':<14}")
print("-" * 65)

for mu in mu_sweep:
    y0 = np.array([1.5, 0.5], dtype=np.float64)
    # GT
    traj_gt = simulate_trajectory(y0, steps, DT, mu)[:, 0]
    asympt_gt = traj_gt[len(traj_gt)//2:]
    ptp_gt = np.ptp(asympt_gt)
    
    # Model A
    pt = torch.tensor([[mu]], dtype=torch.float32)
    curr = torch.tensor([[1.5, 0.5]], dtype=torch.float32)
    pred_x = [1.5]
    with torch.no_grad():
        for _ in range(steps):
            curr = m42.rk4_step_forward(curr, pt)
            pred_x.append(curr[0, 0].item())
    pred_x = np.array(pred_x)
    asympt_a = pred_x[len(pred_x)//2:]
    ptp_a = np.ptp(asympt_a)
    
    # Measure periods if oscillating
    pks_gt, _ = scipy.signal.find_peaks(asympt_gt, distance=int(2.0/DT), prominence=0.2)
    pks_a, _ = scipy.signal.find_peaks(asympt_a, distance=int(2.0/DT), prominence=0.2)
    per_gt = np.mean(np.diff(pks_gt))*DT if len(pks_gt) > 1 else 0.0
    per_a = np.mean(np.diff(pks_a))*DT if len(pks_a) > 1 else 0.0
    
    print(f"{mu:<+6.2f} | {ptp_gt:<8.4f} | {ptp_a:<12.4f} | {per_gt:<10.4f} | {per_a:<14.4f}")
