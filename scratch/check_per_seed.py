import os
import sys
import numpy as np
import torch

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import ProposedFiLMPhysicsConstrainedMLP, DT
from run_mega_parameter_sweep import SWEEP_CASES, rollout_any, find_steady_state, SEEDS

def check_seeds():
    for cid in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']:
        c_info = next(c for c in SWEEP_CASES if c['id'] == cid)
        ic = find_steady_state(c_info['params'])
        steps = int(c_info['duration'] / DT)
        print(f"\n--- {cid}: params = {c_info['params']} ---")
        for s in SEEDS:
            ckpt = os.path.join(CURRENT_DIR, f"film_bifurcation_aware_s{s}.pt")
            if not os.path.exists(ckpt):
                print(f"Seed {s}: ckpt missing")
                continue
            m = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
            m.load_state_dict(torch.load(ckpt, map_location='cpu'))
            m.eval()
            pred, _, drift = rollout_any(m, ic, c_info['params'], steps, c_info['prominence'], c_info['min_dist'], mode='film')
            m1_ptp = np.ptp(pred[:, 2])
            print(f"  Seed {s:4d} | PTP(m1): {m1_ptp:8.2f} mM")

if __name__ == '__main__':
    check_seeds()
