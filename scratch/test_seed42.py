import os
import sys
import numpy as np
import torch

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import ProposedFiLMPhysicsConstrainedMLP, DT, DEFAULT_GUESS, rk4_step
from run_mega_parameter_sweep import SWEEP_CASES, rollout_any, find_steady_state

def test():
    m = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_bifurcation_aware_s42.pt"), map_location="cpu"))
    m.eval()

    print("Evaluating Seed 42 FiLM Bifurcation-Aware across 12 cases:")
    for c in SWEEP_CASES:
        p_arr = np.array(c['params'], dtype=np.float64)
        if c['is_steady_state']:
            ic = find_steady_state(c['params'])
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c['warmup']/DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y
        steps = int(c['duration']/DT)
        pred, sig, drift = rollout_any(m, ic, c['params'], steps, c['prominence'], c['min_dist'], mode='film')
        m1_ptp = np.ptp(pred[:, 2])
        print(f"  {c['id']}: ptp(m1)={m1_ptp:8.2f} mM | amp={sig['steady_amp']:8.2f} mM | period={sig['period']:.4f}s")

if __name__ == "__main__":
    test()
