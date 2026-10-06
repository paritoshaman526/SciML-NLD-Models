import os
import sys
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT, set_seed
from run_mega_parameter_sweep import SWEEP_CASES
from run_complete_evaluation_suite import analyze_signal

set_seed(42)

def find_ss(params, target_ctot=10002.0):
    p_arr = np.asarray(params, dtype=np.float64)
    def res(x):
        d, a1, m0 = x
        m1 = target_ctot - (d + a1)
        y = np.array([d, a1, m1, m0], dtype=np.float64)
        ydot = system(y, p_arr)
        return [ydot[0], ydot[1], ydot[3]]
    x0 = [5800.0, min(p_arr[1], 5500.0), 5.0]
    opt = scipy.optimize.least_squares(res, x0, bounds=([0, 0, 1e-4], [target_ctot, target_ctot, 50.0]))
    d, a1, m0 = opt.x
    m1 = target_ctot - (d + a1)
    return np.array([d, a1, m1, m0], dtype=np.float64)

# Test Ground Truth RK4 on X7 and X8
for c in SWEEP_CASES:
    if c['id'] in ['X1', 'X4', 'X7', 'X8', 'X11', 'X12']:
        p_arr = np.array(c['params'], dtype=np.float64)
        if c['is_steady_state']:
            y_ss = find_ss(c['params'])
            ydot = system(y_ss, p_arr)
            print(f"[{c['id']:3s}] SS: d={y_ss[0]:.1f}, a1={y_ss[1]:.1f}, m1={y_ss[2]:.1f}, m0={y_ss[3]:.3f} | residual norm: {np.linalg.norm(ydot):.2e}")
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c['warmup'] / DT)):
                y = rk4_step(y, DT, p_arr)
            steps = 8000
            traj = simulate_trajectory(y, steps, DT, p_arr)
            sig = analyze_signal(traj[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
            print(f"[{c['id']:3s}] Limit Cycle: True Amp={sig['steady_amp']:.1f} mM | True Period={sig['period']:.4f}s")
