import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import numpy as np
import scipy.optimize
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT

def compute_exact_ss(params, target_ctot=10002.0):
    p_arr = np.asarray(params, dtype=np.float64)
    def res(x):
        d, a1, m0 = x
        m1 = target_ctot - (d + a1)
        y = np.array([d, a1, m1, m0], dtype=np.float64)
        ydot = system(y, p_arr)
        return [ydot[0], ydot[1], ydot[3]]
    x0 = [5800.0, min(p_arr[1], 5500.0), 5.0]
    opt = scipy.optimize.least_squares(
        res, x0, bounds=([0, 0, 1e-4], [target_ctot, target_ctot, 50.0]),
        xtol=1e-12, ftol=1e-12
    )
    d, a1, m0 = opt.x
    m1 = target_ctot - (d + a1)
    return np.array([d, a1, m1, m0], dtype=np.float64)

from run_mega_parameter_sweep import SWEEP_CASES
print("Checking exact steady states for X1-X6:")
for c in SWEEP_CASES:
    if c['is_steady_state']:
        ss = compute_exact_ss(c['params'])
        ydot = system(ss, np.array(c['params']))
        print(f"{c['id']}: ss={ss}, ||ydot||={np.linalg.norm(ydot):.2e}")
