import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
import numpy as np
from scipy.signal import find_peaks
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT

p_arr = np.array([1.0, 3000.0, 0.80, 1.0], dtype=np.float64)
y = DEFAULT_GUESS.copy()
for _ in range(int(6.0 / DT)):
    y = rk4_step(y, DT, p_arr)
ic = y.copy()

steps = 7000
gt_traj = simulate_trajectory(ic, steps, DT, p_arr)

peaks, props = find_peaks(gt_traj[:, 2], prominence=30.0, distance=int(0.15 / DT))
print("GT Peak indices:", peaks)
print("GT Peak times:", peaks * DT)
if len(peaks) > 1:
    print("GT Periods:", np.diff(peaks * DT))
