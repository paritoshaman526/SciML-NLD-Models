import sys
import numpy as np
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT

p_arr = np.array([1.0, 3000.0, 0.80, 1.0], dtype=np.float64)
y = DEFAULT_GUESS.copy()
for _ in range(int(6.0 / DT)):
    y = rk4_step(y, DT, p_arr)

traj = simulate_trajectory(y, int(3.0 / DT), DT, p_arr)
print(f"X7 GT m0 min: {traj[:, 3].min():.6f}, m0 max: {traj[:, 3].max():.6f}")
print(f"X7 GT m1 min: {traj[:, 2].min():.6f}, m1 max: {traj[:, 2].max():.6f}")
print(f"X7 GT d  min: {traj[:, 0].min():.6f}, d  max: {traj[:, 0].max():.6f}")
print(f"X7 GT a1 min: {traj[:, 1].min():.6f}, a1 max: {traj[:, 1].max():.6f}")
