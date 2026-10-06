import os
import sys
sys.path.insert(0, ".")
import numpy as np
from biochemical_global_sciml import simulate_trajectory, DEFAULT_GUESS, DT, rk4_step
from train_all_bifurcation_aware_models import OSC_PARAMS, SS_PARAMS, find_steady_state
from run_complete_evaluation_suite import analyze_signal

print("--- TRAINING SET TRUTH ---")
for p in OSC_PARAMS:
    y = DEFAULT_GUESS.copy()
    for _ in range(int(12.0/DT)):
        y = rk4_step(y, DT, np.array(p))
    traj = simulate_trajectory(y, int(2.0/DT), DT, np.array(p))
    m = analyze_signal(traj[:, 2], prominence=10.0, min_dist=0.04)
    omega = 2*np.pi / m['period'] if m['period'] > 0 else 0
    print(f"OSC {p}: Period={m['period']:.4f}s, omega={omega:.2f} rad/s, Amp={m['steady_amp']:.1f} mM")

for p in SS_PARAMS:
    ss = find_steady_state(p)
    print(f"SS {p}: m1*={ss[2]:.1f}, d*={ss[0]:.1f}, a1*={ss[1]:.1f}, m0*={ss[3]:.2f}")
