import sys
sys.path.insert(0, ".")
import numpy as np
from biochemical_global_sciml import DT, DEFAULT_GUESS, simulate_trajectory, rk4_step
from run_complete_evaluation_suite import analyze_signal
from run_mega_parameter_sweep import SWEEP_CASES

print("--- SWEEP OSCILLATORY CASES ---")
for c in SWEEP_CASES[6:]:
    p_val = c['params']
    p_arr = np.array(p_val, dtype=np.float64)
    y = DEFAULT_GUESS.copy()
    for _ in range(int(c.get('warmup', 6.0) / DT)):
        y = rk4_step(y, DT, p_arr)
    dur = c['duration']
    true_traj = simulate_trajectory(y, int(dur/DT), DT, p_arr)
    res = analyze_signal(true_traj[:, 2], prominence=c['prominence'], min_dist=c['min_dist'], dt=DT)
    T = res['period']
    omega = 2*np.pi / T if T > 0 else 0
    print(f"{c['id']}: p={p_val} | T={T:.4f}s | omega={omega:.2f} rad/s | amp={res['steady_amp']:.1f} mM")
