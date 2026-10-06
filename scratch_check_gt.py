import numpy as np
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from run_mega_parameter_sweep import SWEEP_CASES
from run_complete_evaluation_suite import analyze_signal

for c in SWEEP_CASES:
    if not c.get('is_steady_state', False):
        p = np.array(c['params'])
        y = DEFAULT_GUESS.copy()
        for _ in range(int(c['warmup'] / DT)):
            y = rk4_step(y, DT, p)
        steps = int(c['duration'] / DT)
        traj = simulate_trajectory(y, steps, DT, p)
        sig = analyze_signal(traj[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
        min_m1 = np.min(traj[:, 2])
        max_m1 = np.max(traj[:, 2])
        min_m0 = np.min(traj[:, 3])
        max_m0 = np.max(traj[:, 3])
        print(f"{c['id']}: Amp = {sig['steady_amp']:.2f}, Period = {sig['period']:.4f}s, m1=[{min_m1:.2f}, {max_m1:.2f}], m0=[{min_m0:.6f}, {max_m0:.2f}]")
