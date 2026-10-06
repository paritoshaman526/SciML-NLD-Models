import sys
import json
import numpy as np
import scipy.stats

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from train_and_evaluate_lorenz_style_sciml import SWEEP_CASES
from run_complete_evaluation_suite import analyze_signal

def compute_noise_floor():
    print("=" * 80)
    print("COMPUTING BIOCHEMICAL TRUE-VS-TRUE NUMERICAL & STATISTICAL NOISE FLOOR")
    print("=" * 80)

    results = {}
    lc_cases = [c for c in SWEEP_CASES if not c.get('is_steady_state', False)]

    for c in lc_cases:
        cid = c['id']
        p = c['params']
        p_arr = np.array(p, dtype=np.float64)

        # Baseline trajectory: warm up 6s
        y = DEFAULT_GUESS.copy()
        for _ in range(int(6.0 / DT)):
            y = rk4_step(y, DT, p_arr)

        # Pair 1: Trajectory A from exact warmup attractor
        traj_a = simulate_trajectory(y, int(2.5 / DT), DT, p_arr)[:, 2]

        # Pair 2: Trajectory B from micro-perturbed initial state (1e-6 perturbation on attractor)
        y_pert = y.copy()
        y_pert[2] += 1e-5
        y_pert[0] -= 1e-5
        traj_b = simulate_trajectory(y_pert, int(2.5 / DT), DT, p_arr)[:, 2]

        # Pair 3: Trajectory C from different phase on the orbit
        y_phase = y.copy()
        for _ in range(int(0.05 / DT)):
            y_phase = rk4_step(y_phase, DT, p_arr)
        traj_c = simulate_trajectory(y_phase, int(2.5 / DT), DT, p_arr)[:, 2]

        sig_a = analyze_signal(traj_a, prominence=c['prominence'], min_dist=c['min_dist'])
        sig_b = analyze_signal(traj_b, prominence=c['prominence'], min_dist=c['min_dist'])
        sig_c = analyze_signal(traj_c, prominence=c['prominence'], min_dist=c['min_dist'])

        amp_a = float(sig_a['steady_amp'])
        amp_b = float(sig_b['steady_amp'])
        amp_c = float(sig_c['steady_amp'])

        per_a = float(sig_a['period'])
        per_b = float(sig_b['period'])
        per_c = float(sig_c['period'])

        # Amplitude noise floor (%)
        d_amp_ab = abs(amp_a - amp_b) / amp_a * 100.0
        d_amp_ac = abs(amp_a - amp_c) / amp_a * 100.0

        # Period noise floor (%)
        d_per_ab = abs(per_a - per_b) / per_a * 100.0
        d_per_ac = abs(per_a - per_c) / per_a * 100.0

        # Wasserstein distance W_1 between distributions
        w1_ab = float(scipy.stats.wasserstein_distance(traj_a, traj_b))
        w1_ac = float(scipy.stats.wasserstein_distance(traj_a, traj_c))

        results[cid] = {
            "gt_amp": amp_a,
            "amp_noise_pct_micro": d_amp_ab,
            "amp_noise_pct_phase": d_amp_ac,
            "gt_period": per_a,
            "per_noise_pct_micro": d_per_ab,
            "per_noise_pct_phase": d_per_ac,
            "w1_micro": w1_ab,
            "w1_phase": w1_ac
        }

        print(f"{cid:5s} | Amp: {amp_a:7.1f} mM | dAmp(micro): {d_amp_ab:7.4f}% | dAmp(phase): {d_amp_ac:7.4f}% | W1(phase): {w1_ac:6.3f} mM")

    with open("scratch/biochemical_noise_floor.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved scratch/biochemical_noise_floor.json successfully!")

if __name__ == "__main__":
    compute_noise_floor()
