import sys, os
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
import numpy as np
from biochemical_global_sciml import system, rk4_step, DT, DEFAULT_GUESS
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state
from train_and_benchmark_film_models import autonomous_solve_ss
from scratch_test_ss import compute_exact_ss

print("Comparison of Steady States for X1-X6:")
for c in SWEEP_CASES:
    if c['is_steady_state']:
        p = c['params']
        p_arr = np.array(p)
        y = DEFAULT_GUESS.copy()
        # Simulate true physical RK4 from DEFAULT_GUESS for 15s:
        for _ in range(int(15.0 / DT)):
            y = rk4_step(y, DT, p_arr)
        
        auto_ss = autonomous_solve_ss(p, 10002.0)
        old_ss = find_steady_state(p)

        print(f"\n--- Case {c['id']}: ka={p[0]}, kd2={p[1]}, kd3={p[2]} ---")
        print(f"  1. True RK4 15s rollout : m1={y[2]:.4f} mM, d={y[0]:.4f}, a1={y[1]:.4f}, m0={y[3]:.4f}, c_tot={y[:3].sum():.1f}")
        print(f"  2. Autonomous formula   : m1={auto_ss[2]:.4f} mM, d={auto_ss[0]:.4f}, a1={auto_ss[1]:.4f}, m0={auto_ss[3]:.4f}, c_tot={auto_ss[:3].sum():.1f}")
        print(f"  3. Old find_steady_state: m1={old_ss[2]:.4f} mM, d={old_ss[0]:.4f}, a1={old_ss[1]:.4f}, m0={old_ss[3]:.4f}, c_tot={old_ss[:3].sum():.1f}")
        print(f"  Error: |True RK4 - Autonomous| = {abs(y[2] - auto_ss[2]):.4f} mM")
        print(f"  Error: |True RK4 - Old find_ss| = {abs(y[2] - old_ss[2]):.4f} mM")
