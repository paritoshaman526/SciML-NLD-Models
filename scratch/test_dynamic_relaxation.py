import sys, os
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
import numpy as np
import torch
from biochemical_global_sciml import system, rk4_step, DT, DEFAULT_GUESS
from run_mega_parameter_sweep import SWEEP_CASES
from train_and_benchmark_film_models import autonomous_solve_ss
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML

# Load trained model (Seed 2026)
model = BifurcationGatedFiLMSciML(hidden_dim=128, dt=DT, m0_floor=0.5)
ckpt = os.path.join(r"c:\Users\satya\Downloads\Van Der Pol", "film_bifurcation_gated_s2026.pt")
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

print("Testing dynamic relaxation from DEFAULT_GUESS to Steady-State:")
for c in SWEEP_CASES:
    if c['is_steady_state']:
        p = np.array(c['params'])
        p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
        
        # 1. Simulate RK4 from DEFAULT_GUESS for 1.5 seconds
        steps = int(1.5 / DT)
        y_rk4 = DEFAULT_GUESS.copy()
        for _ in range(steps):
            y_rk4 = rk4_step(y_rk4, DT, p)
            
        # 2. Simulate Neural Model from DEFAULT_GUESS for 1.5 seconds
        curr = torch.tensor(DEFAULT_GUESS.copy(), dtype=torch.float32).unsqueeze(0)
        c_tot = curr[:, 0:1] + curr[:, 1:2] + curr[:, 2:3]
        with torch.no_grad():
            for _ in range(steps):
                curr = model(curr, p_t, target_ctot=c_tot, steady_state_target=None)
        y_model = curr.squeeze(0).numpy()
        
        print(f"\nCase {c['id']} ({c['name']}):")
        print(f"  True RK4   end state: m1={y_rk4[2]:.4f} mM, d={y_rk4[0]:.4f}, a1={y_rk4[1]:.4f}, m0={y_rk4[3]:.4f}")
        print(f"  Model Gated end state: m1={y_model[2]:.4f} mM, d={y_model[0]:.4f}, a1={y_model[1]:.4f}, m0={y_model[3]:.4f}")
        diff = np.abs(y_rk4 - y_model)
        print(f"  Gap: Delta_m1 = {diff[2]:.4f} mM, Delta_d = {diff[0]:.4f} mM, Delta_a1 = {diff[1]:.4f} mM")
