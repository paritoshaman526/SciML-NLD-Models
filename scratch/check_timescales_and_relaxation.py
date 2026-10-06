import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
sys.stdout.reconfigure(line_buffering=True)
from biochemical_global_sciml import system, rk4_step, DT
from train_and_evaluate_lorenz_style_sciml import (
    UnifiedLorenzStyleSciML, find_physical_steady_state, SWEEP_CASES, PHYSICAL_CTOT
)
from validate_biochemical_climate_and_ergodic import true_jacobian

def run_analysis():
    print("=" * 80)
    print("CHECKING PHYSICAL TIMESCALES AND TRANSIENT RELAXATION: RK4 VS NEURAL MODEL")
    print("=" * 80)
    
    seeds = [42, 101, 777, 999, 1234, 2024, 5555]
    models = {}
    for s in seeds:
        m = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
        m.load_state_dict(torch.load(f"unified_sciml_seed_{s}.pt", map_location="cpu"))
        m.eval()
        models[s] = m

    cases_ss = [c for c in SWEEP_CASES if c['is_steady_state']]
    
    # Store trajectory time-series for X1 to X6
    relaxation_data = {}
    
    steps = 5000 # 0.5 seconds (several times tau)
    t_eval = np.linspace(0, steps * DT, steps + 1)
    
    for c in cases_ss:
        cid = c['id']
        p = c['params']
        ystar = find_physical_steady_state(p, c_tot=PHYSICAL_CTOT)
        J = true_jacobian(ystar, p)
        eigs = np.linalg.eigvals(J)
        # Filter out the zero eigenvalue corresponding to the d + a1 + m1 == c_tot manifold
        non_zero_re = [float(np.real(ev)) for ev in eigs if abs(float(np.real(ev))) > 1e-4]
        max_dyn_re = float(np.max(non_zero_re)) if non_zero_re else -1.0
        tau = float(1.0 / abs(max_dyn_re)) if abs(max_dyn_re) > 1e-4 else 1.0
        
        # Perturbation: +15% on m1, balanced on d, +20% on m0
        delta_m1 = 0.15 * ystar[2]
        y_pert = ystar.copy()
        y_pert[2] += delta_m1
        y_pert[0] -= delta_m1
        y_pert[3] *= 1.20
        
        # 1. Run Ground Truth RK4
        y_rk4 = y_pert.copy()
        rk4_m1_traj = [y_rk4[2]]
        p_arr = np.array(p, dtype=np.float64)
        for _ in range(steps):
            y_rk4 = rk4_step(y_rk4, DT, p_arr)
            rk4_m1_traj.append(y_rk4[2])
        rk4_m1_traj = np.array(rk4_m1_traj)
        
        # 2. Run Model for each seed
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        models_m1_traj = {}
        for s in seeds:
            m = models[s]
            curr = torch.tensor(y_pert, dtype=torch.float32).unsqueeze(0)
            c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]
            traj_s = [curr[0, 2].item()]
            with torch.inference_mode():
                for _ in range(steps):
                    curr = m(curr, p_t, target_ctot=c_tot_t)
                    traj_s.append(curr[0, 2].item())
            models_m1_traj[s] = np.array(traj_s)
            
        # Analysis at T = 0.1s, 0.2s, 0.3s, 0.4s, 0.5s
        rk4_final_err = abs(rk4_m1_traj[-1] - ystar[2])
        rk4_init_err = abs(rk4_m1_traj[0] - ystar[2])
        
        print(f"\n--- {cid} (ka={p[0]}, kd2={p[1]:.0f}, kd3={p[2]}) | tau_phys = {tau:.4f} s | init pert: +{delta_m1:.2f} mM ---")
        print(f"  True RK4: init err = {rk4_init_err:.2f} mM -> at 0.1s: {abs(rk4_m1_traj[1000]-ystar[2]):.2f} mM -> at 0.2s: {abs(rk4_m1_traj[2000]-ystar[2]):.2f} mM -> at 0.5s: {rk4_final_err:.4f} mM")
        
        seed_finals = []
        for s in seeds:
            s_traj = models_m1_traj[s]
            err_01 = abs(s_traj[1000] - ystar[2])
            err_02 = abs(s_traj[2000] - ystar[2])
            err_04 = abs(s_traj[4000] - ystar[2])
            err_05 = abs(s_traj[5000] - ystar[2])
            gap_to_rk4 = abs(s_traj[5000] - rk4_m1_traj[-1])
            # Check if trajectory flattened between 0.3s and 0.5s
            d_traj_dt = abs(s_traj[5000] - s_traj[3000])
            seed_finals.append(err_05)
            if s in [42, 101, 777]:
                print(f"  Seed {s:4d}: at 0.1s: {err_01:.2f} | at 0.2s: {err_02:.2f} | at 0.4s: {err_04:.2f} | at 0.5s: {err_05:.2f} mM | d(traj)/dt (0.3->0.5s) = {d_traj_dt:.4f} mM (FLAT = {d_traj_dt < 0.05})")

        print(f"  Seed ensemble at 0.5s: Mean err = {np.mean(seed_finals):.2f} ± {np.std(seed_finals):.2f} mM | RK4 actual residual = {rk4_final_err:.4f} mM")
        
        # Save sample trajectories for plotting
        sample_indices = np.linspace(0, steps, 101, dtype=int)
        relaxation_data[cid] = {
            "params": p,
            "tau_phys_s": tau,
            "y_star_m1": float(ystar[2]),
            "init_pert_mM": float(delta_m1),
            "rk4_final_residual_mM": float(rk4_final_err),
            "t_pts": t_eval[sample_indices].tolist(),
            "rk4_traj": rk4_m1_traj[sample_indices].tolist(),
            "model_mean_traj": np.mean([models_m1_traj[s][sample_indices] for s in seeds], axis=0).tolist(),
            "model_seeds_final_err": {str(s): float(abs(models_m1_traj[s][-1] - ystar[2])) for s in seeds},
            "model_seeds_gap_to_rk4": {str(s): float(abs(models_m1_traj[s][-1] - rk4_m1_traj[-1])) for s in seeds}
        }
        
    with open("transient_relaxation_data.json", "w") as f:
        json.dump(relaxation_data, f, indent=2)
    print("\nSaved transient_relaxation_data.json cleanly!")

if __name__ == "__main__":
    run_analysis()
