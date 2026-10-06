import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from train_and_evaluate_lorenz_style_sciml import (
    UnifiedLorenzStyleSciML, DT, find_physical_steady_state, SWEEP_CASES, PHYSICAL_CTOT
)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
CASES_SS = [c for c in SWEEP_CASES if c['is_steady_state']]

def run_basin_test(steps=4000):
    print("=" * 80)
    print("RUNNING BASIN-OF-ATTRACTION RELAXATION TEST (X1 - X6) ACROSS 7 SEEDS")
    print("=" * 80)
    
    results = {}
    
    # Preload models
    models = {}
    for s in SEEDS:
        ckpt = f"unified_sciml_seed_{s}.pt"
        m = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        models[s] = m

    for c in CASES_SS:
        cid = c['id']
        p = c['params']
        ystar = find_physical_steady_state(p, c_tot=PHYSICAL_CTOT)
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        
        # Perturbed initial condition: +15% perturbation on m1, balanced on d to keep c_tot exact
        delta = 0.15 * ystar[2]
        y_pert = ystar.copy()
        y_pert[2] += delta
        y_pert[0] -= delta
        y_pert[3] *= 1.20 # +20% on pacemaker m0
        
        results[cid] = {
            "params": p,
            "y_star": ystar.tolist(),
            "y_pert": y_pert.tolist(),
            "perturbation_m1_mM": float(delta),
            "per_seed": {}
        }
        
        print(f"\n--- {cid} (ka={p[0]}, kd2={p[1]:.0f}, kd3={p[2]}) | True m1* = {ystar[2]:.2f} mM | Perturbed m1_0 = {y_pert[2]:.2f} mM (+{delta:.2f} mM) ---")
        
        for s in SEEDS:
            m = models[s]
            curr = torch.tensor(y_pert, dtype=torch.float32).unsqueeze(0)
            c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]
            
            with torch.inference_mode():
                for _ in range(steps):
                    curr = m(curr, p_t, target_ctot=c_tot_t)
            
            final_y = curr.squeeze(0).numpy()
            drift_from_star = float(abs(final_y[2] - ystar[2]))
            relaxation_pct = float((1.0 - drift_from_star / max(delta, 1e-4)) * 100.0)
            
            results[cid]["per_seed"][str(s)] = {
                "final_m1": float(final_y[2]),
                "drift_from_ystar_mM": drift_from_star,
                "pct_recovered": relaxation_pct,
                "converged": bool(drift_from_star < 0.05 * delta)
            }
            print(f"  Seed {s:5d} -> Final m1: {final_y[2]:8.2f} mM | Error: {drift_from_star:8.2f} mM | Recovered: {relaxation_pct:6.1f}%")

    with open("basin_of_attraction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved basin_of_attraction_results.json cleanly!")
    return results

if __name__ == "__main__":
    run_basin_test()
