import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from biochemical_global_sciml import system, rk4_step, DT
from train_and_evaluate_lorenz_style_sciml import (
    UnifiedLorenzStyleSciML, find_physical_steady_state, PHYSICAL_CTOT
)

seeds = [42, 101, 777, 2024]
cases = [
    {"id": "X1", "params": (11.0, 6000.0, 0.15, 1.0), "tau": 2.79},
    {"id": "X3", "params": (13.0, 6000.0, 0.25, 1.0), "tau": 1.16}
]

models = {}
for s in seeds:
    m = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
    m.load_state_dict(torch.load(f"unified_sciml_seed_{s}.pt", map_location="cpu"))
    m.eval()
    models[s] = m

steps = 30000 # 3.0 seconds
t_arr = np.linspace(0, steps * DT, steps + 1)
downsample = 150 # 200 points for clean plotting
indices = np.linspace(0, steps, downsample, dtype=int)

out_data = {}

for c_info in cases:
    cid = c_info["id"]
    p = c_info["params"]
    p_arr = np.array(p, dtype=np.float64)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
    ystar = find_physical_steady_state(p, c_tot=PHYSICAL_CTOT)
    
    delta_m1 = 0.15 * ystar[2]
    y_pert = ystar.copy()
    y_pert[2] += delta_m1
    y_pert[0] -= delta_m1
    y_pert[3] *= 1.20
    
    # 1. RK4
    y_rk4 = y_pert.copy()
    rk4_traj = [y_rk4[2]]
    for _ in range(steps):
        y_rk4 = rk4_step(y_rk4, DT, p_arr)
        rk4_traj.append(y_rk4[2])
    rk4_traj = np.array(rk4_traj)
    
    # 2. Seeds
    seeds_traj = {}
    for s in seeds:
        m = models[s]
        curr = torch.tensor(y_pert, dtype=torch.float32).unsqueeze(0)
        c_tot_t = curr[:, 0] + curr[:, 1] + curr[:, 2]
        traj_s = [curr[0, 2].item()]
        with torch.inference_mode():
            for _ in range(steps):
                curr = m(curr, p_t, target_ctot=c_tot_t)
                traj_s.append(curr[0, 2].item())
        seeds_traj[str(s)] = np.array(traj_s)[indices].tolist()
        
    out_data[cid] = {
        "params": p,
        "tau": c_info["tau"],
        "y_star": float(ystar[2]),
        "init_pert": float(delta_m1),
        "t": t_arr[indices].tolist(),
        "rk4": rk4_traj[indices].tolist(),
        "seeds": seeds_traj
    }

with open("transient_curves_x1_x3.json", "w") as f:
    json.dump(out_data, f, indent=2)
print("Saved transient_curves_x1_x3.json cleanly!")
