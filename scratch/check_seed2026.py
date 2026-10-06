import os
import sys
CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
import torch
import numpy as np
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state, rollout_any
from biochemical_global_sciml import DT, DEFAULT_GUESS, rk4_step, simulate_trajectory

for s in [2026, 4242, 7777]:
    ckpt = os.path.join(CURRENT_DIR, f"film_bifurcation_gated_s{s}.pt")
    if not os.path.exists(ckpt):
        continue
    model = BifurcationGatedFiLMSciML(hidden_dim=128, dt=0.0001, m0_floor=0.5)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    print(f"\n========================================================")
    print(f"=== GATE VALUES FOR SEED {s} ({os.path.basename(ckpt)}) ===")
    print(f"========================================================")
    for c in SWEEP_CASES:
        cid = c['id']
        p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
        p_norm = p_t / model.param_scales
        p_embed = model.p_net(p_norm)
        gate = model.gate_net(p_embed).item()
        is_ss = c.get('is_steady_state', False)
        print(f"{cid:4s} | ka={c['params'][0]:4.1f}, kd2={c['params'][1]:4.0f}, kd3={c['params'][2]:4.2f} | is_ss={str(is_ss):5s} | pred_gate={gate:.6f}")

