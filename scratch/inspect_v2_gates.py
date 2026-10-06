import sys, os, torch, numpy as np
CURRENT_DIR = r'c:\Users\satya\Downloads\Van Der Pol'
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
from train_perfect_film_triad import BifurcationGatedFiLMSciML
from run_mega_parameter_sweep import SWEEP_CASES

ckpt = os.path.join(CURRENT_DIR, 'bifurcation_gated_v2.pt')
model = BifurcationGatedFiLMSciML()
model.load_state_dict(torch.load(ckpt, weights_only=True))
model.eval()

print('=== GATE VALUES FOR bifurcation_gated_v2.pt ===')
for c in SWEEP_CASES:
    p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
    p_norm = p_t / model.param_scales
    gate = model.gate_net(p_norm).item()
    is_ss = c.get('is_steady_state', False)
    cid = c['id']
    ka, kd2, kd3, _ = c['params']
    print(f'{cid:4s} | ka={ka:4.1f}, kd2={kd2:4.0f}, kd3={kd3:4.2f} | is_ss={str(is_ss):5s} | pred_gate={gate:.6f}')
