import os
import sys
import numpy as np
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol")
from train_and_evaluate_lorenz_style_sciml import UnifiedLorenzStyleSciML, PHYSICAL_CTOT
from biochemical_global_sciml import rk4_step, DEFAULT_GUESS, DT

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
p_arr = np.array([1.0, 3000.0, 0.80, 1.0], dtype=np.float64)  # X7
p_t = torch.tensor(p_arr, dtype=torch.float32).unsqueeze(0)

# Warmup to limit cycle
y = DEFAULT_GUESS.copy()
for _ in range(int(6.0 / DT)):
    y = rk4_step(y, DT, p_arr)
ic = y.copy()

print("=" * 80, flush=True)
print("AUDITING X7 DYNAMICS & ADD_M0 NEURAL MODULATION ACROSS 7 CANONICAL SEEDS", flush=True)
print("=" * 80, flush=True)

for s in SEEDS:
    path = f"unified_sciml_seed_{s}.pt"
    if not os.path.exists(path):
        print(f"Seed {s}: checkpoint {path} not found!", flush=True)
        continue
    
    m = UnifiedLorenzStyleSciML(hidden_dim=128, dt=DT)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    
    curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
    c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]
    
    add_m0_vals = []
    m0_vals = []
    m1_vals = []
    exploded = False
    
    with torch.no_grad():
        for step in range(4000):  # 0.4s (several cycles)
            feats = m.extract_features(curr, p_t)
            h = m.in_proj(feats)
            h = m.res1(h)
            h = m.res2(h)
            h = m.res3(h)
            mod = m.out_proj(h)
            add_m0 = (mod[:, 4:5] * 5.0).item()
            
            add_m0_vals.append(add_m0)
            m0_vals.append(curr[0, 3].item())
            m1_vals.append(curr[0, 2].item())
            
            curr = m(curr, p_t, target_ctot=c_tot)
            if torch.isnan(curr).any() or curr[0, 2] > 5000.0:
                exploded = True
                break
    
    if len(m1_vals) >= 1000 and not exploded:
        m1_amp = float(np.ptp(m1_vals[-1000:]))
    else:
        m1_amp = np.nan
        
    mean_add_m0 = float(np.mean(add_m0_vals))
    min_add_m0 = float(np.min(add_m0_vals))
    max_add_m0 = float(np.max(add_m0_vals))
    min_m0 = float(np.min(m0_vals))
    
    if exploded or (not np.isnan(m1_amp) and max(m1_vals) > 2500):
        outcome = "EXPLODED"
    elif not np.isnan(m1_amp) and m1_amp < 50.0:
        outcome = "COLLAPSED"
    else:
        outcome = f"LIMIT CYCLE (Amp={m1_amp:.1f})"
        
    print(f"Seed {s:4d} | Outcome: {outcome:24s} | add_m0 mean: {mean_add_m0:+.4f} [min: {min_add_m0:+.4f}, max: {max_add_m0:+.4f}] | min m0: {min_m0:.4f}", flush=True)

print("=" * 80, flush=True)
