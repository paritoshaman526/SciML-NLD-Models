import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    UnconstrainedBaselineMLP,
    build_dataset_with_ic_splits,
    DT,
    DEFAULT_GUESS,
    rk4_step,
    simulate_trajectory
)

def evaluate_baseline(model, p=(6.0, 4208.0, 1.0, 1.0), n_rollout=20000):
    model.eval()
    p_arr = np.array(p, dtype=np.float64)
    y = DEFAULT_GUESS.copy()
    for _ in range(int(20.0 / DT)):
        y = rk4_step(y, DT, p_arr)
    
    curr = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
    traj = [y.copy()]
    with torch.no_grad():
        for _ in range(n_rollout):
            curr = model(curr, p_t)
            traj.append(curr.squeeze(0).cpu().numpy())
    traj = np.array(traj)
    
    quarters = np.array_split(traj[:, 2], 4)
    amps = [np.ptp(q) for q in quarters]
    ret = amps[-1] / max(amps[0], 1e-6) * 100.0
    c_tot = traj[:, 0] + traj[:, 1] + traj[:, 2]
    drift = np.ptp(c_tot)
    return {
        "ret": ret,
        "q1_amp": amps[0],
        "q4_amp": amps[-1],
        "drift": drift,
        "min_m1": traj[:, 2].min(),
        "max_m1": traj[:, 2].max()
    }

print("1. Evaluating current baseline_unconstrained_mlp.pt...")
ckpt_curr = torch.load(os.path.join(CURRENT_DIR, "baseline_unconstrained_mlp.pt"), map_location="cpu")
m_curr = UnconstrainedBaselineMLP()
m_curr.load_state_dict(ckpt_curr)
res_curr = evaluate_baseline(m_curr)
print(f"Current Baseline: Retention = {res_curr['ret']:.1f}%, Q1={res_curr['q1_amp']:.1f}, Q4={res_curr['q4_amp']:.1f}, Drift={res_curr['drift']:.1f}")

# 2. Let's see if fine-tuning with longer horizon K (K=25 -> 50) improves retention
print("\n2. Fine-tuning current baseline with extended curriculum (K=25 -> 50)...")
TRAINING_PARAMS = [
    (2.0, 3000.0, 0.6, 1.0),
    (2.0, 4500.0, 1.0, 1.0),
    (4.0, 3000.0, 1.0, 1.0),
    (4.0, 4500.0, 0.6, 1.0),
    (6.0, 3000.0, 0.6, 1.0),
    (6.0, 4208.0, 1.0, 1.0),
    (6.0, 4500.0, 1.0, 1.0),
    (8.0, 4500.0, 0.6, 1.0),
]

train_ds, val_ds = build_dataset_with_ic_splits(TRAINING_PARAMS, traj_duration=1.2, dt=DT, num_train_ics=4, num_val_ics=2)

m_tuned = UnconstrainedBaselineMLP()
m_tuned.load_state_dict(ckpt_curr)
m_tuned.train()

opt = torch.optim.AdamW(m_tuned.parameters(), lr=5e-4, weight_decay=1e-5)
total_steps = 1500

def sample_batch(ds, K, batch_size=64):
    states, params, future = [], [], []
    indices = np.random.randint(0, len(ds), size=batch_size)
    for idx_sample in indices:
        item = ds[idx_sample]
        tr = item['traj']
        p = item['params']
        idx = np.random.randint(0, len(tr) - K - 1)
        states.append(tr[idx])
        params.append(p)
        future.append(tr[idx + 1 : idx + K + 1])
    return torch.stack(states), torch.stack(params), torch.stack(future)

best_res = None
best_ret = -1.0
best_state = None

for step in range(1, total_steps + 1):
    K = int(np.interp(step, [1, total_steps], [25, 50]))
    s_b, p_b, f_b = sample_batch(train_ds, K)
    weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / m_tuned.state_scales
    
    curr = s_b
    loss = 0.0
    for k in range(K):
        prev = curr
        curr = m_tuned(curr, p_b)
        state_l = torch.mean(((curr - f_b[:, k, :]) * weights) ** 2)
        pred_v = (curr - prev) / m_tuned.dt
        true_v = (f_b[:, k, :] - (s_b if k == 0 else f_b[:, k - 1, :])) / m_tuned.dt
        deriv_l = torch.mean(((pred_v - true_v) / (m_tuned.state_scales * 100.0)) ** 2)
        loss = loss + state_l + 0.05 * deriv_l
    loss = loss / K
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m_tuned.parameters(), 2.0)
    opt.step()
    
    if step % 300 == 0 or step == total_steps:
        eval_res = evaluate_baseline(m_tuned)
        print(f"Step {step:4d} | K={K} | Loss={loss.item():.4e} | Retention={eval_res['ret']:.1f}% (Q1={eval_res['q1_amp']:.1f}, Q4={eval_res['q4_amp']:.1f}) | Drift={eval_res['drift']:.1f}")
        # We want sustained oscillation without blowup (retention near 100, amp not exploding > 350)
        if 50.0 < eval_res['ret'] < 150.0 and eval_res['q4_amp'] < 350.0:
            if abs(eval_res['ret'] - 100.0) < abs(best_ret - 100.0):
                best_ret = eval_res['ret']
                best_res = eval_res
                best_state = {k: v.cpu().clone() for k, v in m_tuned.state_dict().items()}

if best_state is not None:
    torch.save(best_state, os.path.join(CURRENT_DIR, "improved_baseline_unconstrained_mlp.pt"))
    print(f"\nSaved improved baseline: Retention = {best_res['ret']:.1f}%")
else:
    print("\nNo improved stable baseline found that met criteria.")
