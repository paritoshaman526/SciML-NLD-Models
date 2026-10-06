import os
import sys
import torch
import numpy as np
import scipy.optimize
import scipy.signal

sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import ProposedFiLMVDPMLP, simulate_trajectory, rk4_step, DT

SCRATCH_DIR = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

# 1. Exact Jacobian Eigenvalue Crossing via Brentq Root Finding
def eval_real_eig(model, mu_val):
    pt = torch.tensor([[mu_val]], dtype=torch.float32)
    s0 = torch.tensor([[0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    f = model.predict_derivatives(s0, pt)
    df1_ds = torch.autograd.grad(f[0, 0], s0, retain_graph=True)[0]
    df2_ds = torch.autograd.grad(f[0, 1], s0)[0]
    J = np.array([[df1_ds[0, 0].item(), df1_ds[0, 1].item()],
                  [df2_ds[0, 0].item(), df2_ds[0, 1].item()]])
    eigs = np.linalg.eigvals(J)
    return float(np.max(np.real(eigs)))

print("=" * 70)
print("1. EXACT MEASURED MU_C CROSSING (AUTOGRAD CONTINUOUS BRENTQ)")
print("=" * 70)
crossings_a = []
for s in SEEDS:
    ckpt = os.path.join(SCRATCH_DIR, f"vdp_sciml_fromscratch_s{s}.pt")
    m = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    
    mu_root = scipy.optimize.brentq(lambda mu: eval_real_eig(m, mu), -0.3, 0.3, xtol=1e-8)
    crossings_a.append(mu_root)
    print(f"Seed {s:4d} Model A: mu_c = {mu_root:+.8f}")

print(f"\nModel A 7-Seed Mean mu_c: {np.mean(crossings_a):+.8f} +/- {np.std(crossings_a):.8f}")

# Model B
m_b = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=True)
m_b.load_state_dict(torch.load(os.path.join(SCRATCH_DIR, "vdp_sciml_residual_s42.pt"), map_location="cpu", weights_only=True))
m_b.eval()
mu_root_b = scipy.optimize.brentq(lambda mu: eval_real_eig(m_b, mu), -0.3, 0.3, xtol=1e-8)
print(f"Model B Seed  42: mu_c = {mu_root_b:+.8f}")

# 2. High-Precision Sub-Grid Period Measurement via Parabolic Peak Interpolation
def get_precise_period(sig, dt=DT):
    pks, _ = scipy.signal.find_peaks(sig, distance=int(3.0 / dt), prominence=0.5)
    refined_t = []
    for p in pks:
        if 1 <= p < len(sig) - 1:
            alpha = sig[p - 1]
            beta = sig[p]
            gamma = sig[p + 1]
            # Parabolic peak vertex offset: delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            denom = alpha - 2.0 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-9 else 0.0
            refined_t.append((p + delta) * dt)
    diffs = np.diff(refined_t)
    return float(np.mean(diffs)), float(np.ptp(sig))

print("\n" + "=" * 70)
print("2. HIGH PRECISION SUB-GRID PERIOD AT MU = 1.0 (PARABOLIC INTERPOLATION)")
print("=" * 70)
y_w = np.array([2.0, 0.0], dtype=np.float64)
for _ in range(int(50.0 / DT)):
    y_w = rk4_step(y_w, DT, 1.0)

gt_traj = simulate_trajectory(y_w, int(100.0 / DT), DT, 1.0)
gt_p, gt_a = get_precise_period(gt_traj[:, 0])
print(f"Ground Truth RK4: Period = {gt_p:.6f} s | Amplitude = {gt_a:.6f}")

p_t = torch.tensor([[1.0]], dtype=torch.float32)

pers_a = []
for s in SEEDS:
    ckpt = os.path.join(SCRATCH_DIR, f"vdp_sciml_fromscratch_s{s}.pt")
    m = ProposedFiLMVDPMLP(hidden_dim=64, dt=DT, embed_prior=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    curr = torch.tensor([y_w], dtype=torch.float32)
    pred_a = [y_w[0]]
    with torch.no_grad():
        for _ in range(int(100.0 / DT)):
            curr = m.rk4_step_forward(curr, p_t)
            pred_a.append(curr[0, 0].item())
    p_a, a_a = get_precise_period(np.array(pred_a))
    pers_a.append(p_a)
    print(f"Model A Seed {s:4d}: Period = {p_a:.6f} s (Error: {(p_a - gt_p)/gt_p*100:+.3f}%) | Amp = {a_a:.6f}")

print(f"Model A Ensemble Mean Period: {np.mean(pers_a):.6f} +/- {np.std(pers_a):.6f} s (Error: {(np.mean(pers_a)-gt_p)/gt_p*100:+.3f}%)")

# Model B
curr = torch.tensor([y_w], dtype=torch.float32)
pred_b = [y_w[0]]
with torch.no_grad():
    for _ in range(int(100.0 / DT)):
        curr = m_b.rk4_step_forward(curr, p_t)
        pred_b.append(curr[0, 0].item())
p_b, a_b = get_precise_period(np.array(pred_b))
print(f"Model B Seed  42: Period = {p_b:.6f} s (Error: {(p_b - gt_p)/gt_p*100:+.3f}%) | Amp = {a_b:.6f}")
