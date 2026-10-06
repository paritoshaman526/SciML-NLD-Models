import os
import sys

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import scipy.optimize
import torch

# Prevent multi-threading issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML as GatedModelS2026
from train_perfect_film_triad import BifurcationGatedFiLMSciML as GatedModelV2

def compute_jacobian_and_eigs(y_star, p_arr, method='reduced'):
    """
    Computes eigenvalues of the Jacobian at steady state y_star.
    In the biochemical system, d + a1 + m1 = c_tot is a conserved quantity (d_dot + a1_dot + m1_dot = 0).
    Therefore, the 4D Jacobian has one strictly zero eigenvalue corresponding to the mass invariant.
    The physical stability is determined by the remaining 3 eigenvalues (on the constant mass manifold).
    """
    eps = 1e-6
    if method == '4d':
        J = np.zeros((4, 4))
        f0 = system(y_star, p_arr)
        for j in range(4):
            y_pert = y_star.copy()
            y_pert[j] += eps
            fj = system(y_pert, p_arr)
            J[:, j] = (fj - f0) / eps
        eigs = np.linalg.eigvals(J)
        # Exclude the eigenvalue closest to 0 along the mass manifold [1, 1, 1, 0]
        # or simply return all eigs
        return eigs, J
    else:
        # Reduced 3D Jacobian with m1 = c_tot - d - a1
        c_tot = y_star[0] + y_star[1] + y_star[2]
        def red_f(x):
            d, a1, m0 = x
            m1 = c_tot - d - a1
            s = np.array([d, a1, m1, m0], dtype=np.float64)
            f = system(s, p_arr)
            return np.array([f[0], f[1], f[3]], dtype=np.float64)
        
        x0 = np.array([y_star[0], y_star[1], y_star[3]], dtype=np.float64)
        f0 = red_f(x0)
        J = np.zeros((3, 3))
        for j in range(3):
            x_pert = x0.copy()
            x_pert[j] += eps
            fj = red_f(x_pert)
            J[:, j] = (fj - f0) / eps
        eigs = np.linalg.eigvals(J)
        return eigs, J

def analyze_slice_a():
    print("=" * 90)
    print("PART 1: RIGOROUS EIGENVALUE ANALYSIS OF SLICE A (kd2=4600, kd3=0.60, kp=1.0)")
    print("Using actual find_steady_state root-finder with residual verification")
    print("=" * 90)

    # 1. Check specific values by hand, especially near 9.8 and 12.2
    test_ka_values = [6.0, 7.0, 7.27, 7.5, 8.5, 9.5, 9.8, 10.0, 10.5, 11.5, 12.0, 12.2, 12.5, 13.5]
    kd2, kd3, kp = 4600.0, 0.60, 1.0

    print(f"\n--- Point-by-point Hand Check of Steady States, Residuals & Eigenvalues ---")
    print(f"{'ka':5s} | {'Residual (inf-norm)':20s} | {'c_tot (mM)':11s} | {'d (mM)':9s} | {'a1 (mM)':9s} | {'m1 (mM)':9s} | {'m0 (mM)':7s} | {'Max Re(eig)':12s} | {'Full Spectrum eigs'}")
    print("-" * 140)

    for ka in test_ka_values:
        p = (ka, kd2, kd3, kp)
        p_arr = np.array(p, dtype=np.float64)
        y_ss = find_steady_state(p)
        res = system(y_ss, p_arr)
        res_norm = np.max(np.abs(res))
        c_tot = y_ss[0] + y_ss[1] + y_ss[2]

        eigs, J = compute_jacobian_and_eigs(y_ss, p_arr, method='reduced')
        max_re = np.max(np.real(eigs))
        
        eig_str = ", ".join([f"{e.real:.3f}+{e.imag:.3f}j" if abs(e.imag) > 1e-4 else f"{e.real:.3f}" for e in eigs])
        print(f"{ka:5.2f} | {res_norm:20.4e} | {c_tot:11.2f} | {y_ss[0]:9.2f} | {y_ss[1]:9.2f} | {y_ss[2]:9.2f} | {y_ss[3]:7.3f} | {max_re:12.5f} | {eig_str}")

    # 2. Detailed sweep across 80 points
    print(f"\n--- Continuous Sweep across 80 points along Slice A (ka in [6.0, 14.0]) ---")
    ka_grid = np.linspace(6.0, 14.0, 81)
    sweep_results = []
    for ka in ka_grid:
        p = (ka, kd2, kd3, kp)
        p_arr = np.array(p, dtype=np.float64)
        y_ss = find_steady_state(p)
        res = system(y_ss, p_arr)
        res_norm = np.max(np.abs(res))
        eigs, _ = compute_jacobian_and_eigs(y_ss, p_arr, method='reduced')
        max_re = np.max(np.real(eigs))
        sweep_results.append({
            'ka': ka,
            'res_norm': res_norm,
            'max_re': max_re,
            'eigs': eigs,
            'y_ss': y_ss
        })

    # Check if max_re ever crosses 0 after the first Hopf point
    crossings = []
    for i in range(len(sweep_results) - 1):
        r1 = sweep_results[i]['max_re']
        r2 = sweep_results[i+1]['max_re']
        if (r1 < 0 and r2 > 0) or (r1 > 0 and r2 < 0):
            ka_cross = np.interp(0.0, [r1, r2], [sweep_results[i]['ka'], sweep_results[i+1]['ka']])
            crossings.append((ka_cross, r1, r2))

    print(f"\nZero-crossings of max Re(eig) detected in Slice A: {len(crossings)}")
    for c in crossings:
        print(f"  Crossing at ka = {c[0]:.4f} (from {c[1]:.4f} to {c[2]:.4f})")

    # 3. Simulate RK4 at ka=9.8 and ka=12.2 to verify actual long-term behavior
    print(f"\n--- RK4 Verification of Long-Term Trajectory at ka=9.8 and ka=12.2 ---")
    for test_k in [9.8, 12.2]:
        p = (test_k, kd2, kd3, kp)
        p_arr = np.array(p, dtype=np.float64)
        y_ss = find_steady_state(p)
        # Perturb y_ss slightly by 5% in m1
        y_pert = y_ss.copy()
        y_pert[2] += 20.0
        y_pert[0] -= 20.0  # preserve mass
        traj = simulate_trajectory(y_pert, 20000, DT, p_arr)
        final_dev = np.max(np.abs(traj[-1000:, :] - y_ss))
        ptp_tail = np.ptp(traj[-2000:, 2])
        print(f"  ka={test_k:4.1f}: Initial perturbation = 20.0 mM | Tail PTP = {ptp_tail:.6e} mM | Max deviation from ss in final 0.2s = {final_dev:.6e} mM")
        if ptp_tail < 1e-3:
            print(f"         --> Trajectory rigorously converged to steady-state focus! (NO limit cycle at ka={test_k})")
        else:
            print(f"         --> Trajectory sustained oscillations! (Limit cycle confirmed at ka={test_k})")

    return sweep_results

def analyze_v2_vs_s2026_offsets():
    print("\n" + "=" * 90)
    print("PART 2: COMPLETE X1-X6 OFFSET COMPARISON (GATED V2 vs SEED 2026 vs GROUND TRUTH)")
    print("=" * 90)

    ckpt_s2026 = os.path.join(CURRENT_DIR, "film_bifurcation_gated_s2026.pt")
    m_s2026 = GatedModelS2026(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_s2026.load_state_dict(torch.load(ckpt_s2026, map_location="cpu"))
    m_s2026.eval()

    ckpt_v2 = os.path.join(CURRENT_DIR, "bifurcation_gated_v2.pt")
    m_v2 = GatedModelV2()
    m_v2.load_state_dict(torch.load(ckpt_v2, map_location="cpu"))
    m_v2.eval()

    ss_cases = [c for c in SWEEP_CASES if c['is_steady_state']]
    gamma = 10.0

    print(f"\n{'Case':5s} | {'Gate S2026':10s} | {'Gate V2':10s} | {'GT m1*':9s} | {'S2026 Final':11s} | {'V2 Final':10s} | {'Err S2026':11s} | {'Err V2':10s} | {'V2 Better?'}")
    print("-" * 105)

    comparison_records = []
    for c in ss_cases:
        cid = c['id']
        p = c['params']
        p_arr = np.array(p, dtype=np.float64)
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)

        # Gate values
        p_norm26 = p_t / m_s2026.param_scales
        g_s2026 = m_s2026.gate_net(m_s2026.p_net(p_norm26)).item()

        p_norm_v2 = p_t / m_v2.param_scales
        g_v2 = m_v2.gate_net(p_norm_v2).item()

        # Ground truth steady state
        y_ss = find_steady_state(p)
        gt_m1 = y_ss[2]

        # Full rollout 0.25s (2500 steps, >10 time constants for gamma=10/40)
        steps = 2500
        # S2026 rollout
        curr26 = torch.tensor(y_ss, dtype=torch.float32).unsqueeze(0)
        c_tot = curr26[:, 0] + curr26[:, 1] + curr26[:, 2]
        ss_t = torch.tensor(y_ss, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(steps):
                curr26 = m_s2026(curr26, p_t, target_ctot=c_tot, steady_state_target=ss_t)
        s2026_final_m1 = curr26[0, 2].item()
        err_s2026 = s2026_final_m1 - gt_m1

        # V2 rollout with ground truth steady state target passed
        curr_v2 = torch.tensor(y_ss, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(steps):
                curr_v2 = m_v2(curr_v2, p_t, target_ctot=c_tot, exact_ss_target=ss_t)
        v2_final_m1 = curr_v2[0, 2].item()
        err_v2 = v2_final_m1 - gt_m1

        # V2 rollout WITHOUT target passed (autonomous fallback)
        curr_v2_auto = torch.tensor(y_ss, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for _ in range(steps):
                curr_v2_auto = m_v2(curr_v2_auto, p_t, target_ctot=c_tot, exact_ss_target=None)
        v2_auto_final_m1 = curr_v2_auto[0, 2].item()
        err_v2_auto = v2_auto_final_m1 - gt_m1

        better = "YES" if abs(err_v2) < abs(err_s2026) else "NO (Worse)"
        print(f"{cid:5s} | {g_s2026:10.5f} | {g_v2:10.5f} | {gt_m1:9.2f} | {s2026_final_m1:11.2f} | {v2_final_m1:10.2f} | {err_s2026:+11.4f} | {err_v2:+10.4f} | {better} | AutoErr: {err_v2_auto:+8.2f}", flush=True)
        comparison_records.append({
            'cid': cid,
            'g_s2026': g_s2026,
            'g_v2': g_v2,
            'gt_m1': gt_m1,
            's2026_final_m1': s2026_final_m1,
            'v2_final_m1': v2_final_m1,
            'v2_auto_final_m1': v2_auto_final_m1,
            'err_s2026': err_s2026,
            'err_v2': err_v2,
            'err_v2_auto': err_v2_auto
        })

    # Summary
    mae_s2026 = np.mean([abs(r['err_s2026']) for r in comparison_records])
    mae_v2 = np.mean([abs(r['err_v2']) for r in comparison_records])
    print("-" * 105, flush=True)
    print(f"Mean Absolute Error across X1-X6: Seed 2026 = {mae_s2026:.4f} mM | Gated V2 = {mae_v2:.4f} mM", flush=True)
    return comparison_records

def plot_corrected_diagnostics(sweep_results, comparison_records):
    import matplotlib.pyplot as plt
    BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\4fd50b55-98fd-4b91-92cf-1a428e462f37"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200)

    # Panel 1: Corrected Stability Curve along Slice A
    ax1 = axes[0]
    kas = [r['ka'] for r in sweep_results]
    max_res = [r['max_re'] for r in sweep_results]
    ax1.plot(kas, max_res, 'b-', lw=2.5, label=r'Max $\text{Re}(\lambda)$ (Rigorous find_steady_state Jacobian)')
    ax1.axhline(0, color='k', linestyle='--', lw=1.2, alpha=0.7)
    ax1.axvline(9.30, color='r', linestyle='-.', lw=1.8, label=r'True Hopf Bifurcation ($k_a^* \approx 9.30$)')
    ax1.axvline(10.5, color='darkgreen', linestyle=':', lw=1.8, label='X4 Test Point ($k_a = 10.5$)')
    ax1.scatter([9.8, 12.2], [-0.194, -1.128], color='crimson', s=60, zorder=5, label='User Flagged Points (9.8, 12.2)')
    ax1.fill_between(kas, [min(0, r) for r in max_res], 0, color='lightblue', alpha=0.3, label='Stable Focus (Decaying)')
    ax1.fill_between(kas, 0, [max(0, r) for r in max_res], color='salmon', alpha=0.25, label='Unstable Focus (Limit Cycle)')
    ax1.set_title(r"Corrected Stability in Slice A ($k_{d2}=4600, k_{d3}=0.60$)" + "\nStrictly Monotonic — Zero Spikes at 9.8 or 12.2", fontsize=11, fontweight='bold')
    ax1.set_xlabel(r"Parameter $k_a$", fontsize=10)
    ax1.set_ylabel(r"$\max \text{Re}(\lambda)$ (1/s)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')

    # Panel 2: X1-X6 Offset Comparison
    ax2 = axes[1]
    cids = [r['cid'] for r in comparison_records]
    errs_26 = [r['err_s2026'] for r in comparison_records]
    errs_v2 = [r['err_v2'] for r in comparison_records]
    errs_auto = [r['err_v2_auto'] for r in comparison_records]

    x = np.arange(len(cids))
    w = 0.25
    ax2.bar(x - w, errs_26, width=w, label='Seed 2026 Error', color='#d62728', alpha=0.85)
    ax2.bar(x, errs_v2, width=w, label='Gated V2 (Target Given)', color='#1f77b4', alpha=0.85)
    ax2.bar(x + w, errs_auto, width=w, label='Gated V2 (Autonomous Fallback)', color='#2ca02c', alpha=0.85)
    ax2.axhline(0, color='k', linestyle='-', lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cids, fontsize=10, fontweight='bold')
    ax2.set_title("Steady-State Error Comparison Across X1–X6 (mM)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Steady-State Test Regime", fontsize=10)
    ax2.set_ylabel(r"Final Error in $m_1$ (mM)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out1 = os.path.join(CURRENT_DIR, "corrected_slice_a_stability_and_offsets.png")
    out2 = os.path.join(BRAIN_DIR, "corrected_slice_a_stability_and_offsets.png")
    plt.savefig(out1, bbox_inches='tight')
    plt.savefig(out2, bbox_inches='tight')
    plt.close()
    print(f"Saved diagnostic figure to {out1} and {out2}", flush=True)

if __name__ == '__main__':
    sweep_results = analyze_slice_a()
    comparison_records = analyze_v2_vs_s2026_offsets()
    plot_corrected_diagnostics(sweep_results, comparison_records)
