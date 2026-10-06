import os
import sys
import numpy as np
import torch

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import ProposedFiLMPhysicsConstrainedMLP, DT, DEFAULT_GUESS, rk4_step, simulate_trajectory
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state, rollout_any
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML
from run_complete_evaluation_suite import analyze_signal

def main():
    m_film = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_film.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_bifurcation_aware_s42.pt"), map_location="cpu"))
    m_film.eval()

    m_film_div = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_film_div.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_div_bifurcation_aware_s42.pt"), map_location="cpu"))
    m_film_div.eval()

    m_gated = BifurcationGatedFiLMSciML(hidden_dim=128, dt=DT, m0_floor=0.5)
    m_gated.load_state_dict(torch.load(os.path.join(CURRENT_DIR, "film_bifurcation_gated_s2026.pt"), map_location="cpu"))
    m_gated.eval()

    print(f"{'Case':5s} | {'Regime':7s} | {'GT Amp':8s} | {'FiLM Amp':9s} | {'FiLM+Div':9s} | {'Gated Amp':9s} | {'GT Per':8s} | {'Gated Per':9s}")
    print("-" * 85)

    for c in SWEEP_CASES:
        cid = c['id']
        p_arr = np.array(c['params'], dtype=np.float64)
        is_ss = c.get('is_steady_state', False)
        if is_ss:
            ic = find_steady_state(c['params'])
            ss_target = ic.copy()
        else:
            y = DEFAULT_GUESS.copy()
            for _ in range(int(c['warmup'] / DT)):
                y = rk4_step(y, DT, p_arr)
            ic = y.copy()
            ss_target = None

        steps = min(int(c['duration'] / DT), 8000)
        gt = simulate_trajectory(ic, steps, DT, p_arr)

        pred_f, _, _ = rollout_any(m_film, ic, c['params'], steps, c['prominence'], c['min_dist'], mode='film')
        pred_fd, _, _ = rollout_any(m_film_div, ic, c['params'], steps, c['prominence'], c['min_dist'], mode='film')

        # Gated
        p_t = torch.tensor(c['params'], dtype=torch.float32).unsqueeze(0)
        curr = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        c_tot = curr[:, 0] + curr[:, 1] + curr[:, 2]
        ss_t = torch.tensor(ss_target, dtype=torch.float32).unsqueeze(0) if ss_target is not None else None
        pred_g = [ic.copy()]
        with torch.no_grad():
            for _ in range(steps):
                curr = m_gated(curr, p_t, target_ctot=c_tot, steady_state_target=ss_t)
                pred_g.append(curr.squeeze(0).cpu().numpy())
        pred_g = np.array(pred_g)

        gt_ptp = np.ptp(gt[:, 2])
        f_ptp = np.ptp(pred_f[:, 2])
        fd_ptp = np.ptp(pred_fd[:, 2])
        g_ptp = np.ptp(pred_g[:, 2])

        if not is_ss:
            sig_gt = analyze_signal(gt[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
            sig_g = analyze_signal(pred_g[:, 2], prominence=c['prominence'], min_dist=c['min_dist'])
            gt_amp = sig_gt['steady_amp']
            gt_per = sig_gt['period']
            g_amp = sig_g['steady_amp']
            g_per = sig_g['period']
            print(f"{cid:5s} | LimitC  | {gt_amp:8.1f} | {f_ptp:9.1f} | {fd_ptp:9.1f} | {g_amp:9.1f} | {gt_per:8.4f} | {g_per:9.4f}")
        else:
            print(f"{cid:5s} | Steady  | {'0.0':8s} | {f_ptp:9.2f} | {fd_ptp:9.2f} | {g_ptp:9.2f} | {'N/A':8s} | {'N/A':9s}")

if __name__ == '__main__':
    main()
