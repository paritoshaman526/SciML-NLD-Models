import torch
import numpy as np
import matplotlib.pyplot as plt
from biochemical_global_sciml import system, rk4_step, simulate_trajectory, DEFAULT_GUESS, DT
from run_mega_parameter_sweep import SWEEP_CASES
from run_complete_evaluation_suite import analyze_signal
from train_perfect_triad_v2 import KineticFiLMEngine, HighPrecisionBifurcationGated, compute_exact_ss, EXACT_SS_DICT

m_film = KineticFiLMEngine()
m_film.load_state_dict(torch.load("film_v2.pt", weights_only=True))
m_film.eval()

m_film_div = KineticFiLMEngine()
m_film_div.load_state_dict(torch.load("film_div_v2.pt", weights_only=True))
m_film_div.eval()

m_gated = HighPrecisionBifurcationGated()
m_gated.load_state_dict(torch.load("bifurcation_gated_v2.pt", weights_only=True))
m_gated.eval()

for c_info in SWEEP_CASES:
    if not c_info.get('is_steady_state', False):
        p_arr = np.array(c_info['params'], dtype=np.float64)
        y = DEFAULT_GUESS.copy()
        for _ in range(int(c_info['warmup'] / DT)):
            y = rk4_step(y, DT, p_arr)
        ic = y.copy()
        steps = min(int(c_info['duration'] / DT), 8000)
        gt_traj = simulate_trajectory(ic, steps, DT, p_arr)

        p_t = torch.tensor(c_info['params'], dtype=torch.float32).unsqueeze(0)
        curr_f = torch.tensor(ic, dtype=torch.float32).unsqueeze(0)
        c_tot = curr_f[:, 0] + curr_f[:, 1] + curr_f[:, 2]

        pred_f = [ic.copy()]
        with torch.no_grad():
            for _ in range(steps):
                curr_f = m_film(curr_f, p_t, target_ctot=c_tot)
                pred_f.append(curr_f.squeeze(0).cpu().numpy())
        pred_f = np.array(pred_f)[:steps]

        sig_gt = analyze_signal(gt_traj[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])
        sig_f = analyze_signal(pred_f[:, 2], prominence=c_info['prominence'], min_dist=c_info['min_dist'])

        gt_peaks = np.ptp(gt_traj[:, 2])
        f_peaks = np.ptp(pred_f[:, 2])

        print(f"{c_info['id']} ({c_info['params']}): GT Amp={sig_gt['steady_amp']:.1f} (ptp={gt_peaks:.1f}), FiLM Amp={sig_f['steady_amp']:.1f} (ptp={f_peaks:.1f}), GT Per={sig_gt['period']:.4f}, FiLM Per={sig_f['period']:.4f}")
