import numpy as np
import scipy.signal
import sys
sys.path.insert(0, r"c:\Users\satya\Downloads\Van Der Pol\vanderpol")
from vdp_global_sciml import simulate_trajectory, DT

def get_asymptotic_extrema(traj_x):
    n = len(traj_x)
    asympt = traj_x[n // 2:]
    amp = np.ptp(asympt)
    if amp < 0.05:
        return 0.0, 0.0
    pks, _ = scipy.signal.find_peaks(asympt, distance=int(2.0/DT), prominence=0.1)
    troughs, _ = scipy.signal.find_peaks(-asympt, distance=int(2.0/DT), prominence=0.1)
    if len(pks) > 0 and len(troughs) > 0:
        return float(np.mean(asympt[pks])), float(np.mean(asympt[troughs]))
    else:
        return float(np.max(asympt)), float(np.min(asympt))

mu_sweep = np.linspace(-1.0, 2.5, 36)
steps = int(40.0 / DT)
for mu in mu_sweep[:15]:
    y0 = np.array([1.5, 0.5], dtype=np.float64)
    traj_gt = simulate_trajectory(y0, steps, DT, mu)[:, 0]
    gt_max, gt_min = get_asymptotic_extrema(traj_gt)
    print(f"mu = {mu:+.2f} | gt_max = {gt_max:+.4f}, gt_min = {gt_min:+.4f}, ptp = {gt_max-gt_min:.4f}")
