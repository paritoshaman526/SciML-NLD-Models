import sys, os, numpy as np, scipy.optimize
CURRENT_DIR = r'c:\Users\satya\Downloads\Van Der Pol'
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state

def autonomous_solve_ss(params, c_tot):
    ka, kd2, kd3, kp = params
    a1_star = kd2 / kp
    
    def m0_eq(m0):
        m0 = max(m0, 1e-8)
        d = (2.0 * kd2 / ka) * m0
        m1 = max(c_tot - d - a1_star, 1e-6)
        ratio = m1 / m0
        denom = max((ratio - 2.0)**2, 1e-4)
        dm0 = kd3 * (m1 - 4.0 * m0) - (2.0 * 4.6 * kd2 * m0) / denom - 2.0 * kd3 * m0 - kp * (m0**2)
        return dm0
    
    sol = scipy.optimize.root_scalar(m0_eq, bracket=[1e-4, 50.0], method='brentq')
    m0_star = sol.root
    d_star = (2.0 * kd2 / ka) * m0_star
    m1_star = c_tot - d_star - a1_star
    return np.array([d_star, a1_star, m1_star, m0_star])

print('Verifying Autonomous Fixed-Point Solver vs Ground Truth:')
for c in SWEEP_CASES:
    if c['is_steady_state']:
        ss_gt = find_steady_state(c['params'])
        c_tot = sum(ss_gt[:3])
        ss_pred = autonomous_solve_ss(c['params'], c_tot)
        err = np.max(np.abs(ss_pred - ss_gt))
        cid = c['id']
        print(f"{cid}: GT m1*={ss_gt[2]:.4f}, Pred m1*={ss_pred[2]:.4f} | Max Error = {err:.2e} mM")
