import json

with open('mega_parameter_sweep_results.json', 'r') as f:
    res = json.load(f)

cases = ['X7', 'X10', 'X12']
models = [
    'FiLM Pure',
    'FiLM Bifurcation-Aware',
    'FiLM+Div',
    'Projected Baseline',
    'Baseline MLP (Unconstrained)'
]

print(f"{'Case':5s} | {'Model':28s} | {'Period Error (%)':22s} | {'Amp Ratio (A_pred/A_gt)':22s} | {'Drift (mM)':15s}")
print("-" * 105)
for cid in cases:
    d = res[cid]
    gt_t = d['gt_period']
    gt_a = d['gt_metric']
    print(f"\n--- {cid}: GT Period = {gt_t:.4f}s, GT Amp = {gt_a:.1f} mM ({d['name']}) ---")
    for m in models:
        if m in d['models']:
            md = d['models'][m]
            p_str = f"{md['p_err_mean']:5.2f} +/- {md['p_err_std']:5.2f}%"
            a_str = f"{md['ratio_mean']:5.3f} +/- {md['ratio_std']:5.3f}"
            d_str = f"{md['drift_mean']:.4f} mM" if md['drift_mean'] < 1.0 else f"{md['drift_mean']:.1f} mM"
            print(f"{cid:5s} | {m:28s} | {p_str:22s} | {a_str:22s} | {d_str:15s}")
