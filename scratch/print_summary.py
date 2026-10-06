import json

with open('mega_parameter_sweep_results.json', 'r') as f:
    res = json.load(f)

print(f"{'Case':4s} | {'Regime':7s} | {'Params (ka, kd2, kd3)':26s} | {'GT T(s)':7s} | {'GT Amp':7s} | {'FiLM Bifurcation-Aware':32s} | {'FiLM Pure (Osc-Only)':32s}")
print("-" * 125)
for cid, d in res.items():
    is_ss = d['is_steady_state']
    gt_t = d['gt_period']
    gt_a = d['gt_metric']
    p_str = f"({d['params'][0]}, {d['params'][1]:.0f}, {d['params'][2]})"
    m_bif = d['models']['FiLM Bifurcation-Aware']
    m_pure = d['models']['FiLM Pure'] if 'FiLM Pure' in d['models'] else d['models']['FiLM Pure (Osc-Only)']
    if is_ss:
        bif_str = f"Amp: {m_bif['ratio_mean']:6.2f} +/- {m_bif['ratio_std']:5.2f} mM"
        pure_str = f"Amp: {m_pure['ratio_mean']:6.2f} +/- {m_pure['ratio_std']:5.2f} mM"
        reg = "Steady"
    else:
        bif_str = f"P_err: {m_bif['p_err_mean']:5.2f}%, A_ratio: {m_bif['ratio_mean']:.3f}"
        pure_str = f"P_err: {m_pure['p_err_mean']:5.2f}%, A_ratio: {m_pure['ratio_mean']:.3f}"
        reg = "LimitC"
    print(f"{cid:4s} | {reg:7s} | {p_str:26s} | {gt_t:7.4f} | {gt_a:7.1f} | {bif_str:32s} | {pure_str:32s}")
