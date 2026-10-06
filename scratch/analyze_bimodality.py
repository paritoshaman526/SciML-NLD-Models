import json
import numpy as np
import scipy.stats

def analyze_bimodality(values):
    vals = np.array(values, dtype=np.float64)
    n = len(vals)
    mean = np.mean(vals)
    std = np.std(vals, ddof=1) if n > 1 else 0.0
    median = np.median(vals)
    cv = std / max(abs(mean), 1e-6)
    
    # Skewness and Kurtosis
    if std > 1e-8:
        skew = scipy.stats.skew(vals)
        kurt = scipy.stats.kurtosis(vals, fisher=False) # Pearson kurtosis (normal=3)
        # Sarle's Bimodality Coefficient: BC = (skew^2 + 1) / kurt
        # Values > 0.555 (5/9) indicate bimodal or multimodal distribution
        bc = (skew**2 + 1.0) / max(kurt, 1e-4)
    else:
        skew, kurt, bc = 0.0, 3.0, 0.333
        
    is_bimodal = (bc > 0.555 and cv > 0.4) or (cv > 0.8)
    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "cv": float(cv),
        "skew": float(skew),
        "kurt": float(kurt),
        "bc": float(bc),
        "is_bimodal": bool(is_bimodal)
    }

def main():
    with open("biochemical_7seed_results.json") as f:
        d = json.load(f)

    seeds = d['seeds']
    models = ['baseline', 'div', 'film', 'film_div']
    model_labels = {
        'baseline': 'Baseline',
        'div': 'Baseline Div-Only',
        'film': 'FiLM Pure',
        'film_div': 'FiLM+Div'
    }
    cases = ['Case A', 'Case B', 'Case C', 'Case D']

    report = {}
    for c in cases:
        report[c] = {}
        for m in models:
            per_seed_amp_err = [d['raw_runs'][str(s)][c]['models'][m]['amp_err_pct'] for s in seeds]
            per_seed_per_err = [d['raw_runs'][str(s)][c]['models'][m]['period_err_pct'] for s in seeds]
            per_seed_ret = [d['raw_runs'][str(s)][c]['models'][m]['self_decay_pct'] for s in seeds]
            per_seed_drift = [d['raw_runs'][str(s)][c]['models'][m]['mass_drift'] for s in seeds]
            
            b_amp = analyze_bimodality(per_seed_amp_err)
            b_ret = analyze_bimodality(per_seed_ret)
            b_drift = analyze_bimodality(per_seed_drift)
            
            report[c][m] = {
                "per_seed_amp_err": per_seed_amp_err,
                "per_seed_ret": per_seed_ret,
                "per_seed_drift": per_seed_drift,
                "stats_amp_err": b_amp,
                "stats_ret": b_ret,
                "stats_drift": b_drift
            }

    with open("biochemical_7seed_bimodality_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved biochemical_7seed_bimodality_analysis.json!")

if __name__ == '__main__':
    main()
