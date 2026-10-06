import json, numpy as np

with open("scratch/component_ablation_7seed_results.json") as f:
    abl = json.load(f)
with open("scratch/baseline_unconstrained_7seed_results.json") as f:
    base = json.load(f)

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]
LC_CASES = ["X7","X8","X9","X10","X11","X12"]
SS_CASES = ["X1","X2","X3","X4","X5","X6"]

models = {
    "Unified":    abl["Unified_Full"]["per_seed"],
    "NoFiLM":     abl["NoFiLM"]["per_seed"],
    "NoSoftplus": abl["NoSoftplus"]["per_seed"],
    "NoMassPrj":  abl["NoMassProjection"]["per_seed"],
    "Baseline":   {str(s): base[str(s)] for s in SEEDS},
}

print("=== LIMIT CYCLE: Amplitude Error (%) ===")
cols = list(models.keys())
hdr = "Case  | " + " | ".join(f"{c:22s}" for c in cols)
print(hdr); print("-"*len(hdr))
for c in LC_CASES:
    row = f"{c:5s} |"
    for mname, mdata in models.items():
        errs = []
        for s in SEEDS:
            d = mdata[str(s)][c]
            errs.append(d.get("amp_err_pct", 100.0))
        m, sd = np.mean(errs), np.std(errs)
        fails = sum(1 for e in errs if e > 50.0)
        row += f" {m:5.2f}+/-{sd:4.2f}% {fails}/7    |"
    print(row)

print()
print("=== STABLE FOCUS: Terminal Attractor Drift (mM) ===")
hdr2 = "Case  | " + " | ".join(f"{c:22s}" for c in cols)
print(hdr2); print("-"*len(hdr2))
for c in SS_CASES:
    row = f"{c:5s} |"
    for mname, mdata in models.items():
        drifts = []
        for s in SEEDS:
            d = mdata[str(s)][c]
            key = "drift_mM" if "drift_mM" in d else "drift_from_ystar_mM"
            drifts.append(d[key])
        m, sd = np.mean(drifts), np.std(drifts)
        row += f" {m:6.2f}+/-{sd:5.2f} mM       |"
    print(row)

# Also print a compact summary used for the master table
print()
print("=== COMPACT SUMMARY (for audit table) ===")
print("LC cases - Unified mean/std/fails:")
for c in LC_CASES:
    errs = [abl["Unified_Full"]["per_seed"][str(s)][c].get("amp_err_pct",100.0) for s in SEEDS]
    print(f"  {c}: {np.mean(errs):.3f}% +/- {np.std(errs):.3f}% | {sum(1 for e in errs if e>50)}/7 fails")
print("SS cases - Unified mean/std drift:")
for c in SS_CASES:
    drifts = [abl["Unified_Full"]["per_seed"][str(s)][c].get("drift_mM", abl["Unified_Full"]["per_seed"][str(s)][c].get("drift_from_ystar_mM",999)) for s in SEEDS]
    print(f"  {c}: {np.mean(drifts):.3f} +/- {np.std(drifts):.3f} mM")
print()
print("Baseline mean/std/fails (limit cycles):")
for c in LC_CASES:
    errs = [base[str(s)][c].get("amp_err_pct",100.0) for s in SEEDS]
    print(f"  {c}: {np.mean(errs):.1f}% +/- {np.std(errs):.1f}% | {sum(1 for e in errs if e>50)}/7 fails")
print("Baseline drift (stable focus):")
for c in SS_CASES:
    drifts = [base[str(s)][c].get("drift_from_ystar_mM",999) for s in SEEDS]
    print(f"  {c}: {np.mean(drifts):.1f} +/- {np.std(drifts):.1f} mM")
