import os
import sys
import glob
import json
import hashlib
import numpy as np
import scipy.stats
import torch

SEEDS = [42, 101, 777, 999, 1234, 2024, 5555]

def compute_file_sha256(filepath):
    """Compute standard SHA-256 hash of checkpoint file bytes."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_weights_sha256(state_dict):
    """Compute parameter tensor SHA-256 hash."""
    hasher = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        hasher.update(state_dict[k].cpu().numpy().tobytes())
    return hasher.hexdigest()

def sarles_bimodality(values):
    """
    Sarle's Bimodality Coefficient:
    BC = (skewness^2 + 1) / kurtosis
    where kurtosis is Pearson's kurtosis (normal distribution = 3.0).
    Sample-size finite correction can also be applied:
    BC = (g^2 + 1) / (k + 3*(n-1)^2/((n-2)*(n-3)))
    Standard benchmark threshold: BC > 5/9 (~0.555) indicates bimodal or multimodal.
    """
    vals = np.array(values, dtype=np.float64)
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    cv = float(std / max(abs(mean), 1e-8))
    
    if std < 1e-7:
        return {
            "mean": mean, "std": std, "cv": cv,
            "skew": 0.0, "kurt": 3.0, "bc": 0.333333,
            "is_bimodal": False, "n": n
        }
    
    skew = float(scipy.stats.skew(vals, bias=False))
    kurt = float(scipy.stats.kurtosis(vals, fisher=False, bias=False)) # Pearson kurtosis
    bc = float((skew**2 + 1.0) / max(kurt, 1e-4))
    is_bimodal = bool(bc > 0.555556 and cv > 0.15)
    
    return {
        "mean": mean, "std": std, "cv": cv,
        "skew": skew, "kurt": kurt, "bc": bc,
        "is_bimodal": is_bimodal, "n": n
    }

def main():
    results = {
        "checksums": {},
        "bimodality_audits": {}
    }

    # =========================================================================
    # PART 1: COMPLETE CHECKPOINT PROVENANCE CHECKSUMS (ALL SAVED .PT FILES)
    # =========================================================================
    print("=" * 80)
    print("1. COMPUTING SHA-256 PROVENANCE CHECKSUMS ACROSS ALL SAVED CHECKPOINTS")
    print("=" * 80)

    # 1. Van der Pol Checkpoints
    vdp_dir = r"c:\Users\satya\Downloads\Van Der Pol\vanderpol\scratch"
    vdp_pts = sorted(glob.glob(os.path.join(vdp_dir, "*.pt")))
    results["checksums"]["vanderpol"] = {}
    print("\n[Van der Pol Models]")
    for pt in vdp_pts:
        fname = os.path.basename(pt)
        f_hash = compute_file_sha256(pt)
        sd = torch.load(pt, map_location="cpu", weights_only=True)
        w_hash = compute_weights_sha256(sd)
        results["checksums"]["vanderpol"][fname] = {
            "file_sha256": f_hash,
            "weights_sha256": w_hash,
            "short_id": w_hash[:16]
        }
        print(f"  {fname:<35} | Weights: {w_hash[:16]} | File: {f_hash[:16]}")

    # 2. FitzHugh-Nagumo Checkpoints
    fhn_dirs = [
        r"c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo\scratch",
        r"c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo"
    ]
    results["checksums"]["fitzhugh_nagumo"] = {}
    print("\n[FitzHugh-Nagumo Models]")
    for d in fhn_dirs:
        for pt in sorted(glob.glob(os.path.join(d, "*.pt"))):
            fname = os.path.basename(pt)
            if fname not in results["checksums"]["fitzhugh_nagumo"]:
                f_hash = compute_file_sha256(pt)
                sd = torch.load(pt, map_location="cpu", weights_only=True)
                w_hash = compute_weights_sha256(sd)
                results["checksums"]["fitzhugh_nagumo"][fname] = {
                    "file_sha256": f_hash,
                    "weights_sha256": w_hash,
                    "short_id": w_hash[:16]
                }
                print(f"  {fname:<35} | Weights: {w_hash[:16]} | File: {f_hash[:16]}")

    # 3. Lorenz Checkpoints
    lorenz_pts = sorted(glob.glob(r"c:\Users\satya\Downloads\Van Der Pol\*lorenz*.pt"))
    results["checksums"]["lorenz"] = {}
    print("\n[Lorenz 1963 Models]")
    for pt in lorenz_pts:
        fname = os.path.basename(pt)
        f_hash = compute_file_sha256(pt)
        sd = torch.load(pt, map_location="cpu", weights_only=True)
        w_hash = compute_weights_sha256(sd)
        results["checksums"]["lorenz"][fname] = {
            "file_sha256": f_hash,
            "weights_sha256": w_hash,
            "short_id": w_hash[:16]
        }
        print(f"  {fname:<35} | Weights: {w_hash[:16]} | File: {f_hash[:16]}")

    # 4. Biochemical Checkpoints (7 Canonical Seeds)
    bio_pts = sorted(glob.glob(r"c:\Users\satya\Downloads\Van Der Pol\unified_sciml_seed_*.pt"))
    results["checksums"]["biochemical_unified"] = {}
    print("\n[Biochemical Unified 7-Seed Models]")
    for pt in bio_pts:
        fname = os.path.basename(pt)
        f_hash = compute_file_sha256(pt)
        sd = torch.load(pt, map_location="cpu", weights_only=True)
        w_hash = compute_weights_sha256(sd)
        results["checksums"]["biochemical_unified"][fname] = {
            "file_sha256": f_hash,
            "weights_sha256": w_hash,
            "short_id": w_hash[:16]
        }
        print(f"  {fname:<35} | Weights: {w_hash[:16]} | File: {f_hash[:16]}")

    # =========================================================================
    # PART 2: COMPREHENSIVE TIER-1 SARLE'S BIMODALITY SCREENING (ALL 4 SYSTEMS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. SCREENING SARLE'S BIMODALITY COEFFICIENT ACROSS ALL SYSTEMS & REGIMES")
    print("=" * 80)

    # A. Van der Pol Oscillator (Model A 7 seeds)
    # Load VDP metrics from verify_mu_and_model_b.py output or run
    vdp_pers_a = [6.672075, 6.677722, 6.646360, 6.689405, 6.643111, 6.613877, 6.630488]
    vdp_amps_a = [3.982240, 3.988582, 3.980281, 3.989033, 3.982261, 3.980989, 3.979887]
    vdp_muc_a = [0.01699608, 0.03273579, 0.05914341, 0.03526651, 0.01274499, 0.07386795, 0.02581320]

    b_vdp_p = sarles_bimodality(vdp_pers_a)
    b_vdp_a = sarles_bimodality(vdp_amps_a)
    b_vdp_mu = sarles_bimodality(vdp_muc_a)

    results["bimodality_audits"]["vanderpol"] = {
        "period": b_vdp_p,
        "amplitude": b_vdp_a,
        "mu_c": b_vdp_mu
    }
    print("\n[Van der Pol Oscillator - Model A From-Scratch (7 Seeds)]")
    print(f"  Period T     : Mean={b_vdp_p['mean']:.4f}, Std={b_vdp_p['std']:.4f}, CV={b_vdp_p['cv']:.4f}, Skew={b_vdp_p['skew']:+.3f}, Kurt={b_vdp_p['kurt']:.3f}, BC={b_vdp_p['bc']:.3f} -> {'[BIMODAL]' if b_vdp_p['is_bimodal'] else '[UNIMODAL]'}")
    print(f"  Amplitude    : Mean={b_vdp_a['mean']:.4f}, Std={b_vdp_a['std']:.4f}, CV={b_vdp_a['cv']:.4f}, Skew={b_vdp_a['skew']:+.3f}, Kurt={b_vdp_a['kurt']:.3f}, BC={b_vdp_a['bc']:.3f} -> {'[BIMODAL]' if b_vdp_a['is_bimodal'] else '[UNIMODAL]'}")
    print(f"  Crossing mu_c: Mean={b_vdp_mu['mean']:+.5f}, Std={b_vdp_mu['std']:.5f}, CV={b_vdp_mu['cv']:.4f}, Skew={b_vdp_mu['skew']:+.3f}, Kurt={b_vdp_mu['kurt']:.3f}, BC={b_vdp_mu['bc']:.3f} -> {'[BIMODAL]' if b_vdp_mu['is_bimodal'] else '[UNIMODAL]'}")

    # B. FitzHugh-Nagumo (Model B Residual across 7 seeds from table1_fhn_replication.json)
    fhn_json_path = r"c:\Users\satya\Downloads\Van Der Pol\fitzhugh_nagumo\scratch\table1_fhn_replication.json"
    if os.path.exists(fhn_json_path):
        with open(fhn_json_path) as f:
            fhn_data = json.load(f)
        fhn_b_pers = [item["period"] for item in fhn_data.get("proposed_seeds", [])]
        fhn_b_amps = [item["q4_amp"] for item in fhn_data.get("proposed_seeds", [])]
        b_fhn_p = sarles_bimodality(fhn_b_pers)
        b_fhn_a = sarles_bimodality(fhn_b_amps)
        results["bimodality_audits"]["fitzhugh_nagumo_model_b"] = {
            "period": b_fhn_p,
            "amplitude": b_fhn_a
        }
        print("\n[FitzHugh-Nagumo - Model B Residual (7 Seeds)]")
        print(f"  Period T     : Mean={b_fhn_p['mean']:.4f}, Std={b_fhn_p['std']:.4f}, CV={b_fhn_p['cv']:.4f}, Skew={b_fhn_p['skew']:+.3f}, Kurt={b_fhn_p['kurt']:.3f}, BC={b_fhn_p['bc']:.3f} -> {'[BIMODAL]' if b_fhn_p['is_bimodal'] else '[UNIMODAL]'}")
        print(f"  Amplitude    : Mean={b_fhn_a['mean']:.4f}, Std={b_fhn_a['std']:.4f}, CV={b_fhn_a['cv']:.4f}, Skew={b_fhn_a['skew']:+.3f}, Kurt={b_fhn_a['kurt']:.3f}, BC={b_fhn_a['bc']:.3f} -> {'[BIMODAL]' if b_fhn_a['is_bimodal'] else '[UNIMODAL]'}")

    # C. Lorenz 1963 Attractor (5 seeds from lorenz_multiseed_regularized_results.json)
    lorenz_json_path = r"c:\Users\satya\Downloads\Van Der Pol\lorenz_multiseed_regularized_results.json"
    if os.path.exists(lorenz_json_path):
        with open(lorenz_json_path) as f:
            lor_data = json.load(f)
        if "per_seed_results" in lor_data:
            lor_l1 = [item["lambda_1"] for item in lor_data["per_seed_results"]]
            lor_w1 = [item["w1_x"] for item in lor_data["per_seed_results"]]
            lor_dky = [item["d_ky"] for item in lor_data["per_seed_results"]]
            b_lor_l1 = sarles_bimodality(lor_l1)
            b_lor_w1 = sarles_bimodality(lor_w1)
            b_lor_dky = sarles_bimodality(lor_dky)
            results["bimodality_audits"]["lorenz"] = {
                "lambda_1": b_lor_l1,
                "w1_x": b_lor_w1,
                "d_ky": b_lor_dky
            }
            print("\n[Lorenz 1963 Chaotic Attractor (Multi-Seed)]")
            print(f"  Lambda_1     : Mean={b_lor_l1['mean']:.4f}, Std={b_lor_l1['std']:.4f}, CV={b_lor_l1['cv']:.4f}, Skew={b_lor_l1['skew']:+.3f}, Kurt={b_lor_l1['kurt']:.3f}, BC={b_lor_l1['bc']:.3f} -> {'[BIMODAL]' if b_lor_l1['is_bimodal'] else '[UNIMODAL]'}")
            print(f"  W1(X) Dist   : Mean={b_lor_w1['mean']:.4f}, Std={b_lor_w1['std']:.4f}, CV={b_lor_w1['cv']:.4f}, Skew={b_lor_w1['skew']:+.3f}, Kurt={b_lor_w1['kurt']:.3f}, BC={b_lor_w1['bc']:.3f} -> {'[BIMODAL]' if b_lor_w1['is_bimodal'] else '[UNIMODAL]'}")
            print(f"  D_KY Dim     : Mean={b_lor_dky['mean']:.4f}, Std={b_lor_dky['std']:.4f}, CV={b_lor_dky['cv']:.4f}, Skew={b_lor_dky['skew']:+.3f}, Kurt={b_lor_dky['kurt']:.3f}, BC={b_lor_dky['bc']:.3f} -> {'[BIMODAL]' if b_lor_dky['is_bimodal'] else '[UNIMODAL]'}")

    # D. Biochemical Oscillator (All 12 Regimes X1-X12 across 7 seeds)
    bio_json_path = r"c:\Users\satya\Downloads\Van Der Pol\unified_sciml_7seed_results.json"
    if os.path.exists(bio_json_path):
        with open(bio_json_path) as f:
            bio_data = json.load(f)
        results["bimodality_audits"]["biochemical_regimes"] = {}
        print("\n[Biochemical Oscillator - All 12 Regimes Across 7 Seeds]")
        regimes = [f"X{i}" for i in range(1, 13)]
        per_seed = bio_data.get("per_seed", {})
        for r_name in regimes:
            errs = []
            r_type = ""
            for s in SEEDS:
                s_str = str(s)
                if s_str in per_seed and r_name in per_seed[s_str]:
                    errs.append(per_seed[s_str][r_name]["error"])
                    r_type = per_seed[s_str][r_name].get("regime", "")
            if errs:
                b_r = sarles_bimodality(errs)
                results["bimodality_audits"]["biochemical_regimes"][r_name] = b_r
                status_str = "[BIMODAL]" if b_r["is_bimodal"] else "[UNIMODAL]"
                unit = "%" if "Limit" in r_type else "mM"
                print(f"  {r_name:<4} ({r_type:<12}): Mean={b_r['mean']:8.4f} {unit}, Std={b_r['std']:8.4f}, CV={b_r['cv']:6.2f}, Skew={b_r['skew']:+6.2f}, Kurt={b_r['kurt']:6.2f}, BC={b_r['bc']:.3f} -> {status_str}")

    # Save complete audit to json
    out_audit = r"c:\Users\satya\Downloads\Van Der Pol\scratch\complete_provenance_and_bimodality_audit.json"
    os.makedirs(os.path.dirname(out_audit), exist_ok=True)
    with open(out_audit, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete audit to: {out_audit}")

if __name__ == "__main__":
    main()
