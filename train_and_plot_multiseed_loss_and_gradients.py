import os
import sys
import time
import json
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

CURRENT_DIR = r"c:\Users\satya\Downloads\Van Der Pol"
BRAIN_DIR = r"C:\Users\satya\.gemini\antigravity-ide\brain\4fd50b55-98fd-4b91-92cf-1a428e462f37"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from biochemical_global_sciml import (
    ProposedFiLMPhysicsConstrainedMLP,
    UnconstrainedBaselineMLP,
    DT,
    DEFAULT_GUESS,
    rk4_step,
    simulate_trajectory,
    set_seed
)
from train_all_biochemical_models import (
    FiLMWithRates,
    compute_true_divergence,
    compute_unconstrained_divergence,
    compute_film_divergence
)
from train_all_bifurcation_aware_models import build_bifurcation_dataset, sample_batch
from train_and_evaluate_bifurcation_gated_multiseed import BifurcationGatedFiLMSciML
from run_mega_parameter_sweep import SWEEP_CASES, find_steady_state

SEEDS = [42, 999, 2026]
MODEL_NAMES = [
    "Unconstrained Baseline MLP",
    "Divergence-Reg Baseline MLP",
    "Proposed FiLM SciML (Pure)",
    "Bifurcation-Aware FiLM",
    "Bifurcation-Gated FiLM (Ours)"
]

def compute_gradient_norm(model):
    """Computes the total L2 norm of the gradients across all trainable parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5

def train_and_log_model(model_name, seed=42, total_steps=400, dataset=None, val_dataset=None):
    set_seed(seed)
    print(f"\n=======================================================")
    print(f"Training: {model_name} | Seed: {seed} | Steps: {total_steps}")
    print(f"=======================================================")

    if model_name == "Unconstrained Baseline MLP":
        model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
        lr = 1e-3
    elif model_name == "Divergence-Reg Baseline MLP":
        model = UnconstrainedBaselineMLP(hidden_dim=128, dt=DT)
        lr = 5e-4
    elif model_name == "Proposed FiLM SciML (Pure)":
        model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
        lr = 8e-4
    elif model_name == "Bifurcation-Aware FiLM":
        model = ProposedFiLMPhysicsConstrainedMLP(hidden_dim=128, dt=DT, m0_floor=0.5)
        lr = 3e-4
    elif model_name == "Bifurcation-Gated FiLM (Ours)":
        model = BifurcationGatedFiLMSciML(hidden_dim=128, dt=DT, m0_floor=0.5)
        lr = 8e-4
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    ss_tensors = []
    if model_name == "Bifurcation-Aware FiLM":
        for c in SWEEP_CASES:
            if c['is_steady_state']:
                p_t = torch.tensor(c['params'], dtype=torch.float32)
                y_ss = torch.tensor(find_steady_state(c['params']), dtype=torch.float32)
                w = 3.5 if c['id'] in ['X2', 'X3'] else 1.0
                ss_tensors.append((p_t, y_ss, w))

    history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'grad_norm_raw': [],
        'grad_norm_clipped': [],
        'lr': []
    }

    t0 = time.time()
    for step in range(1, total_steps + 1):
        model.train()
        K = int(np.interp(step, [1, total_steps * 0.4, total_steps], [5, 12, 16]))

        state_b, params_b, future_b = sample_batch(dataset, K, batch_size=64)
        c_tot0 = state_b[:, 0:1] + state_b[:, 1:2] + state_b[:, 2:3]
        weights = torch.tensor([1.0, 1.0, 2.0, 5.0]) / model.state_scales

        curr = state_b
        traj_loss = 0.0
        for k in range(K):
            prev = curr
            if model_name in ["Unconstrained Baseline MLP", "Divergence-Reg Baseline MLP"]:
                curr = model(curr, params_b)
            else:
                curr = model(curr, params_b, target_ctot=c_tot0)

            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            rate_sc = getattr(model, 'rate_scales', model.state_scales * 100.0)
            deriv_l = torch.mean(((pred_v - true_v) / rate_sc) ** 2)
            traj_loss = traj_loss + state_l + 0.05 * deriv_l
        traj_loss = traj_loss / K
        total_loss = traj_loss

        if model_name == "Divergence-Reg Baseline MLP":
            sub_s = state_b[:16]
            sub_p = params_b[:16]
            true_div = compute_true_divergence(sub_s, sub_p)
            pred_div = compute_unconstrained_divergence(model, sub_s, sub_p)
            div_loss = torch.mean(((pred_div - true_div) / 100.0) ** 2)
            total_loss = total_loss + 1e-4 * div_loss

        elif model_name == "Bifurcation-Aware FiLM":
            regime_pen = 0.0
            for p_t, y_ss, w in ss_tensors:
                s_in = y_ss.unsqueeze(0)
                p_in = p_t.unsqueeze(0)
                c_t0 = s_in[:, 0:1] + s_in[:, 1:2] + s_in[:, 2:3]
                s_next = model(s_in, p_in, target_ctot=c_t0)
                v_ss = (s_next - s_in) / model.dt
                regime_pen = regime_pen + w * torch.mean((v_ss / model.state_scales) ** 2)
            regime_pen = regime_pen / len(ss_tensors)
            total_loss = total_loss + 0.05 * regime_pen

        elif model_name == "Bifurcation-Gated FiLM (Ours)":
            ka_vals = params_b[:, 0]
            regime_labels = (ka_vals <= 8.5).float().unsqueeze(1)
            p_norm = params_b / model.param_scales
            p_embed = model.p_net(p_norm)
            pred_gate = model.gate_net(p_embed)
            gate_loss = torch.nn.functional.binary_cross_entropy(pred_gate, regime_labels)
            total_loss = total_loss + 0.3 * gate_loss

        optimizer.zero_grad()
        total_loss.backward()

        raw_grad_norm = compute_gradient_norm(model)
        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0).item()

        optimizer.step()
        scheduler.step()

        val_loss = float('nan')
        if step % 10 == 0 or step == total_steps or step == 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for val_item in val_dataset:
                    v_traj = val_item['traj']
                    v_p = val_item['params'].unsqueeze(0)
                    v_s = v_traj[0:1]
                    v_ctot = v_s[:, 0:1] + v_s[:, 1:2] + v_s[:, 2:3]
                    v_curr = v_s
                    for k in range(min(30, len(v_traj) - 1)):
                        if model_name in ["Unconstrained Baseline MLP", "Divergence-Reg Baseline MLP"]:
                            v_curr = model(v_curr, v_p)
                        else:
                            v_curr = model(v_curr, v_p, target_ctot=v_ctot)
                        val_losses.append(torch.mean(((v_curr - v_traj[k + 1:k + 2]) / model.state_scales) ** 2).item())
                val_loss = float(np.mean(val_losses))

        history['step'].append(step)
        history['train_loss'].append(float(total_loss.item()))
        history['val_loss'].append(val_loss)
        history['grad_norm_raw'].append(float(raw_grad_norm))
        history['grad_norm_clipped'].append(float(clipped_norm))
        history['lr'].append(float(scheduler.get_last_lr()[0]))

        if step % 100 == 0 or step == total_steps:
            print(f"  [{model_name[:18]}] Step {step:3d}/{total_steps} | Loss: {total_loss.item():.4e} | Val: {val_loss:.4e} | Grad: {raw_grad_norm:.2f} ({time.time()-t0:.1f}s)", flush=True)

    return history

def plot_multiseed_comprehensive_figure(multiseed_histories):
    print("\nRendering Multi-Seed Loss & Gradient Norm Publication Figure...")
    
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.1,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })

    fig = plt.figure(figsize=(19, 12), dpi=220)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.26)

    colors = {
        "Unconstrained Baseline MLP": "#d62728",
        "Divergence-Reg Baseline MLP": "#ff7f0e",
        "Proposed FiLM SciML (Pure)": "#2ca02c",
        "Bifurcation-Aware FiLM": "#9467bd",
        "Bifurcation-Gated FiLM (Ours)": "#1f77b4"
    }

    linestyles = {
        "Unconstrained Baseline MLP": ":",
        "Divergence-Reg Baseline MLP": "-.",
        "Proposed FiLM SciML (Pure)": "--",
        "Bifurcation-Aware FiLM": "-.",
        "Bifurcation-Gated FiLM (Ours)": "-"
    }

    short_names = {
        "Unconstrained Baseline MLP": "Vanilla MLP",
        "Divergence-Reg Baseline MLP": "Div-Reg MLP",
        "Proposed FiLM SciML (Pure)": "FiLM (Pure)",
        "Bifurcation-Aware FiLM": "Bif-Aware",
        "Bifurcation-Gated FiLM (Ours)": "Bif-Gated (Ours)"
    }

    # -------------------------------------------------------------
    # Panel 1: Multi-Seed Training Loss vs. Epoch (Mean ± Std)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    w = 5
    for name in MODEL_NAMES:
        seed_data = multiseed_histories[name]
        steps = seed_data[str(SEEDS[0])]['step']
        all_tr = []
        for s in SEEDS:
            raw_tr = np.array(seed_data[str(s)]['train_loss'])
            sm = np.convolve(raw_tr, np.ones(w)/w, mode='valid')
            all_tr.append(sm)
            # Plot individual seed faint lines
            ax1.plot(steps[w-1:], sm, color=colors[name], alpha=0.22, lw=0.9, linestyle=linestyles[name])

        all_tr = np.array(all_tr)
        mean_tr = np.mean(all_tr, axis=0)
        std_tr = np.std(all_tr, axis=0)
        
        ax1.plot(steps[w-1:], mean_tr, label=short_names[name], color=colors[name],
                 linestyle=linestyles[name], lw=2.2)
        ax1.fill_between(steps[w-1:], np.maximum(mean_tr - std_tr, 1e-6), mean_tr + std_tr,
                         color=colors[name], alpha=0.15)

    ax1.set_yscale('log')
    ax1.set_title("A. Multi-Seed Training Loss (Mean ± 1σ across 3 Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch / Training Step", fontsize=10)
    ax1.set_ylabel(r"Training Loss $\mathcal{L}_{\mathrm{total}}$ (Log Scale)", fontsize=10)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 2: Multi-Seed Validation Rollout Loss vs. Epoch
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for name in MODEL_NAMES:
        seed_data = multiseed_histories[name]
        # Val steps where val is not NaN
        sample_hist = seed_data[str(SEEDS[0])]
        val_idx = [i for i, v in enumerate(sample_hist['val_loss']) if not np.isnan(v)]
        val_steps = [sample_hist['step'][i] for i in val_idx]

        all_val = []
        for s in SEEDS:
            v_losses = [seed_data[str(s)]['val_loss'][i] for i in val_idx]
            all_val.append(v_losses)
            ax2.plot(val_steps, v_losses, color=colors[name], alpha=0.22, lw=0.9)

        all_val = np.array(all_val)
        mean_val = np.mean(all_val, axis=0)
        std_val = np.std(all_val, axis=0)

        ax2.plot(val_steps, mean_val, label=short_names[name], color=colors[name],
                 linestyle=linestyles[name], lw=2.2, marker='o', markersize=3.5, alpha=0.95)
        ax2.fill_between(val_steps, np.maximum(mean_val - std_val, 1e-6), mean_val + std_val,
                         color=colors[name], alpha=0.15)

    ax2.set_yscale('log')
    ax2.set_title("B. Multi-Seed Validation Rollout Loss (Mean ± 1σ)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch / Training Step", fontsize=10)
    ax2.set_ylabel(r"Val Rollout Loss $\mathcal{L}_{\mathrm{val}}$ (Log Scale)", fontsize=10)
    ax2.grid(True)

    # -------------------------------------------------------------
    # Panel 3: Multi-Seed Gradient Norm vs. Epoch (Log Scale)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for name in MODEL_NAMES:
        seed_data = multiseed_histories[name]
        steps = seed_data[str(SEEDS[0])]['step']
        all_gn = []
        for s in SEEDS:
            raw_gn = np.array(seed_data[str(s)]['grad_norm_raw'])
            sm = np.convolve(raw_gn, np.ones(w)/w, mode='valid')
            all_gn.append(sm)
            ax3.plot(steps[w-1:], sm, color=colors[name], alpha=0.22, lw=0.9, linestyle=linestyles[name])

        all_gn = np.array(all_gn)
        mean_gn = np.mean(all_gn, axis=0)
        std_gn = np.std(all_gn, axis=0)

        ax3.plot(steps[w-1:], mean_gn, label=short_names[name], color=colors[name],
                 linestyle=linestyles[name], lw=2.2)
        ax3.fill_between(steps[w-1:], np.maximum(mean_gn - std_gn, 1e-6), mean_gn + std_gn,
                         color=colors[name], alpha=0.15)

    ax3.axhline(2.0, color='black', linestyle=':', lw=1.2, label="Clip Threshold ($2.0$)")
    ax3.set_yscale('log')
    ax3.set_title(r"C. Gradient Norm vs. Epoch ($\|\nabla_{\theta} \mathcal{L}\|_2$ across Seeds)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch / Training Step", fontsize=10)
    ax3.set_ylabel(r"Gradient $L_2$ Norm (Log Scale)", fontsize=10)
    ax3.grid(True)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 4: Aggregated Gradient Norm Distribution (All Seeds)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    combined_gnorm = []
    box_names = []
    for name in MODEL_NAMES:
        pts = []
        for s in SEEDS:
            pts.extend(multiseed_histories[name][str(s)]['grad_norm_raw'])
        combined_gnorm.append(pts)
        box_names.append(short_names[name].replace(" ", "\n"))

    bplot = ax4.boxplot(combined_gnorm, patch_artist=True, showmeans=True, showfliers=False,
                        meanprops={"marker": "^", "markerfacecolor": "green", "markeredgecolor": "black", "markersize": 7})
    ax4.set_xticks(range(1, len(box_names) + 1))
    ax4.set_xticklabels(box_names, fontsize=8.5)
    for patch, color in zip(bplot['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bplot['medians']:
        median.set(color='black', linewidth=1.5)

    ax4.set_yscale('log')
    ax4.set_title("D. Gradient Norm Distribution (Aggregated across Seeds)", fontsize=11, fontweight="bold")
    ax4.set_ylabel(r"Gradient $L_2$ Norm (Log Scale)", fontsize=10)
    ax4.grid(True, axis='y', which='both', linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # Panel 5: Seed Sensitivity Comparison (Final Val Loss per Seed)
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    n_models = len(MODEL_NAMES)
    x = np.arange(n_models)
    bar_width = 0.25

    seed_colors = ['#4575b4', '#74add1', '#f46d43']
    for idx, s in enumerate(SEEDS):
        val_losses_s = []
        for name in MODEL_NAMES:
            v_hist = multiseed_histories[name][str(s)]['val_loss']
            last_val = [v for v in v_hist if not np.isnan(v)][-1]
            val_losses_s.append(last_val)
        ax5.bar(x + idx * bar_width, val_losses_s, width=bar_width, 
                label=f"Seed {s}", color=seed_colors[idx % len(seed_colors)], alpha=0.85, edgecolor='black', lw=0.8)

    ax5.set_xticks(x + bar_width)
    ax5.set_xticklabels([short_names[m].replace(" ", "\n") for m in MODEL_NAMES], fontsize=8.5)
    ax5.set_yscale('log')
    ax5.set_title("E. Seed Sensitivity: Final Val Loss per Seed", fontsize=11, fontweight="bold")
    ax5.set_ylabel(r"Final Val Rollout Loss (Log Scale)", fontsize=10)
    ax5.grid(True, axis='y')
    ax5.legend(loc="upper right", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 6: Multi-Seed Statistical Summary Table
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    col_labels = ['Model', 'Val Loss (Mean±Std)', 'Mean ‖∇L‖', 'Peak ‖∇L‖', 'Consistency']
    table_rows = []

    for name in MODEL_NAMES:
        s_data = multiseed_histories[name]
        final_vals = []
        mean_gns = []
        max_gns = []
        for s in SEEDS:
            v_hist = s_data[str(s)]['val_loss']
            final_vals.append([v for v in v_hist if not np.isnan(v)][-1])
            mean_gns.append(np.mean(s_data[str(s)]['grad_norm_raw']))
            max_gns.append(np.max(s_data[str(s)]['grad_norm_raw']))

        v_m, v_s = np.mean(final_vals), np.std(final_vals)
        gn_m, gn_s = np.mean(mean_gns), np.std(mean_gns)
        peak_gn = np.max(max_gns)

        val_str = f"{v_m:.2e} ± {v_s:.1e}"
        gn_str = f"{gn_m:.3f} ± {gn_s:.3f}"
        peak_str = f"{peak_gn:.1f}"
        consistency = "Invariant" if v_s / v_m < 0.25 and peak_gn < 2.0 else ("Moderate" if peak_gn < 10.0 else "Explosive")

        s_label = short_names[name].replace(" (Ours)", "")
        table_rows.append([s_label, val_str, gn_str, peak_str, consistency])

    col_widths = [0.21, 0.28, 0.19, 0.14, 0.18]
    table = ax6.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1.0, 2.0)

    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#24292e')
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8.2)

    for row_idx, color in enumerate(colors.values(), start=1):
        for col_idx in range(len(col_labels)):
            table[(row_idx, col_idx)].set_facecolor(color)
            table[(row_idx, col_idx)].set_alpha(0.12)
            if col_idx == 0:
                table[(row_idx, col_idx)].set_text_props(fontweight='bold')

    ax6.set_title("F. Multi-Seed Optimization & Convergence Metrics", fontsize=11, fontweight="bold", pad=15)

    fig.suptitle(f"Multi-Seed Convergence & Gradient Dynamics Benchmark (Seeds: {', '.join(map(str, SEEDS))})", 
                 fontsize=14, fontweight="bold", y=0.98)

    out_png1 = os.path.join(CURRENT_DIR, "multiseed_loss_and_gradient_norm_convergence.png")
    out_png2 = os.path.join(BRAIN_DIR, "multiseed_loss_and_gradient_norm_convergence.png")
    plt.savefig(out_png1, bbox_inches="tight", dpi=220)
    plt.savefig(out_png2, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Saved multi-seed figure to {out_png1} and {out_png2}")

def main():
    print("================================================================================")
    print(f"MULTI-SEED CONVERGENCE BENCHMARK: LOSS & GRADIENT NORM (SEEDS: {SEEDS})")
    print("================================================================================")

    out_json = os.path.join(CURRENT_DIR, "multiseed_training_history.json")
    multiseed_histories = {}

    if os.path.exists(out_json):
        print(f"Loading existing multi-seed history from {out_json}...")
        with open(out_json, "r") as f:
            multiseed_histories = json.load(f)

    # Initialize structure
    for name in MODEL_NAMES:
        if name not in multiseed_histories:
            multiseed_histories[name] = {}

    # Check if we can seed with already computed seed 42 from training_loss_and_grad_norm_history.json
    single_seed_json = os.path.join(CURRENT_DIR, "training_loss_and_grad_norm_history.json")
    if os.path.exists(single_seed_json):
        with open(single_seed_json, "r") as f:
            s42_data = json.load(f)
        for name in MODEL_NAMES:
            if "42" not in multiseed_histories[name] and name in s42_data:
                print(f"Preloaded Seed 42 for '{name}' from single-seed cache.")
                multiseed_histories[name]["42"] = s42_data[name]

    # Build dataset if needed
    needs_training = False
    for name in MODEL_NAMES:
        for s in SEEDS:
            if str(s) not in multiseed_histories[name]:
                needs_training = True
                break

    if needs_training:
        print("\nBuilding dual-regime dataset for multi-seed training...")
        dataset = build_bifurcation_dataset()
        val_dataset = dataset[:12]
        train_dataset = dataset[12:]

        for s in SEEDS:
            print(f"\n>>>>>>>>>>>>>>> RUNNING FOR SEED {s} <<<<<<<<<<<<<<<")
            for name in MODEL_NAMES:
                if str(s) in multiseed_histories[name]:
                    print(f"  Skipping {name} (Seed {s} already in cache)")
                    continue
                hist = train_and_log_model(name, seed=s, total_steps=400, dataset=train_dataset, val_dataset=val_dataset)
                multiseed_histories[name][str(s)] = hist

                # Save intermediate progress
                with open(out_json, "w") as f:
                    json.dump(multiseed_histories, f, indent=2)

        print(f"\nAll seeds complete. Saved multi-seed time series to {out_json}")
    else:
        print("All seeds already cached in multiseed_training_history.json.")

    plot_multiseed_comprehensive_figure(multiseed_histories)

if __name__ == "__main__":
    main()
