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
    print(f"Training and Logging: {model_name} (Seed {seed}, {total_steps} epochs)")
    print(f"=======================================================")

    # Initialize model architecture
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

    # Pre-select steady-state tensors for bifurcation penalty if needed
    ss_tensors = []
    if model_name == "Bifurcation-Aware FiLM":
        for c in SWEEP_CASES:
            if c['is_steady_state']:
                p_t = torch.tensor(c['params'], dtype=torch.float32)
                y_ss = torch.tensor(find_steady_state(c['params']), dtype=torch.float32)
                w = 3.5 if c['id'] in ['X2', 'X3'] else 1.0
                ss_tensors.append((p_t, y_ss, w))

    # Metric history
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

        # Compute forward rollout loss
        curr = state_b
        traj_loss = 0.0
        for k in range(K):
            prev = curr
            if model_name in ["Unconstrained Baseline MLP", "Divergence-Reg Baseline MLP"]:
                curr = model(curr, params_b)
            elif model_name in ["Proposed FiLM SciML (Pure)", "Bifurcation-Aware FiLM"]:
                curr = model(curr, params_b, target_ctot=c_tot0)
            elif model_name == "Bifurcation-Gated FiLM (Ours)":
                curr = model(curr, params_b, target_ctot=c_tot0)

            state_l = torch.mean(((curr - future_b[:, k, :]) * weights) ** 2)
            pred_v = (curr - prev) / model.dt
            true_v = (future_b[:, k, :] - (state_b if k == 0 else future_b[:, k - 1, :])) / model.dt
            rate_sc = getattr(model, 'rate_scales', model.state_scales * 100.0)
            deriv_l = torch.mean(((pred_v - true_v) / rate_sc) ** 2)
            traj_loss = traj_loss + state_l + 0.05 * deriv_l
        traj_loss = traj_loss / K

        total_loss = traj_loss

        # Auxiliary loss terms
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

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Compute raw gradient norm BEFORE clipping
        raw_grad_norm = compute_gradient_norm(model)

        # Clip gradient norm
        clipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0).item()

        optimizer.step()
        scheduler.step()

        # Validation evaluation every 10 steps
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

        # Record metrics
        history['step'].append(step)
        history['train_loss'].append(float(total_loss.item()))
        history['val_loss'].append(val_loss)
        history['grad_norm_raw'].append(float(raw_grad_norm))
        history['grad_norm_clipped'].append(float(clipped_norm))
        history['lr'].append(float(scheduler.get_last_lr()[0]))

        if step % 100 == 0 or step == total_steps:
            print(f"  Step {step:3d}/{total_steps} | Loss: {total_loss.item():.4e} | Val: {val_loss:.4e} | GradNorm: {raw_grad_norm:.2f} ({time.time()-t0:.1f}s)", flush=True)

    return history

def plot_comprehensive_loss_and_gradients(all_histories):
    print("\nRendering Publication-Quality Convergence & Gradient Norm Figure...")
    
    # Visual Styling
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.1,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })

    fig = plt.figure(figsize=(18, 12), dpi=220)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)

    colors = {
        "Unconstrained Baseline MLP": "#d62728",      # Crimson Red
        "Divergence-Reg Baseline MLP": "#ff7f0e",     # Safety Orange
        "Proposed FiLM SciML (Pure)": "#2ca02c",      # Emerald Green
        "Bifurcation-Aware FiLM": "#9467bd",          # Purple
        "Bifurcation-Gated FiLM (Ours)": "#1f77b4"     # Deep Cobalt Blue
    }

    linestyles = {
        "Unconstrained Baseline MLP": ":",
        "Divergence-Reg Baseline MLP": "-.",
        "Proposed FiLM SciML (Pure)": "--",
        "Bifurcation-Aware FiLM": "-.",
        "Bifurcation-Gated FiLM (Ours)": "-"
    }

    linewidths = {
        "Unconstrained Baseline MLP": 1.6,
        "Divergence-Reg Baseline MLP": 1.6,
        "Proposed FiLM SciML (Pure)": 1.8,
        "Bifurcation-Aware FiLM": 2.0,
        "Bifurcation-Gated FiLM (Ours)": 2.4
    }

    # -------------------------------------------------------------
    # Panel 1: Training Loss vs. Epoch (Log Scale)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for name, hist in all_histories.items():
        steps = hist['step']
        losses = hist['train_loss']
        # Smooth with moving average for cleaner visual trend
        w = 5
        smoothed = np.convolve(losses, np.ones(w)/w, mode='valid')
        ax1.plot(steps[w-1:], smoothed, label=name, color=colors[name], 
                 linestyle=linestyles[name], lw=linewidths[name])

    ax1.set_yscale('log')
    ax1.set_title("A. Training Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch / Training Step", fontsize=10)
    ax1.set_ylabel(r"Training Loss $\mathcal{L}_{\mathrm{total}}$ (Log Scale)", fontsize=10)
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 2: Validation Rollout Loss vs. Epoch (Generalization)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for name, hist in all_histories.items():
        val_steps = [s for s, v in zip(hist['step'], hist['val_loss']) if not np.isnan(v)]
        val_losses = [v for v in hist['val_loss'] if not np.isnan(v)]
        ax2.plot(val_steps, val_losses, label=name, color=colors[name], 
                 linestyle=linestyles[name], lw=linewidths[name], marker='o', markersize=3.5, alpha=0.9)

    ax2.set_yscale('log')
    ax2.set_title("B. Validation Rollout Loss vs. Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch / Training Step", fontsize=10)
    ax2.set_ylabel(r"Held-Out Val Loss $\mathcal{L}_{\mathrm{val}}$ (Log Scale)", fontsize=10)
    ax2.grid(True)

    # -------------------------------------------------------------
    # Panel 3: Gradient Norm vs. Epoch (Raw Unclipped L2 Norm)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for name, hist in all_histories.items():
        steps = hist['step']
        gnorms = hist['grad_norm_raw']
        w = 5
        smoothed_g = np.convolve(gnorms, np.ones(w)/w, mode='valid')
        ax3.plot(steps[w-1:], smoothed_g, label=name, color=colors[name], 
                 linestyle=linestyles[name], lw=linewidths[name])

    ax3.axhline(2.0, color='black', linestyle=':', lw=1.2, label="Clip Threshold ($2.0$)")
    ax3.set_yscale('log')
    ax3.set_title(r"C. Gradient Norm vs. Epoch ($\|\nabla_{\theta} \mathcal{L}\|_2$)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch / Training Step", fontsize=10)
    ax3.set_ylabel(r"Total Gradient $L_2$ Norm (Log Scale)", fontsize=10)
    ax3.grid(True)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel 4: Gradient Norm Variance & Distribution (Boxplot - Log Scale)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    gnorm_data = [hist['grad_norm_raw'] for hist in all_histories.values()]
    names_short = [
        "Vanilla\nBaseline",
        "Div-Reg\nBaseline",
        "FiLM\nPure",
        "Bifurcation\nAware",
        "Bifurcation\nGated (Ours)"
    ]
    bplot = ax4.boxplot(gnorm_data, patch_artist=True, showmeans=True, showfliers=False,
                        meanprops={"marker": "^", "markerfacecolor": "green", "markeredgecolor": "black", "markersize": 7})
    ax4.set_xticks(range(1, len(names_short) + 1))
    ax4.set_xticklabels(names_short, fontsize=8.5)
    for patch, color in zip(bplot['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in bplot['medians']:
        median.set(color='black', linewidth=1.5)

    ax4.set_yscale('log')
    ax4.set_title(r"D. Gradient Norm Distribution ($\|\nabla_{\theta} \mathcal{L}\|_2$)", fontsize=11, fontweight="bold")
    ax4.set_ylabel(r"Gradient $L_2$ Norm (Log Scale)", fontsize=10)
    ax4.grid(True, axis='y', which='both', linestyle='--', alpha=0.5)

    # -------------------------------------------------------------
    # Panel 5: Phase Trajectory: Loss vs. Gradient Norm
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    for name, hist in all_histories.items():
        w = 10
        l_smooth = np.convolve(hist['train_loss'], np.ones(w)/w, mode='valid')
        g_smooth = np.convolve(hist['grad_norm_raw'], np.ones(w)/w, mode='valid')
        ax5.plot(l_smooth, g_smooth, color=colors[name], lw=linewidths[name], 
                 label=name, alpha=0.85)
        # Mark final point
        ax5.scatter([l_smooth[-1]], [g_smooth[-1]], color=colors[name], s=40, zorder=5)

    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_title(r"E. Optimization Phase Plane ($\mathcal{L}$ vs. $\|\nabla \mathcal{L}\|_2$)", fontsize=11, fontweight="bold")
    ax5.set_xlabel(r"Loss $\mathcal{L}$ (Log Scale)", fontsize=10)
    ax5.set_ylabel(r"Gradient Norm $\|\nabla \mathcal{L}\|_2$ (Log Scale)", fontsize=10)
    ax5.grid(True)

    # -------------------------------------------------------------
    # Panel 6: Quantitative Convergence Summary Table
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    col_labels = ['Model', 'Train Loss', 'Val Loss', 'Mean ‖∇L‖', 'Max ‖∇L‖', 'Stability']
    table_rows = []
    short_model_names = [
        "Vanilla MLP",
        "Div-Reg MLP",
        "FiLM (Pure)",
        "Bif-Aware",
        "Bif-Gated"
    ]
    for (name, hist), s_name in zip(all_histories.items(), short_model_names):
        f_tr = f"{hist['train_loss'][-1]:.2e}"
        v_losses = [v for v in hist['val_loss'] if not np.isnan(v)]
        f_val = f"{v_losses[-1]:.2e}" if v_losses else "N/A"
        m_gnorm = f"{np.mean(hist['grad_norm_raw']):.3f}"
        mx_gnorm = f"{np.max(hist['grad_norm_raw']):.1f}"
        std_gnorm = np.std(hist['grad_norm_raw'])
        stability = "Stable" if std_gnorm < 0.1 else ("Moderate" if std_gnorm < 5.0 else "Explosive")
        table_rows.append([s_name, f_tr, f_val, m_gnorm, mx_gnorm, stability])

    col_widths = [0.22, 0.15, 0.15, 0.16, 0.16, 0.16]
    table = ax6.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 2.0)

    # Highlight headers
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#24292e')
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=8.5)

    # Color row borders
    for row_idx, color in enumerate(colors.values(), start=1):
        for col_idx in range(len(col_labels)):
            table[(row_idx, col_idx)].set_facecolor(color)
            table[(row_idx, col_idx)].set_alpha(0.12)
            if col_idx == 0:
                table[(row_idx, col_idx)].set_text_props(fontweight='bold')

    ax6.set_title("F. Convergence & Optimization Benchmark", fontsize=11, fontweight="bold", pad=15)

    fig.suptitle("Comprehensive Training Convergence & Gradient Dynamics Across All 5 Model Architectures", 
                 fontsize=14, fontweight="bold", y=0.98)

    # Save to both workspace root and brain directory
    out_png1 = os.path.join(CURRENT_DIR, "loss_and_gradient_norm_convergence.png")
    out_png2 = os.path.join(BRAIN_DIR, "loss_and_gradient_norm_convergence.png")
    plt.savefig(out_png1, bbox_inches="tight", dpi=220)
    plt.savefig(out_png2, bbox_inches="tight", dpi=220)
    plt.close()
    print(f"Saved figure to {out_png1} and {out_png2}")

def main():
    print("================================================================================")
    print("CONVERGENCE BENCHMARK: LOSS & GRADIENT NORM VS. EPOCH FOR ALL MODELS")
    print("================================================================================")

    out_json = os.path.join(CURRENT_DIR, "training_loss_and_grad_norm_history.json")
    if os.path.exists(out_json):
        print(f"Loading existing training history from {out_json}...")
        with open(out_json, "r") as f:
            all_histories = json.load(f)
    else:
        # Build dual-regime dataset
        dataset = build_bifurcation_dataset()
        val_dataset = dataset[:12]  # Held-out 12 validation trajectories across regimes
        train_dataset = dataset[12:]

        model_names = [
            "Unconstrained Baseline MLP",
            "Divergence-Reg Baseline MLP",
            "Proposed FiLM SciML (Pure)",
            "Bifurcation-Aware FiLM",
            "Bifurcation-Gated FiLM (Ours)"
        ]

        all_histories = {}
        for name in model_names:
            hist = train_and_log_model(name, seed=42, total_steps=400, dataset=train_dataset, val_dataset=val_dataset)
            all_histories[name] = hist

        # Save raw convergence time-series data
        with open(out_json, "w") as f:
            json.dump(all_histories, f, indent=2)
        print(f"\nSaved raw training time-series to {out_json}")

    # Render figure
    plot_comprehensive_loss_and_gradients(all_histories)

if __name__ == "__main__":
    main()
