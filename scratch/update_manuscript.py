import os

dst_dirs = [
    r"C:\Users\satya\.gemini\antigravity-ide\brain\fac6f262-a7ea-43cd-9cf8-6ed5750f1c06",
    r"C:\Users\satya\.gemini\antigravity-ide\brain\c0229372-9c2c-4e39-b054-0c23ec47512a"
]

sec6_text = """## 6. Comprehensive Parameter Sweep Across Deep Extrapolation ($k_a > 10$, Diverse $k_{d2}, k_{d3}$)

To rigorously test model generalization well beyond the nominal training box ($k_a \\in [2, 8]$, $k_{d2} \\in [3000, 4500]$, $k_{d3} \\in [0.6, 1.0]$), we evaluated 12 diverse parameter configurations across all 7 canonical seeds (`[42, 101, 2024, 777, 999, 1234, 5555]`):

- **Deep Extrapolation ($k_a > 10$) [Stable Region: Pure Steady-State Flat Line]**: Past the supercritical Hopf bifurcation ($k_a \\approx 7.1 - 8.2$), the real physical system possesses no sustained oscillations ($\\mathrm{Re}(\\lambda) < 0$). On the attractor, **the ground truth is literally a flat horizontal straight line** ($T = 0.0\\,\\mathrm{s}, \\mathrm{Amp} = 0.0\\,\\mathrm{mM}$). Models are evaluated on their ability to maintain this flat line versus hallucinating spurious oscillations (**X1–X6**).
- **Diverse $k_{d2}, k_{d3}$ Regimes [Oscillatory Limit Cycles]**: Explores ultra-low $k_a=1.0$ (slow relaxation cycle, $T \\approx 0.70\\,\\mathrm{s}$), low $k_a=2.5$ ($T \\approx 0.36\\,\\mathrm{s}$, peak $886\\,\\mathrm{mM}$), low $k_{d2}$ with high $k_{d3}$ ($k_{d2}=2600, k_{d3}=1.1$), elevated $k_{d3}=1.3$, high $k_{d2}=5200$ with low $k_{d3}=0.35$ ($A = 813\\,\\mathrm{mM}$), and near-critical Hopf boundary oscillations ($k_a=8.5$).

### Summary Multi-Seed Benchmark (12 Regimes across 7 Seeds)

| Regime ID | Regime Description | Kinetic Parameters $(k_a, k_{d2}, k_{d3})$ | Dynamics Type | FiLM Bifurcation-Aware Amp / Period Err | FiLM Pure (Original) Amp / Period Err | FiLM Pure Drift (mM) | Vanilla Baseline Drift (mM) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **X1** | Deep Extrap ($k_a=11$) | $(11.0, 6000.0, 0.15)$ | **Flat Straight Line** | **$12.66 \\pm 8.95\\,\\mathrm{mM}$** | $910.63 \\pm 803.19\\,\\mathrm{mM}$ | **$0.0035$** | $1066.0 \\pm 792.3$ |
| **X2** | Deep Extrap ($k_a=12$) | $(12.0, 6000.0, 0.15)$ | **Flat Straight Line** | $454.68 \\pm 990.52\\,\\mathrm{mM}$ | $912.92 \\pm 753.43\\,\\mathrm{mM}$ | **$0.0035$** | $1094.5 \\pm 834.7$ |
| **X3** | Deep Extrap ($k_a=13$) | $(13.0, 6000.0, 0.25)$ | **Flat Straight Line** | $407.95 \\pm 873.16\\,\\mathrm{mM}$ | $326.79 \\pm 336.75\\,\\mathrm{mM}$ | **$0.0029$** | $996.7 \\pm 215.0$ |
| **X4** | Boundary Focus ($k_a=10.5$) | $(10.5, 4600.0, 0.60)$ | **Flat Straight Line** | **$7.06 \\pm 1.83\\,\\mathrm{mM}$** | $206.27 \\pm 142.61\\,\\mathrm{mM}$ | **$0.0032$** | $874.0 \\pm 92.2$ |
| **X5** | Stiff Focus ($k_a=12$) | $(12.0, 4500.0, 0.50)$ | **Flat Straight Line** | **$5.32 \\pm 1.70\\,\\mathrm{mM}$** | $364.04 \\pm 194.77\\,\\mathrm{mM}$ | **$0.0032$** | $1007.0 \\pm 80.5$ |
| **X6** | Extreme Collapse ($k_a=14$) | $(14.0, 5000.0, 0.30)$ | **Flat Straight Line** | **$5.56 \\pm 2.37\\,\\mathrm{mM}$** | $400.15 \\pm 170.88\\,\\mathrm{mM}$ | **$0.0029$** | $1210.4 \\pm 88.1$ |
| **X7** | Ultra-Low $k_a$ Pulse | $(1.0, 3000.0, 0.80)$ | Sustained Cycle | $30.33 \\pm 7.27\\%$ | $29.01 \\pm 12.52\\%$ | **$0.0029$** | $995.4 \\pm 619.2$ |
| **X8** | Low $k_a$ Asymmetric | $(2.5, 3200.0, 0.80)$ | Sustained Cycle | **$3.48 \\pm 1.47\\%$** | $5.17 \\pm 4.45\\%$ | **$0.0029$** | $380.4 \\pm 412.5$ |
| **X9** | Low $k_{d2}$, High $k_{d3}$ | $(3.5, 2600.0, 1.10)$ | Sustained Cycle | **$10.10 \\pm 3.42\\%$** | $13.88 \\pm 4.30\\%$ | **$0.0032$** | $807.5 \\pm 555.0$ |
| **X10** | Elevated $k_{d3}$ Fast | $(4.5, 3500.0, 1.30)$ | Sustained Cycle | **$28.07 \\pm 8.77\\%$** | $42.99 \\pm 16.91\\%$ | **$0.0030$** | $502.1 \\pm 369.0$ |
| **X11** | High $k_{d2}$, Low $k_{d3}$ | $(5.0, 5200.0, 0.35)$ | Sustained Cycle | $11.29 \\pm 4.02\\%$ | $10.46 \\pm 5.17\\%$ | **$0.0031$** | $202.0 \\pm 145.4$ |
| **X12** | Near-Critical Hopf | $(8.5, 4000.0, 0.85)$ | Sustained Cycle | **$7.71 \\pm 2.97\\%$** | $24.84 \\pm 14.97\\%$ | **$0.0028$** | $607.8 \\pm 500.0$ |

![Comprehensive 12-Regime Generalization Sweep](C:/Users/satya/.gemini/antigravity-ide/brain/fac6f262-a7ea-43cd-9cf8-6ed5750f1c06/mega_parameter_sweep_comprehensive.png)

*(Generated to [`mega_parameter_sweep_comprehensive.png`](file:///c:/Users/satya/Downloads/Van%20Der%20Pol/mega_parameter_sweep_comprehensive.png) and mirrored in the artifact directory).*
"""

for d in dst_dirs:
    fp = os.path.join(d, "biochemical_final_publication_manuscript_tables.md")
    if os.path.exists(fp):
        with open(fp, "r", encoding="utf-8") as f:
            c = f.read()
        marker = "## 6. Comprehensive Parameter Sweep"
        idx = c.find(marker)
        if idx != -1:
            new_c = c[:idx] + sec6_text
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new_c)
            print(f"Updated {fp}")
