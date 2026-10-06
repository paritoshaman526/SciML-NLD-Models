# Table 1: Global Multi-Seed Benchmark Performance (Unified Continuous SciML)

> **Evaluation Protocol & Audit Notice**:
> - **Architecture**: Unified Continuous Neural ODE (learned continuous vector field $\dot{\mathbf{y}} = \mathbf{f}_\theta(\mathbf{y}, \mathbf{p})$ with embedded RK4 integration and stoichiometric mass conservation). **All external gate / artificial Hopf-detection switching branches are removed**.
> - **Seeds**: 7 canonical independent seeds (`[42, 101, 777, 999, 1234, 2024, 5555]`).
> - **Basin-of-Attraction Audit ($X_1 - X_6$)**: Initialized from a perturbed state $\mathbf{y}_0 = \mathbf{y}^* + 15\%\,\delta m_1$ (with stoichiometric conservation strictly enforced) to test whether autonomous dynamics relax back to physical equilibrium. Prior claims of "0.000 mM drift" are discarded as artifacts of initializing directly on $y^*$.
> - **Bimodality Check**: Sarle's Bimodality Coefficient ($BC > 0.555$) and Coefficient of Variation ($CV = \sigma / \mu$).

---

## 1. Stable Focus Regimes ($X_1 - X_6$): Perturbed Basin-of-Attraction Test

*Evaluates autonomous relaxation from $+15\%$ initial perturbation ($\delta m_1 \in [42.3, 63.0]\,\text{mM}$) over 4,000 steps ($0.4\,\text{s}$).*

| Case | Parameters ($k_a, k_{d2}, k_{d3}$) | True Equilibrium $m_1^*$ | Initial Perturbation $\delta m_1$ | Terminal Attractor Offset (mM) <br> `Mean ± Std` | `[Median]` | `(Min – Max)` | Basin Recovery Rate (%) | Distribution & Audit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **X1** | `(11.0, 6000, 0.15)` | 285.4 mM | +42.8 mM | **24.27 ± 0.04 mM** | `[24.27 mM]` | (24.22 – 24.32) | **43.3%** | Unimodal ($CV=0.002$); persistent learned fixed-point shift |
| **X2** | `(12.0, 6000, 0.15)` | 312.1 mM | +46.8 mM | **13.28 ± 0.03 mM** | `[13.28 mM]` | (13.24 – 13.35) | **71.6%** | Unimodal ($CV=0.002$); strongest basin recovery |
| **X3** | `(13.0, 6000, 0.25)` | 281.9 mM | +42.3 mM | **36.65 ± 0.03 mM** | `[36.66 mM]` | (36.61 – 36.67) | **13.3%** | Unimodal ($CV=0.001$); weak contraction, large attractor offset |
| **X4** | `(10.5, 4600, 0.60)` | 283.0 mM | +42.5 mM | **18.09 ± 0.04 mM** | `[18.10 mM]` | (18.02 – 18.13) | **57.4%** | Unimodal ($CV=0.002$); moderate basin recovery |
| **X5** | `(12.0, 4500, 0.50)` | 359.1 mM | +53.9 mM | **19.58 ± 0.04 mM** | `[19.57 mM]` | (19.54 – 19.66) | **63.6%** | Unimodal ($CV=0.002$); moderate basin recovery |
| **X6** | `(14.0, 5000, 0.30)` | 419.9 mM | +63.0 mM | **21.74 ± 0.06 mM** | `[21.76 mM]` | (21.60 – 21.81) | **65.5%** | Unimodal ($CV=0.003$); moderate basin recovery |

---

## 2. Limit-Cycle Oscillation Regimes ($X_7 - X_{12}$): Peak-to-Peak Amplitude Error

| Case | Parameters ($k_a, k_{d2}, k_{d3}$) | True Limit Amplitude | Amplitude Error (%) <br> `Mean ± Std` | `[Median]` | `(Min – Max)` | Bimodality Audit ($BC, CV$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **X7** | `(1.0, 3000, 0.80)` | 721.2 mM | **81.78 ± 77.21 %** | `[27.24 %]` | (8.33 – 212.49 %) | $BC = 0.592, CV = 0.94$ ⚠️ | **Catastrophic Bimodal: 43% failure rate (collapse or explosion)** |
| **X8** | `(2.5, 3200, 0.80)` | 842.0 mM | **0.01 ± 0.01 %** | `[0.01 %]` | (0.00 – 0.03 %) | $BC = 0.441, CV = 0.98$ | **Passed (Sub-0.03% across all seeds)** |
| **X9** | `(3.5, 2600, 1.10)` | 302.5 mM | **0.08 ± 0.04 %** | `[0.07 %]` | (0.04 – 0.14 %) | $BC = 0.395, CV = 0.44$ | **Passed (Sub-0.15% across all seeds)** |
| **X10** | `(4.5, 3500, 1.30)` | 141.8 mM | **0.13 ± 0.04 %** | `[0.12 %]` | (0.08 – 0.20 %) | $BC = 0.334, CV = 0.31$ | **Passed (Sub-0.20% across all seeds)** |
| **X11** | `(5.0, 5200, 0.35)` | 812.7 mM | **0.10 ± 0.10 %** | `[0.08 %]` | (0.01 – 0.34 %) | $BC = 0.655, CV = 1.02$ | **Passed (Sub-0.35% across all seeds)** |
| **X12** | `(8.5, 4000, 0.85)` | 30.8 mM | **0.10 ± 0.02 %** | `[0.10 %]` | (0.06 – 0.15 %) | $BC = 0.339, CV = 0.24$ | **Passed (Sub-0.15% across all seeds)** |

---

## 3. Granular Per-Seed Breakdown (All 12 Regimes across 7 Seeds)

### A. Perturbed Basin Relaxation: Final State $m_1(T)$ (mM) for $X_1 - X_6$

| Seed | X1 (True: 285.4) | X2 (True: 312.1) | X3 (True: 281.9) | X4 (True: 283.0) | X5 (True: 359.1) | X6 (True: 419.9) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 261.1 mM (Err: 24.3) | 298.8 mM (Err: 13.3) | 318.6 mM (Err: 36.7) | 264.9 mM (Err: 18.1) | 339.5 mM (Err: 19.6) | 398.2 mM (Err: 21.8) |
| **101** | 261.1 mM (Err: 24.3) | 298.8 mM (Err: 13.3) | 318.5 mM (Err: 36.6) | 264.9 mM (Err: 18.1) | 339.5 mM (Err: 19.6) | 398.2 mM (Err: 21.7) |
| **777** | 261.1 mM (Err: 24.3) | 298.7 mM (Err: 13.4) | 318.5 mM (Err: 36.6) | 264.9 mM (Err: 18.1) | 339.4 mM (Err: 19.7) | 398.3 mM (Err: 21.6) |
| **999** | 261.1 mM (Err: 24.3) | 298.8 mM (Err: 13.3) | 318.6 mM (Err: 36.7) | 265.0 mM (Err: 18.1) | 339.5 mM (Err: 19.5) | 398.1 mM (Err: 21.8) |
| **1234** | 261.1 mM (Err: 24.3) | 298.8 mM (Err: 13.3) | 318.6 mM (Err: 36.7) | 264.9 mM (Err: 18.1) | 339.5 mM (Err: 19.6) | 398.2 mM (Err: 21.8) |
| **2024** | 261.2 mM (Err: 24.2) | 298.8 mM (Err: 13.2) | 318.5 mM (Err: 36.6) | 265.0 mM (Err: 18.0) | 339.5 mM (Err: 19.6) | 398.2 mM (Err: 21.8) |
| **5555** | 261.1 mM (Err: 24.3) | 298.8 mM (Err: 13.3) | 318.6 mM (Err: 36.7) | 264.9 mM (Err: 18.1) | 339.5 mM (Err: 19.6) | 398.1 mM (Err: 21.8) |

### B. Limit-Cycle Regimes: Amplitude Error (%) for $X_7 - X_{12}$

| Seed | X7 ($k_a=1.0$) | X8 ($k_a=2.5$) | X9 ($k_a=3.5$) | X10 ($k_a=4.5$) | X11 ($k_a=5.0$) | X12 ($k_a=8.5$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | **22.05%** | 0.03% | 0.06% | 0.09% | 0.08% | 0.06% |
| **101** | **177.42% (Exploded)** | 0.01% | 0.07% | 0.08% | 0.02% | 0.10% |
| **777** | **100.00% (Collapsed)**| 0.00% | 0.07% | 0.12% | 0.10% | 0.09% |
| **999** | **8.33%** | 0.01% | 0.11% | 0.18% | 0.06% | 0.12% |
| **1234** | **212.49% (Exploded)** | 0.00% | 0.04% | 0.12% | 0.01% | 0.10% |
| **2024** | **24.95%** | 0.03% | 0.12% | 0.15% | 0.10% | 0.10% |
| **5555** | **27.24%** | 0.00% | 0.14% | 0.20% | 0.34% | 0.15% |

---

## 4. Key Takeaways & Recommendations

1. **Stationarity vs Basin Convergence**:
   - The model learns contractive focus dynamics with tight inter-seed consistency ($CV \le 0.003$).
   - However, the autonomous learned fixed points have a persistent offset of $13.3\,\text{mM} - 36.7\,\text{mM}$ from the true physical equilibrium $y^*(p)$.
2. **Catastrophic Bimodality on Stiff Regime $X_7$**:
   - For $X_7$ ($k_a=1.0\,\text{s}^{-1}$), **3 out of 7 seeds ($43\%$) fail catastrophically** (1 collapses to a fixed point, 2 explode into runaway oscillations $>2000\,\text{mM}$).
   - The reported mean of $81.78\%$ is misleading; median performance is $27.24\%$.
3. **Stop Introducing New Variants — Consolidate Current Architecture**:
   - Standard limit cycles ($X_8 - X_{12}$) are unconditionally stable and accurate across all seeds ($<0.35\%$).
   - Future work must focus on resolving stiff relaxation kinetics ($X_7$) and fixed-point calibration without introducing unverified heuristic gating networks.
