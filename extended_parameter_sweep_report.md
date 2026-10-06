# Extended Parameter Sweep Across Deep Extrapolation ($k_a > 10$, Diverse $k_{d2}$, $k_{d3}$)

> **Experimental Overview**:
> - **Deep Extrapolation ($k_a > 10$)**: Evaluates system behavior well past the nominal training upper bound ($k_a = 8.0$) and beyond the Hopf bifurcation boundary ($k_a \approx 7.1 - 8.2$).
> - **Diverse $k_{d2}, k_{d3}$ Combinations**: Assesses robustness across high and low binding rates ($k_{d2} \in [3200, 5000]$) and activation rates ($k_{d3} \in [0.3, 1.3]$).
> - **Multi-Seed Replication**: All metrics computed as `Mean ± Std` across all 7 canonical seeds (`[42, 101, 2024, 777, 999, 1234, 5555]`).

### E1: Deep Extrapolation (ka=11.0, kd2=5000, kd3=0.3) — [Damped Stiff Extrapolation]
- **Ground Truth**: Period $T = 0.1948$ s, Target Metric $= 0.36$

| Model | Period Error (%) | Decay Match ($D_{\text{model}}/D_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 13.43 ± 10.65% | 1.772 ± 1.119 | 0.0031 ± 0.0001 |
| **FiLM FT (alpha=0)** | 13.99 ± 10.46% | 1.761 ± 1.146 | 0.0031 ± 0.0000 |
| **FiLM+Div** | 15.85 ± 11.65% | 2.069 ± 0.799 | 0.0031 ± 0.0001 |
| Projected Baseline | 21.28 ± 21.62% | 0.930 ± 1.186 | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 30.74 ± 38.76% | 0.432 ± 0.781 | 557.1 ± 529.7 |

---

### E2: Deep Extrapolation (ka=12.0, kd2=4500, kd3=0.5) — [Damped Stiff Extrapolation]
- **Ground Truth**: Period $T = 0.1542$ s, Target Metric $= 0.14$

| Model | Period Error (%) | Decay Match ($D_{\text{model}}/D_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 40.65 ± 17.00% | 8.540 ± 4.399 | 0.0031 ± 0.0001 |
| **FiLM FT (alpha=0)** | 80.42 ± 108.80% | 8.327 ± 4.319 | 0.0031 ± 0.0001 |
| **FiLM+Div** | 27.21 ± 3.74% | 8.056 ± 3.372 | 0.0031 ± 0.0000 |
| Projected Baseline | 17.50 ± 2.89% | 0.639 ± 1.060 | 0.0036 ± 0.0002 |
| Baseline MLP (Unconstrained) | 17.42 ± 3.57% | 0.426 ± 0.430 | 859.0 ± 675.8 |

---

### E3: Extreme Stiff Extrapolation (ka=14.0, kd2=5000, kd3=0.3) — [Damped Stiff Extrapolation]
- **Ground Truth**: Period $T = 0.1765$ s, Target Metric $= 0.06$

| Model | Period Error (%) | Decay Match ($D_{\text{model}}/D_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 64.88 ± 48.99% | 12.212 ± 4.947 | 0.0030 ± 0.0002 |
| **FiLM FT (alpha=0)** | 72.13 ± 51.77% | 11.994 ± 5.197 | 0.0029 ± 0.0001 |
| **FiLM+Div** | 34.21 ± 15.63% | 17.516 ± 4.695 | 0.0031 ± 0.0001 |
| Projected Baseline | 14.18 ± 3.94% | 1.315 ± 1.279 | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 13.96 ± 3.31% | 1.317 ± 1.208 | 809.4 ± 622.6 |

---

### E4: Transition Extrapolation (ka=10.5, kd2=4600, kd3=0.6) — [Damped Stiff Extrapolation]
- **Ground Truth**: Period $T = 0.1558$ s, Target Metric $= 0.30$

| Model | Period Error (%) | Decay Match ($D_{\text{model}}/D_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 33.24 ± 24.59% | 3.297 ± 1.507 | 0.0031 ± 0.0001 |
| **FiLM FT (alpha=0)** | 33.70 ± 24.22% | 3.343 ± 1.532 | 0.0031 ± 0.0001 |
| **FiLM+Div** | 12.72 ± 5.02% | 2.157 ± 1.285 | 0.0030 ± 0.0001 |
| Projected Baseline | 17.83 ± 5.89% | 0.188 ± 0.139 | 0.0036 ± 0.0002 |
| Baseline MLP (Unconstrained) | 18.70 ± 6.51% | 0.222 ± 0.157 | 458.9 ± 478.4 |

---

### E5: High kd2 Sustained Surge (ka=7.0, kd2=5000, kd3=0.4) — [Sustained Limit Cycle]
- **Ground Truth**: Period $T = 0.2445$ s, Target Metric $= 717.00$

| Model | Period Error (%) | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 16.22 ± 7.99% | 0.847 ± 0.320 | 0.0032 ± 0.0000 |
| **FiLM FT (alpha=0)** | 18.67 ± 9.59% | 0.722 ± 0.432 | 0.0031 ± 0.0000 |
| **FiLM+Div** | 23.36 ± 12.79% | 0.416 ± 0.434 | 0.0030 ± 0.0001 |
| Projected Baseline | 16.43 ± 5.94% | 1.081 ± 0.356 | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 37.25 ± 17.51% | 1.869 ± 0.782 | 291.9 ± 85.9 |

---

### E6: High kd3 Moderate Cycle (ka=4.5, kd2=3500, kd3=1.3) — [Sustained Limit Cycle]
- **Ground Truth**: Period $T = 0.1637$ s, Target Metric $= 140.67$

| Model | Period Error (%) | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 42.85 ± 15.77% | 2.235 ± 0.536 | 0.0030 ± 0.0000 |
| **FiLM FT (alpha=0)** | 44.61 ± 22.45% | 2.278 ± 0.561 | 0.0030 ± 0.0000 |
| **FiLM+Div** | 42.72 ± 31.76% | 2.657 ± 0.824 | 0.0030 ± 0.0001 |
| Projected Baseline | 12.55 ± 9.16% | 2.983 ± 1.519 | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 10.98 ± 8.48% | 4.997 ± 4.408 | 528.3 ± 392.1 |

---

### E7: Low ka Relaxation Cycle (ka=2.5, kd2=3200, kd3=0.8) — [Sustained Limit Cycle]
- **Ground Truth**: Period $T = 0.3545$ s, Target Metric $= 869.64$

| Model | Period Error (%) | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) | Mass Drift (mM) |
| :--- | :---: | :---: | :---: |
| **FiLM Pure** | 5.54 ± 4.64% | 1.043 ± 0.037 | 0.0030 ± 0.0001 |
| **FiLM FT (alpha=0)** | 5.85 ± 4.26% | 1.042 ± 0.034 | 0.0029 ± 0.0001 |
| **FiLM+Div** | 33.43 ± 12.90% | 0.667 ± 0.330 | 0.0031 ± 0.0001 |
| Projected Baseline | 25.51 ± 9.01% | 0.761 ± 0.304 | 0.0036 ± 0.0001 |
| Baseline MLP (Unconstrained) | 35.46 ± 16.91% | 0.780 ± 0.300 | 474.7 ± 394.9 |

---

