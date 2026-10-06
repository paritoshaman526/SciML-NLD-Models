# Table 1: Global Benchmark Performance Across 7 Seeds

> **Evaluation Protocol**:
> - **Seeds**: 7 independent seeds (`[42, 101, 2024, 777, 999, 1234, 5555]`)
> - **Statistics**: Reported as `Mean ± Std`, `[Median]`, and `(Min – Max)` across all 7 seeds
> - **Regime Classification**: Case A is **In-Sample**; Cases B & C are **Sustained Limit Cycles**; Case D is a **Damped Stiff Regime** (evaluated by decay ratio $D_{\text{model}} / D_{\text{true}}$)

### Case A: Nominal Slide [In-Sample]

| Model | Period Error (%) <br> `Mean ± Std [Med] (Min-Max)` | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) <br> `Mean ± Std [Med] (Min-Max)` | Mass Drift (mM) <br> `Mean ± Std [Med] (Min-Max)` |
| :--- | :---: | :---: | :---: |
| Baseline (Standard) | 6.00 ± 4.28% <br> `[5.45%]` <br> (1.62 – 12.40%) | 1.187 ± 0.400 <br> `[1.229]` <br> (0.402 – 1.791) | 155.8 ± 74.8 <br> `[174.4]` <br> (42.1 – 261.0) |
| Baseline FT (alpha_div=0 Control) | 8.27 ± 5.55% <br> `[5.00%]` <br> (2.17 – 18.74%) | 1.631 ± 0.468 <br> `[1.602]` <br> (0.970 – 2.307) | 199.5 ± 75.7 <br> `[203.9]` <br> (84.5 – 335.1) |
| Baseline Div-Only | 4.65 ± 2.36% <br> `[5.09%]` <br> (1.80 – 8.90%) | 1.141 ± 0.238 <br> `[1.111]` <br> (0.741 – 1.609) | 1130.2 ± 520.6 <br> `[1156.2]` <br> (465.0 – 1993.7) |
| **FiLM Pure** | 5.70 ± 4.79% <br> `[3.10%]` <br> (0.03 – 13.25%) | 1.188 ± 0.209 <br> `[1.134]` <br> (0.927 – 1.521) | 0.0028 ± 0.0002 <br> `[0.0029]` <br> (0.0025 – 0.0031) |
| **FiLM FT (alpha_div=0 Control)** | 5.69 ± 4.81% <br> `[3.14%]` <br> (0.13 – 13.12%) | 1.194 ± 0.220 <br> `[1.127]` <br> (0.946 – 1.541) | 0.0029 ± 0.0002 <br> `[0.0030]` <br> (0.0026 – 0.0031) |
| **FiLM+Div** | 2.72 ± 2.86% <br> `[1.48%]` <br> (0.06 – 8.69%) | 0.633 ± 0.289 <br> `[0.750]` <br> (0.135 – 0.937) | 0.0026 ± 0.0002 <br> `[0.0025]` <br> (0.0025 – 0.0030) |

---

### Case B: Slower High-Amplitude [Interpolation]

| Model | Period Error (%) <br> `Mean ± Std [Med] (Min-Max)` | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) <br> `Mean ± Std [Med] (Min-Max)` | Mass Drift (mM) <br> `Mean ± Std [Med] (Min-Max)` |
| :--- | :---: | :---: | :---: |
| Baseline (Standard) | 43.59 ± 39.41% <br> `[29.75%]` <br> (18.59 – 138.89%) | 1.094 ± 0.693 <br> `[1.081]` <br> (0.297 – 2.567) | 329.2 ± 297.5 <br> `[175.9]` <br> (37.2 – 948.8) |
| Baseline FT (alpha_div=0 Control) | 30.99 ± 9.78% <br> `[35.52%]` <br> (19.24 – 43.05%) | 1.101 ± 0.457 <br> `[1.215]` <br> (0.390 – 1.773) | 157.1 ± 61.3 <br> `[129.3]` <br> (96.0 – 271.0) |
| Baseline Div-Only | 36.32 ± 15.85% <br> `[40.70%]` <br> (2.33 – 50.81%) | 0.504 ± 0.248 <br> `[0.357]` <br> (0.291 – 1.056) | 2114.9 ± 1180.3 <br> `[1585.0]` <br> (831.2 – 4316.2) |
| **FiLM Pure** | 2.95 ± 3.76% <br> `[1.70%]` <br> (0.36 – 11.82%) | 0.962 ± 0.036 <br> `[0.971]` <br> (0.896 – 1.009) | 0.0031 ± 0.0000 <br> `[0.0031]` <br> (0.0031 – 0.0031) |
| **FiLM FT (alpha_div=0 Control)** | 2.86 ± 3.73% <br> `[1.71%]` <br> (0.11 – 11.77%) | 0.965 ± 0.034 <br> `[0.970]` <br> (0.907 – 1.019) | 0.0030 ± 0.0000 <br> `[0.0031]` <br> (0.0030 – 0.0031) |
| **FiLM+Div** | 38.58 ± 13.41% <br> `[39.18%]` <br> (11.88 – 56.01%) | 0.659 ± 0.326 <br> `[0.772]` <br> (0.123 – 1.020) | 0.0030 ± 0.0000 <br> `[0.0030]` <br> (0.0030 – 0.0031) |

---

### Case C: Giant Amplitude Surge [Extrapolation]

| Model | Period Error (%) <br> `Mean ± Std [Med] (Min-Max)` | Amp Ratio ($A_{\text{model}}/A_{\text{true}}$) <br> `Mean ± Std [Med] (Min-Max)` | Mass Drift (mM) <br> `Mean ± Std [Med] (Min-Max)` |
| :--- | :---: | :---: | :---: |
| Baseline (Standard) | 40.28 ± 36.05% <br> `[33.90%]` <br> (2.05 – 122.86%) | 1.349 ± 0.230 <br> `[1.416]` <br> (1.026 – 1.766) | 210.0 ± 102.5 <br> `[165.1]` <br> (59.2 – 350.8) |
| Baseline FT (alpha_div=0 Control) | 47.34 ± 39.68% <br> `[36.83%]` <br> (1.32 – 111.23%) | 1.413 ± 0.252 <br> `[1.411]` <br> (1.041 – 1.766) | 184.9 ± 109.0 <br> `[119.5]` <br> (66.3 – 327.5) |
| Baseline Div-Only | 14.19 ± 14.06% <br> `[5.16%]` <br> (0.64 – 34.48%) | 0.574 ± 0.239 <br> `[0.658]` <br> (0.098 – 0.797) | 1468.6 ± 670.8 <br> `[1152.9]` <br> (865.7 – 2592.3) |
| **FiLM Pure** | 4.64 ± 2.82% <br> `[3.44%]` <br> (0.42 – 8.43%) | 1.023 ± 0.030 <br> `[1.028]` <br> (0.984 – 1.068) | 0.0031 ± 0.0001 <br> `[0.0032]` <br> (0.0030 – 0.0032) |
| **FiLM FT (alpha_div=0 Control)** | 5.42 ± 3.08% <br> `[3.98%]` <br> (1.60 – 9.47%) | 1.027 ± 0.029 <br> `[1.030]` <br> (0.988 – 1.075) | 0.0032 ± 0.0000 <br> `[0.0032]` <br> (0.0031 – 0.0032) |
| **FiLM+Div** | 26.24 ± 13.49% <br> `[30.48%]` <br> (2.26 – 41.82%) | 0.388 ± 0.330 <br> `[0.194]` <br> (0.109 – 0.988) | 0.0031 ± 0.0001 <br> `[0.0031]` <br> (0.0030 – 0.0032) |

---

### Case D: Damped Stiff Regime [Extrapolation]

| Model | Period Error (%) <br> `Mean ± Std [Med] (Min-Max)` | Decay Ratio ($D_{\text{model}}/D_{\text{true}}$) <br> `Mean ± Std [Med] (Min-Max)` | Mass Drift (mM) <br> `Mean ± Std [Med] (Min-Max)` |
| :--- | :---: | :---: | :---: |
| Baseline (Standard) | 31.05 ± 43.65% <br> `[3.94%]` <br> (0.11 – 100.00%) | 0.789 ± 0.392 <br> `[0.577]` <br> (0.249 – 1.273) | 805.8 ± 503.1 <br> `[627.8]` <br> (96.5 – 1659.8) |
| Baseline FT (alpha_div=0 Control) | 18.12 ± 33.62% <br> `[5.45%]` <br> (0.24 – 100.00%) | 0.770 ± 0.705 <br> `[0.543]` <br> (0.113 – 2.273) | 790.1 ± 409.4 <br> `[731.3]` <br> (259.6 – 1289.4) |
| Baseline Div-Only | 9.99 ± 8.29% <br> `[7.29%]` <br> (0.62 – 23.34%) | 0.263 ± 0.206 <br> `[0.185]` <br> (0.068 – 0.713) | 2958.4 ± 2054.2 <br> `[3305.6]` <br> (237.3 – 6195.9) |
| **FiLM Pure** | 2.08 ± 1.46% <br> `[1.54%]` <br> (0.50 – 4.90%) | 0.253 ± 0.218 <br> `[0.159]` <br> (0.085 – 0.757) | 0.0025 ± 0.0000 <br> `[0.0025]` <br> (0.0025 – 0.0026) |
| **FiLM FT (alpha_div=0 Control)** | 1.76 ± 1.70% <br> `[1.41%]` <br> (0.06 – 5.51%) | 0.313 ± 0.268 <br> `[0.182]` <br> (0.092 – 0.746) | 0.0026 ± 0.0000 <br> `[0.0026]` <br> (0.0025 – 0.0026) |
| **FiLM+Div** | 19.05 ± 33.22% <br> `[8.40%]` <br> (0.62 – 100.00%) | 0.560 ± 0.340 <br> `[0.477]` <br> (0.124 – 1.273) | 0.0026 ± 0.0000 <br> `[0.0026]` <br> (0.0026 – 0.0026) |

---

