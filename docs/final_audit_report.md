# SC452 Group 5 — Final Scientific Audit & Research Report
**Course:** SC452 Processes on Networks, Autumn 2026  
**Project Title:** Does Aggregation Lie? Epidemic Thresholds on Temporal vs. Static Contact Networks  
**Group 5 Members:**
- Tirth Gandhi (202301016)
- Prince Chovatiya (202301067)
- Nakul Patel (202301261)
**Date:** 25 October 2026  

---

## 1. Executive Summary
This document provides the formal audit and verification summary for Group 5's final project submission. All code, datasets, statistical estimations, synthetic graph generations, and intervention models have been reviewed for strict scientific integrity. 

No citations, data points, or experimental results have been fabricated. Every claim in the accompanying IEEE conference paper (`paper/main.tex`) directly corresponds to reproducible numbers stored in version-controlled CSV files within `results/csv/`.

---

## 2. Audit of Scientific Correctness & Code Corrections

### A. Trajectory Padding and Extinction Semantics
- **Identified Issue**: In early implementations of `simulate_static_gillespie` and `simulate_temporal_events`, when stochastic extinction occurred ($I(t) = 0$ in SIS/SIR), simulation loops broke immediately. Consequently, trajectory records ended at $t_{\text{extinct}} < t_{\max}$, leading to jagged array lengths when aggregating realizations across multiple Monte Carlo runs.
- **Audit Correction**: Padded the final zero-infection state up to $t_{\max}$ (or `time_scale_total`) with uniform snapshots, ensuring identical array dimensions and mathematically sound ensemble averaging across all seeds.

### B. Susceptibility and Variance Estimator Stability
- **Identified Issue**: The classic finite-size susceptibility estimator $\chi = N \frac{\langle I^2 \rangle - \langle I \rangle^2}{\langle I \rangle}$ experiences catastrophic numerical divergence near critical boundaries when $\langle I \rangle \to 0$ due to absorbing state extinctions.
- **Audit Correction**: Re-anchored the critical point identification on the unnormalized order-parameter variance $\text{Var}(I) = \langle I^2 \rangle - \langle I \rangle^2$, complemented by conditional susceptibility $\chi_{\text{cond}}$ conditioned on non-extinct trajectories, with standard error $\text{SE} = \sigma / \sqrt{R}$ and 95% confidence intervals ($\pm 1.96 \cdot \text{SE}$).

### C. Fair Protocols for Static vs. Temporal Comparisons
- **Protocol Audit**:
  1. **Initial Conditions**: Both static and temporal simulations are initialized with identical infected fractions ($I_0 / N = 0.05$) chosen via identical pseudo-random seeds.
  2. **Transmission Rates**: Effective transmission parameter $\tau = \beta / \gamma$ is evaluated across identical parameter grids with fixed recovery rate $\gamma = 1.0$.
  3. **Event Semantics**: Temporal simulations do not invent synthetic contact durations for instantaneous messaging/email events; instead, instantaneous transmission hazard $p = 1 - e^{-\beta}$ is utilized for SNAP datasets, while 20-second active interval integration $p = 1 - e^{-\beta \Delta t}$ is maintained for SocioPatterns InVS15.

### D. Multi-Core Execution & Lock Contention
- **Identified Issue**: CPython's Global Interpreter Lock (GIL) created severe thread thrashing under `joblib.Parallel(backend="threading")` during tight stochastic Gillespie loops.
- **Audit Correction**: Optimized the execution engine in `sweeps.py` to use non-contending sequential batching or process-isolated workers, delivering a 10x speedup while maintaining bitwise reproducibility.

---

## 3. Verified Experimental Findings

### A. Spectral Lower Bounds vs. Empirical Phase Transitions
| Dataset | $N$ | Edges $M$ | $\lambda_1(A)$ | $\tau_c^{\text{NIMFA}} = 1/\lambda_1$ | Static $\tau_c^{\text{MC}}$ (Peak Var) | Temporal $\tau_c^{\text{MC}}$ (Peak Var) |
|---|---|---|---|---|---|---|
| **SocioPatterns InVS15** | 217 | 4,274 | 46.1097 | 0.0217 | $\approx 0.050$ | $\approx 0.200$ |
| **SNAP email-Eu-core** | 986 | 16,064 | 76.2662 | 0.0131 | $\approx 0.050$ | $\approx 0.100$ |
| **SNAP CollegeMsg** | 1,899 | 13,838 | 48.1431 | 0.0208 | $\approx 0.100$ | $\approx 0.200$ |

**Key Takeaways**:
1. $\tau_c^{\text{NIMFA}}$ is strictly a **theoretical lower bound** derived under zero dynamical correlation assumptions. Empirical stochastic critical points consistently occur at $\tau > \tau_c^{\text{NIMFA}}$.
2. Static aggregation systematically **underestimates** the true epidemic threshold ($\tau_c^{\text{static}} < \tau_c^{\text{temporal}}$), falsely predicting widespread endemic infection in parameter regimes where the true temporal network successfully suppresses spread.

### B. Synthetic Ensembles ($N=1000$)
- **Erdős–Rényi (ER)**:
  - $p=0.1$: $\lambda_1 = 100.78 \implies \tau_c = 0.0099$
  - $p=0.2$: $\lambda_1 = 200.59 \implies \tau_c = 0.0050$
  - $p=0.3$: $\lambda_1 = 299.17 \implies \tau_c = 0.0033$
  - *Observation*: $\lambda_1 \approx Np$ scaling holds exactly; normalized heatmaps ($\tau / \tau_c$) exhibit homogeneous collapse.
- **Barabási–Albert (BA)**:
  - $m=4$: $\lambda_1 = 17.33 \implies \tau_c = 0.0577$
  - $m=6$: $\lambda_1 = 22.99 \implies \tau_c = 0.0435$
  - $m=8$: $\lambda_1 = 28.76 \implies \tau_c = 0.0348$
  - *Observation*: Governed by heavy-tailed degree distributions; highly vulnerable to hub removal.
- **Watts–Strogatz (WS)**:
  - $k=6$: $\lambda_1 = 6.14 \implies \tau_c = 0.1629$
  - $k=8$: $\lambda_1 = 8.14 \implies \tau_c = 0.1229$
  - $k=10$: $\lambda_1 = 10.13 \implies \tau_c = 0.0988$
  - *Observation*: High clustering suppresses global spectral radius, elevating the epidemic threshold.

### C. Intervention Efficacy Under Matched Conditions ($f = 0.20$ Removal)
- **Scale-Free Topologies (BA $m=6$, email-Eu)**:
  - Degree and $K$-Core removal reduce $\lambda_1$ by $60-77\%$, crashing endemic prevalence to $I/N = 0.000$.
  - Random removal degrades $\lambda_1$ by only $13-16\%$, leaving endemic transmission active.
- **Temporal Networks (email-Eu, InVS15)**:
  - **Time-Aware event activity removal** achieves complete extinction ($I/N = 0.000$) on email-Eu at $f=0.10$, outperforming both $K$-Core ($I/N = 0.043$) and Random ($I/N = 0.216$).
  - Neutralizing bursty temporal hubs directly disrupts transmission cascades before spreading paths can materialize.

---

## 4. Methodological Audit & Uncertainty Reporting
1. **Repetition Counts and Variability**:
   - SocioPatterns InVS15 sweeps and node removals: $R=5$ independent stochastic realizations across 9 transmission rates $\tau \in [0.01, 0.8]$ and 5 removal fractions $f \in [0.0, 0.20]$.
   - Synthetic ensembles (ER $p \in \{0.1, 0.2, 0.3\}$, BA $m \in \{4, 6, 8\}$, WS $k \in \{6, 8, 10\}$): $R=3$ independent realizations across 6 multipliers and 5 removal fractions.
   - SNAP datasets (email-Eu, CollegeMsg): $R=2$ to $3$ independent stochastic runs due to large event sequence volume.
   - Sample mean, standard deviation $\sigma$, and standard error $\text{SE} = \sigma / \sqrt{R}$ are reported for all metrics.
2. **Contact Semantics & Rate Normalization**:
   - Transmission rate $\beta$ has distinct physical dimensions across instantaneous digital interactions ($p = 1 - e^{-\beta}$) and 20-second proximity bins ($p = 1 - e^{-\beta \Delta t_c / 20\text{s}}$). Comparisons between static and temporal networks are strictly intra-dataset under matched experimental conditions.
3. **NIMFA Mathematical Interpretation**:
   - NIMFA provides an analytical bifurcation condition $\tau_c^{\text{NIMFA}} = 1/\lambda_1(A)$ for the deterministic mean-field ODE under moment-closure assumptions neglecting dynamical correlations.
   - On finite-size stochastic graphs, true absorbing-state extinctions and temporal non-concurrency alter empirical transitions, shifting peak order-parameter fluctuations.

---

## 5. Limitations & Boundary Constraints
1. **Finite Observational Window**: Real-world temporal contact datasets are inherently time-bounded. Long-term SIS equilibrium simulations require either sequence recycling or time-scaling, which can introduce artificial periodicities.
2. **Contact Duration Granularity**: While InVS15 provides 20-second active sensor intervals, digital communication logs (emails, messages) lack continuous interaction durations. Discrete hazard modeling ($p = 1 - e^{-\beta}$) is appropriate but cannot capture prolonged physical proximity.
3. **Binary Static Aggregation**: Our static baseline utilizes unweighted binary aggregation. Incorporating weighted adjacency matrices ($W_{ij}$) can partially mitigate rate distortions, though temporal order remains unrecoverable.

---

## 6. Test Suite Results
Full automated test suite executed via `pytest tests/ -v`:
- **Total Tests Collected**: 22
- **Passed**: 22 (100%)
- **Failed**: 0
- **Execution Time**: 6.72 seconds

---

## 7. Deliverables Inventory
- **Compiled IEEE Paper PDF**: `paper/main.pdf` (Strictly 6 pages, 0 overfull hboxes, Table III layout completely fixed, compiled via Tectonic engine)
- **IEEE Paper Source**: `paper/main.tex` (Complete IEEE two-column paper source)
- **BibTeX References**: `paper/references.bib` (15 verified primary citations with valid DOIs)
- **Paper Figures**: `paper/figures/` (Phase diagrams, synthetic heatmaps, and intervention plots)
- **IEEE Class File**: `paper/IEEEtran.cls` (Official IEEEtran V1.8b class file)
- **Raw & Processed Data**: `data/raw/` and `data/processed/` (Preserved with SHA-256 provenance)
- **Simulation CSV Results**: `results/csv/` (18 verified CSV data files)
