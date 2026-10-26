# Does Aggregation Lie? Epidemic Thresholds on Temporal vs. Static Contact Networks

**SC452 — Processes on Networks (Group 5)**
- Tirth Gandhi (202301016)
- Prince Chovatiya (202301067)
- Nakul Patel (202301261)

---

## 1. Project Overview & Methodology

This project investigates whether static network aggregation biases epidemic threshold estimations ($\tau_c$) and spreading dynamics compared to time-resolved temporal contact sequences.

### Complete Project Features:
- **Modular Python Architecture**: Full package structure with reproducible seeding, YAML configuration, and logging.
- **Empirical Dataset Pipeline**: Automated acquisition with SHA-256 provenance records and validation for:
  1. SNAP email-Eu-core-temporal ($N=986$, $M=16,064$)
  2. SNAP CollegeMsg ($N=1,899$, $M=13,838$)
  3. SocioPatterns InVS15 ($N=217$, $M=4,274$)
- **Synthetic Ensembles**: Reproducible generation of all 9 TA configurations:
  - Erdős–Rényi: $p \in \{0.1, 0.2, 0.3\}$
  - Barabási–Albert: $m \in \{4, 6, 8\}$
  - Watts–Strogatz: $k \in \{6, 8, 10\}$ ($p_{\text{rewire}}=0.05$)
- **Epidemic Dynamics**:
  - **Static Aggregated Networks**: Continuous-time Gillespie Stochastic Simulation Algorithm (SSA) with $O(1)$ active stub indexing.
  - **Temporal Sequences**: Event-driven simulation with exact inter-event Poisson recovery and empirical contact interval semantics.
  - **NIMFA Reference**: Deterministic N-Intertwined Mean-Field Approximation ODE integration and spectral threshold lower bound $\tau_c^{NIMFA} = 1/\lambda_1(A)$.
- **Intervention Analysis**: Matched node-removal strategies across Degree, $K$-Core, Time-Aware (interaction frequency), and Uniform Random removal.
- **Analysis & Outputs**: Tidy CSVs, phase transition diagrams, epidemic trajectory curves, outbreak distributions, heatmaps, and IEEE paper manuscript.

---

## 2. Repository Layout

```
.
├── config/
│   ├── default.yaml            # Master seeds, logging, paths
│   ├── synthetic.yaml          # All 9 TA-specified synthetic parameters
│   ├── real_datasets.yaml      # URLs, mirrors, format semantics
│   └── epidemic.yaml           # SIS/SIR sweep grids, parameters, recovery models
├── data/
│   ├── raw/                    # Immutable raw logs and provenance.json records
│   ├── processed/              # Clean event tables, node mappings, validation reports
│   └── synthetic/              # Generated synthetic graphs and manifests
├── docs/
│   ├── open_decisions.md       # Methodological choices and justifications
│   └── final_audit_report.md   # Comprehensive scientific audit and findings
├── figures/
│   ├── epidemic_curves/        # Temporal trajectories I(t)/N over time
│   ├── heatmaps/               # Parameter sweep heatmaps for ER, BA, WS
│   ├── interventions/          # Targeted node removal comparison curves
│   ├── outbreak_distributions/ # Attack rate boxplots for SIR
│   └── phase_diagrams/         # Static vs temporal phase transition curves
├── paper/
│   ├── IEEEtran.cls            # Official IEEE Transactions/Conference LaTeX class
│   ├── main.tex                # 6-page IEEE two-column research paper
│   ├── references.bib          # Verified BibTeX citations with DOIs
│   └── figures/                # Self-contained figures for compilation
├── results/
│   ├── csv/                    # 18 verified tidy CSV summary and run tables
│   └── sprint2_experiment_report.md # Synthesis report and threshold analysis
├── scripts/
│   ├── run_pipeline.py         # Data download, preprocessing, and synthetic generation
│   ├── run_experiments.py      # Real datasets SIS/SIR sweeps & interventions
│   └── run_synthetic_experiments.py # Synthetic graph sweeps & heatmaps
├── src/spm/
│   ├── data/                   # Acquisition, validation, and preprocessing
│   ├── graphs/                 # Synthetic models and spectral eigensolvers
│   ├── interventions/          # Degree, k-core, time-aware, and random node removal
│   ├── models/                 # Gillespie SSA, temporal simulator, NIMFA ODE, plotting
│   └── utils/                  # Seeding, configuration, and logging
└── tests/
    ├── unit/                   # Spectral analytical checks, parsing, epidemic transitions
    └── smoke/                  # Pipeline end-to-end smoke tests
```

---

## 3. Installation & Setup

```bash
# Clone or navigate to the workspace
cd "c:/Users/Prince Chovatiya/OneDrive/Desktop/Process on networks project"

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 4. Execution Commands

### Run Full Test Suite (22 Tests Passing)
```bash
pytest tests/ -v
```

### 1. Data Pipeline (Acquisition & Preprocessing)
```bash
python scripts/run_pipeline.py
```

### 2. Empirical Epidemic Experiments & Interventions
```bash
# Smoke test (quick execution across all 3 datasets):
python scripts/run_experiments.py --smoke

# Full run on specific dataset:
python scripts/run_experiments.py --dataset sociopatterns_invs15

# Full run across all empirical datasets:
python scripts/run_experiments.py
```

### 3. Synthetic Network Experiments & Heatmaps
```bash
# Smoke test across all 9 configurations:
python scripts/run_synthetic_experiments.py --smoke

# Full synthetic sweeps and heatmaps:
python scripts/run_synthetic_experiments.py
```

### 4. Compiling the IEEE Research Paper
The complete paper is compiled at `paper/main.pdf` (strict 6-page IEEE format). To recompile using the included Tectonic engine or pdflatex:
```bash
# Using local Tectonic engine:
cd paper
..\tools\tectonic.exe main.tex

# Or using standard pdflatex / Overleaf:
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
*The `paper/` directory is fully self-contained with its figures, bibliography, and `IEEEtran.cls`.*

---

## 5. Methodological Invariants & Temporal Semantics

1. **Static vs. Stochastic Thresholds**:
   - The NIMFA bound $\tau_c^{NIMFA} = 1/\lambda_1(A)$ is an analytical lower bound derived from mean-field theory under static edge assumptions.
   - Stochastic thresholds are evaluated empirically via the peak of the order-parameter variance $\text{Var}(I) = \langle I^2 \rangle - \langle I \rangle^2$.
2. **Instantaneous Contact Semantics (email-Eu, CollegeMsg)**:
   - Events represent discrete communication points without physical duration. Transmission probability per contact is $p = 1 - e^{-\beta}$. No synthetic duration is invented.
3. **Interval Contact Semantics (SocioPatterns InVS15)**:
   - Active face-to-face contacts have 20-second interval duration. Transmission hazard is integrated over active contact intervals: $p = 1 - e^{-\beta \cdot (\Delta t / 20.0)}$.
4. **Inter-event Poisson Recovery**:
   - Between successive contact events $\Delta t$, infected individuals recover independently with continuous-time Poisson probability $p_{rec} = 1 - e^{-\gamma \Delta t}$.

