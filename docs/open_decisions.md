# Open Decisions Log (SC452 Group 5)

## OD1 — Synthetic ER Connectivity Parameter (p=0.3)
- **Status**: Provisional default adopted (`p = 0.3`).
- **Context**: In the handwritten TA notes, the ER parameters list `p = 0.1, 0.2, 0.3?`. The third value was partially cut off.
- **Action**: Adopted `0.3` provisionally. Scheduled for confirmation with TA.

## OD2 — SocioPatterns Face-to-Face Dataset Choice
- **Status**: Resolved (`InVS15` workplace contact dataset, 2015 2nd deployment).
- **Context**: TA required a verified SocioPatterns dataset. InVS15 (~217–232 individuals, 20s interval resolution) has clean temporal face-to-face contacts.
- **Action**: Official source verified at `http://www.sociopatterns.org/assets/data/workplace_InVS15_tij.dat.gz`. Download and validation incorporated into pipeline.

## OD3 — Directionality of Temporal Interaction Logs (email-Eu & CollegeMsg)
- **Status**: Events preserved as directed in event tables; static network aggregated as undirected for primary spectral threshold analysis.
- **Context**: Epidemic spreading along contact opportunities typically treats communication contact as a bidirectional opportunity or symmetric interaction unless strictly directed propagation is modeled.
- **Action**: Primary static graph is undirected. Directed statistics and directed adjacency are preserved in event tables and optional analysis.

## OD4 — Static Adjacency Weighting: Binary vs. Contact-Count-Weighted
- **Status**: Dual implementation.
- **Context**: Aggregation can be unweighted binary ($A_{ij} \in \{0, 1\}$) or weighted by total contact count ($W_{ij} = \sum_t E_{ij}(t)$).
- **Action**: Both representations are generated and saved. Primary spectral threshold analysis compares both against temporal process dynamics.

## OD5 — Simulation Repetition Count ($R$)
- **Status**: Deferred to Sprint 2 pilot study.
- **Context**: Estimating threshold variance requires sufficient Monte Carlo realizations $R$. Provisional default $R=200$ set in `default.yaml`.

## OD6 — Multiprocessing / Parallelization Strategy
- **Status**: `joblib` with configurable `n_jobs` (default `-1`).

## OD7 — N4 Null Model Scope
- **Status**: Deferred beyond initial baseline null models (N1 timeline shuffling, N2/N3 synthetic ER/BA topology with empirical timestamps).

## OD8 — High-Complexity Centrality on Large Datasets
- **Status**: Betweenness centrality excluded on large networks ($N > 2000$) to avoid excessive compute during rapid test cycles; degree and eigenvector/k-core retained.
