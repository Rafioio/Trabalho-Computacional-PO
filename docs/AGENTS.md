# AGENTS.md — SouBuz: Otimização de Pontos de Parada de Ônibus

## Project Overview

Operations Research project solving the optimal placement of bus stops in Belo Horizonte, Brazil. The model balances walk access vs. commercial speed — minimizing social cost, maximizing technical feasibility, minimizing infrastructure cost, and penalizing excessive spacing violations.

**Keywords:** Operations Research, Mixed-Integer Linear Programming, Gurobi, Urban Transit, Facility Location

## Repository Structure

```
./
├── README.md                    # Project documentation
├── src/
│   ├── data/
│   │   ├── dados.dat            # Original canonical input (OPL format)
│   │   ├── dados.json           # Converted JSON (fast load)
│   │   └── loader.py            # .dat parser + .json I/O
│   ├── model/
│   │   ├── domains.py           # Sparse domains (a_domain, S_Indices)
│   │   ├── variables.py         # Decision variables (x, x_k, a, Cap, Cad, s_k)
│   │   ├── objective.py         # Weighted-sum objective: f1–f4
│   │   ├── constraints.py       # 8 constraint groups (C1–C8)
│   │   └── solver.py            # Model assembly + optimize
│   ├── scripts/
│   │   └── map_viewer.py        # Visualization (map, density, comparison)
│   ├── utils/
│   │   ├── generate_data.py     # Realistic synthetic data generator
│   │   ├── validator.py         # Pre-solve data consistency checks
│   │   ├── weight_normalizer.py # Utopia/anti-utopia payoff table
│   │   └── export_solution.py   # Solver results → solucao.json
│   └── run.py                   # Entry point
├── docs/
│   ├── AGENTS.md                # This file
│   ├── DATA.md                  # Data format reference
│   ├── Modelagem_PO.pdf         # Mathematical formulation (LaTeX)
│   └── WEIGHT_NORMALIZATION.md  # Weight normalization guide
└── .gitignore
```

## Tech Stack

| Component | Technology | License |
|-----------|-----------|---------|
| Solver | Gurobi via `gurobipy` | Proprietary (Gurobi) |
| Runtime | Python 3.10+ | PSF |
| Visualization | matplotlib, numpy | BSD |

**No** `requirements.txt` or `pyproject.toml` yet.

## How to Run

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run solver
python -m src.run                                    # default (dados.json or dados.dat)
python -m src.run --data dados_generated.json        # synthetic data
python -m src.run --data dados_generated.json --normalize-weights
python src/utils/generate_data.py                    # generate synthetic data
python src/scripts/map_viewer.py                     # visualize
python src/scripts/map_viewer.py cenario_01.json     # specific scenario
```

## Model Formulation

### Dimensions
- `NumN` = nodes (stops)
- `NumK` = routes
- `NumQ` = demand zones
- `C` = centroids (⊂ N)
- `T` = N \ C (candidate stops)

### Decision Variables

| Variable | Type | Domain | Meaning |
|----------|------|--------|---------|
| `x[n]` | Binary | n ∈ N | Stop n is active |
| `x_k[n,k]` | Binary | ⟨n,k⟩ ∈ I | Route k uses stop n |
| `a[q,n,k]` | Continuous [0,1] | ⟨q,n,k⟩ ∈ a_domain | Fraction of demand q assigned to stop n on route k |
| `Cap[k]` | Integer ≥0 | k ∈ K | Capacity allocated to route k |
| `Cad` | Integer ≥0 | — | Additional system capacity beyond Capt |
| `s_k[k,idx]` | Continuous ≥0 | ⟨k,idx⟩ ∈ S_Indices | Slack for max-spacing constraint |

### Objective Function (Weighted Sum, Minimize)

```
W1 × f1  +  W2 × f2  +  W3 × f3  +  W4 × f4
```

| Component | Description |
|-----------|-------------|
| `f1` | Social cost: sum(de[q] × (walk_dist + unserved_penalty × P)) |
| `f2` | Technical feasibility: sum((1/w[n]) × x_k) |
| `f3` | Infrastructure cost: ω × Cad + sum(x[n] for n ∈ T) |
| `f4` | Spacing penalty: sum(s_k) |

### Constraints (8)

1. **Infrastructure activation** — `x_k[n,k] ≤ x[n]` for all ⟨n,k⟩ ∈ I
2. **Boarding feasibility** — `a[q,n,k] ≤ x_k[n,k]` for all ⟨q,n,k⟩ ∈ a_domain
3. **Demand conservation** — sum(a[q,n,k]) ≤ 1 for each q ∈ Q
4. **Mandatory terminals** — first and last V[k][*] of each route must have x_k = 1
5. **Max spacing with slack** — each active stop must have a prior active stop within d_route_max
6. **Route capacity** — total assigned demand per route ≤ Cap[k]
7. **Operational limit** — at most m_max routes per stop
8. **Total system capacity** — sum(Cap[k]) ≤ Capt + Cad

## Python/Gurobi Coding Conventions
- Portuguese variable names (`inicio`, `fim`, `a_domain`, `V_tamanho`, `S_Indices`)
- Uses `gp.quicksum` (never plain `sum` for model expressions)
- `model.addVars` with tuple-keyed dictionaries
- One function per constraint group in `constraints.py`

## Known Gaps
1. No `requirements.txt` or `pyproject.toml`
2. No unit tests
3. No LICENSE file
4. Solver lacks default TimeLimit / MIPGap parameters
5. `run.py` lacks error handling for solve failure