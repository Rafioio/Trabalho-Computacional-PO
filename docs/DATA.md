# SouBuz Data & Utilities Documentation

## File Reference

| File | Role |
|------|------|
| `src/data/dados.dat` (17 MB) | Original canonical input data |
| `src/data/dados.json` (39 MB) | Converted JSON (fast loading via `load_json()`) |
| `src/data/loader.py` | Parse .dat, I/O for .json |
| `src/model/domains.py` | Build sparse domains: `a_domain`, `S_Indices` |
| `src/model/variables.py` | Create Gurobi decision variables |
| `src/model/objective.py` | Define f1–f4 objective function |
| `src/model/constraints.py` | 8 constraint groups (one function each) |
| `src/model/solver.py` | Assemble model, call optimize |
| `src/utils/generate_data.py` | Realistic random data generator (JSON only) — clustered positions, log-normal demand, beta quality, routes |
| `src/utils/validator.py` | Pre-solve data consistency checks |
| `src/utils/export_solution.py` | Export solver results → `solucao.json` for viewer |
| `src/scripts/map_viewer.py` | Map visualization, density analysis, before/after comparison |
| `src/run.py` | Entry point: load → validate → solve → print |
| `docs/Modelagem_PO.pdf` | Mathematical model description (LaTeX) |

---

## Data Generator: `src/utils/generate_data.py`

The generator creates realistic synthetic cities using:
- **Clustered positions** (60% of nodes cluster near existing points, Gaussian scatter)
- **Log-normal demand** levels per zone
- **Beta-distributed** technical quality per node
- **Minimum-distance enforcement** between nodes (default 80 m)
- **Route generation** with X-sorted node sequences to simulate bus lines
- **Batch mode** for creating multiple scenarios

### Usage Patterns

```bash
# Default (50 nodes, 5 routes, 35 zones)
python src/utils/generate_data.py

# Full-scale (matches original dimensions from dados.dat)
python src/utils/generate_data.py \
    --num-n 500 --num-k 10 --num-q 30 \
    --route-len 140 --num-c 40 \
    --output test_full.json

# Small validation instance (fast solver runs)
python src/utils/generate_data.py \
    --num-n 30 --num-k 3 --num-q 5 \
    --route-len 8 --d-walk-max 200 \
    --output small.json

# Reproduce exact instance
python src/utils/generate_data.py --seed 42 --output fixed.json

# Batch-generate 10 scenarios
python src/utils/generate_data.py --scenarios 10 --prefix cenario
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num-n` | 50 | Number of nodes |
| `--num-k` | 5 | Number of bus routes |
| `--num-q` | 35 | Number of demand zones |
| `--route-len` | ~35% of N | Nodes per route sequence |
| `--num-c` | ~8% of N | Number of centroid nodes in C |
| `--grid-width` | 3000 | City width in meters |
| `--grid-height` | 3000 | City height in meters |
| `--min-dist` | 80 | Minimum distance between nodes (m) |
| `--d-walk-max` | 500.0 | Max walking distance |
| `--d-route-max` | 800.0 | Max spacing between consecutive stops |
| `--capt` | 800 | Base system capacity |
| `--seed` | random | RNG seed for reproducibility |
| `--scenarios` | — | Batch-generate N scenarios |
| `--prefix` | cenario | Filename prefix for batch mode |

### Output JSON format

```json
{
  "NumN": 50, "NumK": 5, "NumQ": 35,
  "Q": [1, 2, ..., 35],
  "N": [1, 2, ..., 50],
  "K": [1, 2, ..., 5],
  "C": [...],
  "T": [...],
  "I": [[1, 1], ...],
  "L": [[1, 1], ...],
  "de": [...],
  "d": [[d11, d12, ...], ...],
  "w": [...],
  "D": [[[D111, ...], ...], ...],
  "V": [[n1, n2, ...], ...],
  "V_tamanho": [18, 18, ...],
  "P": 1000, "omega": 50.0,
  "W1": 0.35, "W2": 0.15, "W3": 0.30, "W4": 0.20,
  "Capt": 800, "m_max": 3,
  "d_route_max": 800.0, "d_walk_max": 500.0,
  "metadata": { ... },
  "visualizacao": { "posicoes_pontos": [...], "posicoes_demandas": [...], ... },
  "estatisticas": { "demanda_total": ..., "perc_acessiveis": ..., ... }
}
```

The 26 solver fields (`NumN`, `Q`, `N`, `K`, `I`, `L`, etc.) are at the top level — this is the **flat format** expected by `loader.load_json()`. The `metadata`, `visualizacao`, and `estatisticas` sections are preserved for the map viewer but ignored by the solver.

| Field | Shape | Description |
|-------|-------|-------------|
| `Q` | `[Q]` | Demand zone indices (1-indexed) |
| `N` | `[N]` | Node indices (1-indexed) |
| `K` | `[K]` | Route indices (1-indexed) |
| `C` | `[40]` | Centroids (subset of N) |
| `T` | `[460]` | Non-centroids (N \ C) |
| `I` | `[~1400][2]` | Valid `(node, route)` pairs |
| `L` | `[~170][2]` | Valid `(demand, route)` pairs |
| `de` | `[Q]` | Demand level per zone |
| `d` | `[Q][N]` | Walking distances demand→node |
| `w` | `[N]` | Technical feasibility per node |
| `D` | `[K][N][N]` | Route-wise distance matrix |
| `V` | `[K][V_tamanho[k]]` | Ordered route node sequences |
| `V_tamanho` | `[K]` | Length of each route |

---

## Data Flow

```
                          ┌──────────────────┐
                          │  generate_data.py │
                          └────────┬─────────┘
                                   │ .json (flat)
                                   v
      ┌──────────────┐      ┌──────────────┐
      │  dados.dat   │─────>│  dados.json  │
      │  (original)  │ .dat │  (fast load) │
      └──────┬───────┘      └──────┬────────┘
             │                     │
             v                     v
      ┌──────────────────────────────────────┐
      │            data/loader.py            │
      │    load_dat()         load_json()   │
      └────────────────┬─────────────────────┘
                       │ data dict
                       v
      ┌──────────────────────────────────────┐
      │          utils/validator.py          │
      │           validate(data)             │
      └────────────────┬─────────────────────┘
                       │ (optional, catches issues early)
                       v
      ┌──────────────────────────────────────┐
      │           model/solver.py            │
      │  build_and_solve(data) → results     │
      │                                      │
      │  domains → variables → objective     │
      │  → constraints (8) → optimize       │
      └────────────────┬─────────────────────┘
                       │ results dict
                       v                          ┌──────────────────┐
      ┌──────────────────────────────┐            │  map_viewer.py   │
      │       src/run.py            │            │ (visualization)  │
      │ prints obj, f1-f4, Cap[k]   │─────JSON──>│                  │
      │ export_solution(results)    │ solucao    │ pontos_ativos,   │
      └──────────────────────────────┘ .json      │ route_stops,     │
                                                  │ demand_assign    │
                                                  └──────────────────┘

---

## How to Run

```bash
# From repo root:
python -m src.run
python -m src.run --data /path/to/data.json
python -m src.run --data /path/to/dados.dat
```

---

## Module APIs

### `data/loader.py`

```python
from src.data.loader import load_dat, load_json, save_json

data = load_dat()            # parses src/data/dados.dat (~3s)
data = load_json()           # loads src/data/dados.json (~0.5s)
data = load_json("custom.json")
save_json(data, "output.json")
```

### `utils/validator.py`

```python
from src.utils.validator import validate
ok = validate(data)    # True/False, prints errors
```

### `model/solver.py`

```python
from src.model.solver import build_and_solve
results = build_and_solve(data, verbose=True)
# results["status"]  → "optimal", "infeasible", or "status_N"
# results["model"]   → gp.Model (after optimize)
# results["vars"]    → dict of tupledicts
# results["obj_exprs"] → {"f1": ..., "f2": ..., "f3": ..., "f4": ...}
```

---

## Archived File Reference

These files were removed after migration from CPLEX/OPL to Gurobi:

| Old file | Superseded by |
|----------|--------------|
| `src/modelo.mod` | `src/model/*.py` modules |
| `src/modelo_gurobi.py` | `src/model/solver.py` + `src/run.py` |

---

## Original Data Statistics

| Metric | Value |
|--------|-------|
| Nodes (N) | 500 |
| Routes (K) | 10 |
| Demand zones (Q) | 30 |
| Centroids (C) | 40 |
| Non-centroids (T) | 460 |
| Route-stop pairs (I) | 1400 |
| Demand-route pairs (L) | 171 |
| Entries per route (V_tamanho) | 140 (uniform) |
| Demand range (de) | 154–483 |
| Feasibility range (w) | 0.50–1.00 |
| D matrix size | 10 × 500 × 500 = 2.5M floats |

## Post-Solve Export

The solution can be exported for visualization:

```python
from src.utils.export_solution import export_solution
from src.model.solver import build_and_solve
from src.data.loader import load_json

data = load_json("dados_generated.json")
results = build_and_solve(data)
export_solution(results, data)  # creates solucao.json
```

The `solucao.json` file contains `pontos_ativos`, `paradas_por_rota`, `capacidade_rotas`, `atribuicao_demanda`, and objective values — all consumed by `map_viewer.py`.

## Map Viewer: `src/scripts/map_viewer.py`

Opens a generated JSON file and produces interactive visualizations:

```bash
python src/scripts/map_viewer.py                   # dados_generated.json
python src/scripts/map_viewer.py cenario_01.json   # batch output
python src/scripts/map_viewer.py solucao.json       # with solution overlay
```

Features:
- City map with stop quality gradient (red→green), demand bubbles, accessibility circle
- Scale bar and compass rose (adaptive to viewport)
- Density hexbin plots for stops and demand nodes
- Quality and demand distribution histograms
- Before/after comparison when `solucao.json` exists

---

## Validation Patterns

### Programmatic validation

```python
from src.utils.validator import validate
if not validate(data):
    print("Data invalid — check printed errors")
    exit(1)
```

### Common issues & fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `a_domain` empty | Nodes too far from demand zones | Reduce `d_walk_max` OR shrink coordinate range (1000×1000 grid recommended) |
| Model infeasible | `Capt` too small for total demand | Increase `--capt` |
| Model infeasible | Route spacing violation (constraint 5) | Add more stops per route so consecutive stops are ≤ `d_route_max` |
| Slow solve | D matrix too large | Reduce `--num-n` |
| Terminal not in I | Generated route's endpoints not in I set | Check `generate_data.py` — every route node should be added to I |

---

## Logging & Debugging

### Gurobi solver parameters

```python
# In src/model/solver.py, before model.optimize():
model.Params.LogToConsole = 1
model.Params.MIPGap = 0.01       # 1% optimality gap
model.Params.TimeLimit = 300     # 5 minute limit
model.Params.MIPFocus = 1        # focus on feasible solutions
```

### Log file patterns

The `.gitignore` excludes these solver artifacts:

```
*.log          # solver logs (gurobi.log, cplex.log)
*.lp           # LP file dumps
*.mps          # MPS file dumps
*.sol          # solution files
*.sav          # saved models
```

---

## Performance Notes

### Problem size vs. solve time

| Variables | Growth | Typical count (full) |
|-----------|--------|---------------------|
| `x[n]` | N | 500 |
| `x_k[n,k]` | \|I\| | ~1,400 |
| `a[q,n,k]` | \|a_domain\| | ~769 (with d_walk_max=100) |
| `Cap[k]` | K | 10 |
| `Cad` | 1 | 1 |
| `s_k[k,idx]` | ~\|V_tamanho\| | 1,390 |

Total: ~4,000 variables (mixed-integer) for the full instance.

### Memory for D matrix

`D[K][N][N]` = 10 × 500 × 500 = **2.5 million floats**. At 8 bytes each ≈ 20 MB. The JSON file is ~39 MB due to formatting overhead.

---

## Future Work

- [ ] Add `pyproject.toml` with tool configs (ruff, mypy, pytest)
- [ ] Add open-source solver alternative (e.g., `pulp` + HiGHS)
- [ ] Add unit tests for each constraint group
- [ ] Post-solve analysis: which stops are active, demand allocation map
- [ ] Parallel solve with parameter sweeps (weight sensitivity, capacity scenarios)