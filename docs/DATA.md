# SouBuz Data & Utilities Documentation

## File Reference

| File | Role |
|------|------|
| `src/data/dados.dat` (17 MB) | Original canonical input data |
| `src/data/dados.json` (39 MB) | Converted JSON (fast loading via `load_json()`) |
| `src/data/20260504_ponto_onibus.csv` (5 MB) | BH bus stop CSV — 70k stop-route pairs, UTM coordinates |
| `src/data/loader.py` | Parse .dat, I/O for .json |
| `src/model/domains.py` | Build sparse domains: `a_domain`, `S_Indices` |
| `src/model/variables.py` | Create Gurobi decision variables |
| `src/model/objective.py` | Define f1–f4 objective function |
| `src/model/constraints.py` | 8 constraint groups (one function each) |
| `src/model/solver.py` | Assemble model, call optimize |
| `src/utils/generate_data.py` | Realistic random data generator (JSON only) — clustered positions, log-normal demand, beta quality, routes |
| `src/utils/convert_csv_to_data.py` | Converts BH bus stop CSV → model input JSON (UTM coordinates, deduplication, demand generation) |
| `src/utils/validator.py` | Pre-solve data consistency checks |
| `src/utils/export_solution.py` | Export solver results → `solucao.json` for viewer |
| `src/utils/batch_runner.py` | Batch experiment runner — 6 instance tiers, multi-weight grid, full result capture |
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
- **Route generation** with three methods: `nearest` (no intersections), `grid` (vertical strips), and `hybrid` (K-means + intersections, default)
- **Intersection support** — nodes can belong to multiple routes (simulating transfer points)
- **Coverage guarantee** — every node belongs to at least one route
- **Batch mode** for creating multiple scenarios

### Usage Patterns

```bash
# Default (200 nodes, 3 routes, 60 zones)
python src/utils/generate_data.py

# Full-scale (matches original dimensions from dados.dat)
python src/utils/generate_data.py \
    --num-n 500 --num-k 10 --num-q 30 \
    --route-len 140 \
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

# Choose route generation method
python src/utils/generate_data.py --route-method hybrid     # K-means + intersections (default)
python src/utils/generate_data.py --route-method nearest    # no intersections
python src/utils/generate_data.py --route-method grid       # vertical strips
```

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num-n` | 200 | Number of nodes |
| `--num-k` | 3 | Number of bus routes |
| `--num-q` | 60 | Number of demand zones |
| `--route-len` | ~35% of N | Nodes per route sequence |
| `--route-method` | `hybrid` | Route generation: `nearest` (no intersections), `grid`, or `hybrid` (with intersections) |
| `--grid-width` | 3000 | City width in meters |
| `--grid-height` | 3000 | City height in meters |
| `--min-dist` | 80 | Minimum distance between nodes (m) |
| `--d-walk-max` | 250.0 | Max walking distance |
| `--d-route-max` | 1000.0 | Max spacing between consecutive stops |
| `--capt` | 800 | Base system capacity |
| `--seed` | random | RNG seed for reproducibility |
| `--scenarios` | — | Batch-generate N scenarios |
| `--prefix` | cenario | Filename prefix for batch mode |

### Output JSON format

```json
{
  "NumN": 200, "NumK": 3, "NumQ": 60,
  "Q": [1, 2, ..., 60],
  "N": [1, 2, ..., 200],
  "K": [1, 2, ..., 3],
  "C": [...],
  "T": [...],
  "I": [[1, 1], ...],
  "L": [[1, 1], ...],
  "de": [...],
  "d": [[d11, d12, ...], ...],
  "w": [...],
  "D": [[[D111, ...], ...], ...],
  "V": [[n1, n2, ...], ...],
  "V_tamanho": [70, 70, 70],
  "P": 1000, "omega": 50.0,
  "W1": 0.35, "W2": 0.15, "W3": 0.30, "W4": 0.20,
  "Capt": 800, "m_max": 3,
  "d_route_max": 1000.0, "d_walk_max": 250.0,
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
                           ┌──────────────────┐      ┌──────────────────────────┐
                           │  generate_data.py │      │  convert_csv_to_data.py │
                           │  (synthetic)      │      │  (BH CSV → JSON)        │
                           └────────┬─────────┘      └───────────┬──────────────┘
                                    │ .json (flat)              │ .json (flat)
                                    v                            v
       ┌──────────────┐      ┌──────────────────────────────────────────────┐
       │  dados.dat   │─────>│               dados.json                    │
       │  (original)  │ .dat │  (fast load, accepts BH CSV output too)     │
       └──────┬───────┘      └──────────────────────┬───────────────────────┘
              │                                     │
              v                                     v
       ┌────────────────────────────────────────────────────┐
       │               data/loader.py                        │
       │   load_dat()                 load_json()           │
       └────────────────┬───────────────────────────────────┘
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

## BH CSV Converter: `src/utils/convert_csv_to_data.py`

Converts the public BH bus stop CSV (available at `src/data/20260504_ponto_onibus.csv`) into the SouBuz model JSON format.

### Input Format

The CSV has the following columns (semicolon-delimited):

| Column | Description |
|--------|-------------|
| `ID_PONTO_ONIBUS_LINHA` | Unique route-stop pair ID |
| `COD_LINHA` | Route code |
| `NOME_LINHA` | Route name |
| `NOME_SUB_LINHA` | Sub-line variant (PRINCIPAL, NOTURNO, etc.) |
| `ORIGEM` | Origin terminal |
| `IDENTIFICADOR_PONTO_ONIBUS` | Stop identifier |
| `GEOMETRIA` | UTM coordinate as `POINT (x y)` (UTM zone 23S) |

### Usage

```bash
# Full conversion (943 route variants, ~70k unique stop coordinates)
python src/utils/convert_csv_to_data.py --output dados_bh.json

# Small test instance (first 3 routes, 20 stops each, 5 demand zones)
python src/utils/convert_csv_to_data.py \
    --max-routes 3 --max-stops 20 --num-q 5 \
    --output small_bh.json

# Custom parameters
python src/utils/convert_csv_to_data.py \
    --max-routes 10 --max-stops 50 --num-q 30 \
    --d-walk-max 400 --d-route-max 800 \
    --capt 1000 --seed 123 \
    --output dados_bh_10rotas.json
```

### What the Converter Does

1. **Parses** the CSV, grouping stops by `(COD_LINHA, NOME_SUB_LINHA)` as route variants
2. **Deduplicates** stops by UTM coordinate — stops at the same location used by different routes become a single node
3. **Builds** the `I` set (route-stop pairs) and `V` (ordered stop sequences per route)
4. **Generates** synthetic demand zones (`Q`, `de`, `L`) clustered near real stops
5. **Computes** distance matrix `d[Q][N]` (walking) and `D[K][N][N]` (route spacing)
6. **Assigns** random technical quality `w[n]` via beta distribution (0.2–1.0)
7. **Outputs** flat JSON compatible with `loader.load_json()` and `solver.build_and_solve()`

### Important: Route Ordering

The CSV lists stops in the order they appear for each route variant. The converter preserves this order as the `V[k]` sequence. If the CSV order does not reflect the actual physical path, the spacing constraint (C5) may produce unrealistic results.

### Downsampling

With 943 route variants and ~70k unique coordinates, the full dataset is too large for the MILP solver. Use `--max-routes` and `--max-stops` to create practical instances:

| Scenario | Routes | Stops/Route | Nodes | Solve time (est.) |
|----------|--------|-------------|-------|-------------------|
| Tiny | 3 | 20 | ~200 | < 1s |
| Small | 5 | 50 | ~900 | ~30s |
| Medium | 10 | 100 | ~3k | ~5 min |
| Full | all | all | ~70k | infeasible |

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

---

## Batch Runner: `src/utils/batch_runner.py`

The batch runner executes a structured experiment campaign across 6 instance tiers, each with multiple seeds and weight combinations, saving full results to `docs/runs/`.

### Six Instance Tiers

| Tier | N | K | Q | Seeds | Weights | TimeLimit | Est. solves |
|------|---|---|---|-------|---------|-----------|-------------|
| micro | 20 | 2 | 5 | 3 | 8 | 60s | 24 final + 12 norm |
| pequena | 50 | 4 | 10 | 3 | 8 | 120s | 24 final + 12 norm |
| media | 150 | 6 | 20 | 2 | 4 | 300s | 8 final + 8 norm |
| grande | 300 | 8 | 30 | 2 | 4 | 600s | 8 final + 8 norm |
| muito_grande | 500 | 10 | 30 | 2 | 2 | 1200s | 4 final + 8 norm |
| extrema | 800 | 15 | 40 | 1 | 2 | 1800s | 2 final + 4 norm |

Each run includes the normalization phase (4 payoff-table solves) amortised across all weight combos for that seed. Total: ~70 final solves + ~52 normalization solves, estimated ~5 h wall time.

### Weight Grid

The `WEIGHT_GRID` dictionary defines three resolution levels:

| Grid | Combos | Emphasis |
|------|--------|----------|
| `full` | 8 | AHP default, social, operator, spacing, technical, equal, extreme user, extreme operator |
| `medium` | 4 | AHP default, social, operator, equal |
| `sparse` | 2 | AHP default, operator |

Larger tiers use sparser grids to keep runtime reasonable.

### Usage

```bash
# Full experiment (~5 h)
python -m src.utils.batch_runner

# Selected tiers only
python -m src.utils.batch_runner --tiers micro,media

# Dry run (print plan, exit)
python -m src.utils.batch_runner --dry-run

# Quick mode (half-size instances, for debugging)
python -m src.utils.batch_runner --quick --dry-run
```

### Output Structure

```
docs/runs/
├── runs_index.csv           # Global index of all runs
├── run_0001/
│   ├── params.json          # Run parameters (tier, seed, weights, time limits)
│   ├── summary.json         # Results (f1–f4 raw/normalized, macro-objectives, buses, stops)
│   ├── solucao.json         # Full solution for visualization
│   ├── objectives.png       # Bar chart: raw + normalized/weighted objectives
│   └── map.png              # 2D projection of active stops and route assignments
├── run_0002/
│   └── ...
└── ...
```

### AI Model Integration (Future)

The batch runner generates labelled training data for a future ML model that predicts optimal weight corrections. Each run records:

- `params.json` — instance size, weights, seed, timing
- `summary.json` — raw objective values, normalised values, macro-objectives, capacity allocation
- `runs_index.csv` — flat table for cross-run analysis

A future `src/utils/weight_tuner.py` will consume this data to recommend weight adjustments for instances where AHP-estimated weights produce poor trade-off balance.

### Per-Run Time Limit

Every solve sets `model.Params.TimeLimit` to the tier's per-solve limit (60–1800s). If a single solve exceeds this limit, it records as `"time_limit"` status and continues to the next run. This prevents any single instance from consuming the entire experiment window.

## Future Work

- [ ] Add `pyproject.toml` with tool configs (ruff, mypy, pytest)
- [ ] Add open-source solver alternative (e.g., `pulp` + HiGHS)
- [x] Fix broken tests after `generate_single` → `generate_data` rename
- [ ] Post-solve analysis: which stops are active, demand allocation map
- [ ] Parallel solve with parameter sweeps (weight sensitivity, capacity scenarios)