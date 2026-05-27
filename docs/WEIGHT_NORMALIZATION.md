# Weight Normalization via Utopia/Anti-Utopia Points

## Motivation

The SouBuz model has four objectives (f1–f4) with vastly different numerical scales:

| Objective | Typical range | Description |
|-----------|--------------|-------------|
| f1 (social cost) | 500k – 10M | Walking distance × demand + unserved penalties |
| f2 (tech feasibility) | 1k – 5k | Sum of 1/w[n] over active route-stop pairs |
| f3 (infrastructure) | 0 – 500 | Number of active stops + ω × additional capacity |
| f4 (spacing penalty) | 0 – 350k | Slack variables for max-spacing violations |

The original weights (`W1=0.001, W2=1.0, W3=0.94, W4=1.0`) were hand-tuned to compensate for these scale differences. This makes them hard to interpret as *true preferences* — a weight of 1.0 on f2 and 0.001 on f1 does not mean f2 is 1000× more important; it just reflects that f1 naturally produces values ~1000× larger.

## Solution: Utopia/Anti-Utopia Normalization

The normalizer computes a **payoff table** by solving the model four times, each minimizing a single objective. From this table it derives:

- **Utopia point** (ideal): the best value each objective can achieve independently
- **Anti-utopia point** (nadir): the worst value each objective takes across all single-objective solutions
- **Normalization factor** = 1 / (anti_utopia − utopia) for each objective

The original weights are multiplied by these factors, making the weighted sum objective *dimensionless and scale-independent*. After normalization, each term `Wi × (fi − utopia_i) / range_i` lives in approximately [0, 1], and Wi reflects true priority.

## Module: `src/utils/weight_normalizer.py`

```
compute_payoff_table(data, verbose=False)
    → 4×4 matrix M where M[i][j] = fj evaluated at solution that minimizes fi

normalize_weights(data, verbose=False)
    → computes payoff, derives utopia/anti-utopia, updates data["W1"]..["W4"] in-place
    → stores payoff metadata in data["_normalization"]
```

### Algorithm

1. **Build model** once via `solver.build_model(data)` — creates vars, constraints, and objective expressions without setting any objective
2. **For each i in [f1, f2, f3, f4]**:
   - Set `model.setObjective(fi_expr, GRB.MINIMIZE)`
   - Solve
   - Record fj values for all 4 objectives at the optimal solution
3. **Payoff table**: 4×4 matrix, rows = minimized objective, columns = observed values
4. **Utopia**: diagonal of the payoff table (best of each objective)
5. **Anti-utopia**: column-wise maximum (worst across all solutions for each objective)
6. **Range** = anti_utopia − utopia
7. **Factor** = 1 / range (or 1.0 if range ≈ 0)
8. **New weights** = original_weights × factors

### Usage

```bash
# Basic — normalizes silently
python -m src.run --normalize-weights

# Verbose — prints the full payoff table and factor computation
python -m src.run --normalize-weights --verbose-normalization

# Full pipeline with synthetic data
python src/utils/generate_data.py --num-n 30 --num-k 3 --num-q 5 -o /tmp/test.json
python -m src.run --data /tmp/test.json --json --normalize-weights --verbose-normalization
```

## Payoff Table Example

Output from a 20-node, 2-route, 3-zone instance:

```
Matriz payoff (linha = obj minimizado, coluna = valor obtido)
                                  f1                f2                f3                f4
                f1       544153.2000            7.8865           19.0000            2.0000
                f2       668000.0000            5.1463           19.0000            2.0000
                f3       668000.0000            5.1463            2.0000            2.0000
                f4       668000.0000            8.9199           19.0000            0.0000

            utopia       544153.2000            5.1463            2.0000            0.0000
       anti-utopia       668000.0000            8.9199           19.0000            2.0000
             fator          0.000008          0.265000          0.058824          0.500000

          original          0.001000          1.000000          0.940000          1.000000
       normalizado          0.000000          0.265000          0.055294          0.500000
```

Interpretation:
- **Row "f1"** = minimizing social cost yields f1=544k, f2=7.9, f3=19, f4=2
- **Row "f2"** = minimizing tech feasibility yields f2=5.15, but f1 jumps to 668k (all demand unserved)
- **Utopia f1** = 544k (best social cost), **anti-utopia f1** = 668k (worst social cost)
- **Factor f1** = 1/(668k−544k) = 0.000008 — f1 is already well-optimized by its own structure
- **Factor f2** = 0.265 — moderate range, reduced from 1.0 to compensate for scale

## Integration Points

| File | Change |
|------|--------|
| `src/model/objective.py` | Extracted `build_objective_exprs()` — creates f1–f4 expressions without `setObjective` |
| `src/model/solver.py` | Extracted `build_model()` — builds vars + constraints but no objective, reusable by normalizer |
| `src/utils/weight_normalizer.py` | New — `compute_payoff_table()` and `normalize_weights()` |
| `src/run.py` | Added `--normalize-weights` and `--verbose-normalization` flags |

## Performance

Normalization adds **4 MIP solves** (one per objective) before the final weighted solve. Each single-objective solve is typically faster than the full weighted sum:

- Small instance (20 nodes, 2 routes): < 1s total overhead
- Medium instance (100 nodes, 5 routes): ~5–30s
- Full instance (500 nodes, 10 routes): ~1–5 min

MIP starts are reused between solves (Gurobi loads the previous solution as a warm start), speeding up subsequent solves.

## Limitations & Future Improvements

- **Anti-utopia via payoff table** is a lower bound (the true nadir may be worse). A tighter bound requires additional exploration.
- **If a single-objective solve is infeasible**, that factor falls back to 1.0 with a warning. The main solve still proceeds.
- **Factors depend only on data**, not weights. They can be cached to disk after the first computation, avoiding repeated normalization on the same dataset.