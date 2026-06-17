#!/usr/bin/env python3
"""Generate visualizations for selected runs."""

import csv
import json
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

BASE = Path("runs")

# Load runs
runs = []
with open(BASE / "runs_index.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        runs.append(row)

optimal = [r for r in runs if r['status'] == 'optimal']
tiers_data = defaultdict(list)
for r in optimal:
    tiers_data[r['tier']].append(r)

def find_run_dir(run_id_prefix):
    for d in BASE.iterdir():
        if d.is_dir() and d.name.startswith(run_id_prefix):
            return d
    return None

def select_4(tier_entries):
    priority = ['ahp_default', 'extreme_user', 'extreme_op', 'equal',
                'social', 'operator', 'spacing', 'technical']
    selected = []
    sel_w = []
    for w in priority:
        c = [e for e in tier_entries if e['weights'] == w]
        if not c:
            continue
        for mu, th in [('1.0','1.0'),('1.5','0.5'),('0.5','1.5'),('2.0','1.0'),('1.0','2.0')]:
            for e in c:
                if e['mu'] == mu and e['theta'] == th:
                    selected.append(e)
                    sel_w.append(w)
                    break
            if sel_w and sel_w[-1] == w:
                break
        if not sel_w or sel_w[-1] != w:
            selected.append(c[0])
            sel_w.append(w)
        if len(selected) >= 4:
            break
    return selected

SELECTED = {}
for tier in ['micro', 'pequena', 'media', 'grande']:
    SELECTED[tier] = select_4(tiers_data.get(tier, []))

# Generate data and visualizations
import numpy as np
from src.utils.generate_data import generate_data, parse_args as gen_args
from src.data.loader import save_json

for tier, sel in SELECTED.items():
    print(f"\n[{tier}] Processing {len(sel)} runs...")
    for s in sel:
        rid = s['run_id']
        run_dir = find_run_dir(rid)
        if not run_dir:
            print(f"  run_{rid}: directory not found")
            continue
        if not (run_dir / "solucao.json").exists():
            print(f"  run_{rid}: no solucao.json")
            continue

        pfile = run_dir / "params.json"
        if not pfile.exists():
            print(f"  run_{rid}: no params.json")
            continue
        with open(pfile) as f:
            p = json.load(f)

        N, K, Q, seed = p['NumN'], p['NumK'], p['NumQ'], p['seed']
        print(f"  run_{rid}: N={N} K={K} Q={Q} seed={seed} {p['weight_label']}")

        try:
            args = gen_args([
                "--num-n", str(N), "--num-k", str(K), "--num-q", str(Q),
                "--seed", str(seed), "--output", "/dev/null", "--quiet"])
            rng = np.random.default_rng(seed)
            data = generate_data(args, rng, seed)
            save_json(data, str(run_dir / "input_data.json"))
        except Exception as e:
            print(f"    data generation failed: {e}")
            continue

        # map_viewer: solucao
        out = run_dir / "map_solucao.png"
        cmd = [sys.executable, "-m", "src.scripts.map_viewer",
               str(run_dir / "input_data.json"),
               "--solution", str(run_dir / "solucao.json"),
               "--output", str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                print(f"    map_solucao.png OK")
            else:
                print(f"    map_solucao ERROR: {r.stderr[:150]}")
        except Exception as e:
            print(f"    map_solucao FAILED: {e}")

        # map_viewer: comparison
        out2 = run_dir / "map_comparison.png"
        cmd2 = [sys.executable, "-m", "src.scripts.map_viewer",
                str(run_dir / "input_data.json"),
                "--solution", str(run_dir / "solucao.json"),
                "--output", str(out2), "--comparison"]
        try:
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
            if r2.returncode == 0:
                print(f"    map_comparison.png OK")
        except:
            pass

        # map_viewer: density-only
        out3 = run_dir / "map_density.png"
        cmd3 = [sys.executable, "-m", "src.scripts.map_viewer",
                str(run_dir / "input_data.json"),
                "--density-only", "--output", str(out3)]
        try:
            r3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=60)
            if r3.returncode == 0:
                print(f"    map_density.png OK")
        except:
            pass

print("\nDone!")