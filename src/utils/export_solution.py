#!/usr/bin/env python3
"""Post-process solver results and export to JSON, CSV, or Excel."""

import json

import numpy as np


def export_solution(results, data, domains=None, path=None):
    model = results["model"]
    v = results["vars"]
    obj_exprs = results["obj_exprs"]

    if domains is None:
        from src.model.domains import build_a_domain
        domains = {"a_domain": build_a_domain(data)}

    pontos_ativos = sorted([
        int(n) for n in range(1, data["NumN"] + 1)
        if v["x"][n].X > 0.5
    ])

    route_stops = {}
    for (n, k) in data["I"]:
        if v["x_k"][n, k].X > 0.5:
            route_stops.setdefault(int(k), []).append(int(n))
    for k in route_stops:
        route_stops[k].sort()

    demand_assignment = []
    for q, n, k in domains.get("a_domain", []):
        val = v["a"][q, n, k].X
        if val > 1e-6:
            demand_assignment.append({
                "demanda": int(q),
                "ponto": int(n),
                "rota": int(k),
                "fracao": round(float(val), 4),
            })

    solucao = {
        "valor_objetivo": round(float(results.get("obj", 0)), 4),
        "status": results.get("status", "unknown"),
        "num_pontos_ativos": len(pontos_ativos),
        "pontos_ativos": pontos_ativos,
        "paradas_por_rota": route_stops,
        "capacidade_rotas": {int(k): int(v["Cap"][k].X) for k in data["K"]},
        "capacidade_adicional": int(v["Cad"].X),
        "objetivos": {
            "f1_custo_social": round(float(obj_exprs["f1"].getValue()), 2),
            "f2_viabilidade_tecnica": round(float(obj_exprs["f2"].getValue()), 2),
            "f3_custo_infraestrutura": round(float(obj_exprs["f3"].getValue()), 2),
            "f4_penalidade_espacamento": round(float(obj_exprs["f4"].getValue()), 2),
        },
        "atribuicao_demanda": demand_assignment,
    }

    if path:
        with open(path, "w") as f:
            json.dump(solucao, f, indent=2)
        print(f"Solucao exportada: {path} ({len(pontos_ativos)} pontos ativos)")
    return solucao


def export_solution_csv(results, data, domains=None, path="solucao.csv"):
    import pandas as pd

    sol = export_solution(results, data, domains, path=None)
    writer = pd.ExcelWriter(path, engine="openpyxl") if path.endswith(".xlsx") else None

    df_objetivos = pd.DataFrame([sol["objetivos"]])
    df_objetivos.insert(0, "valor_objetivo", sol["valor_objetivo"])

    rows_stops = []
    for k, stops in sol["paradas_por_rota"].items():
        for s in stops:
            rows_stops.append({"rota": k, "ponto": s})
    df_paradas = pd.DataFrame(rows_stops) if rows_stops else pd.DataFrame({"rota": [], "ponto": []})

    rows_capacidade = []
    for k, cap in sol["capacidade_rotas"].items():
        rows_capacidade.append({"rota": k, "capacidade": cap})
    df_capacidade = pd.DataFrame(rows_capacidade)
    df_capacidade.loc[len(df_capacidade)] = ["adicional", sol["capacidade_adicional"]]

    df_demanda = pd.DataFrame(sol["atribuicao_demanda"]) if sol["atribuicao_demanda"] else pd.DataFrame(
        {"demanda": [], "ponto": [], "rota": [], "fracao": []}
    )

    df_pontos = pd.DataFrame({
        "ponto": sol["pontos_ativos"],
        "tipo": "selecionado",
    })

    if path.endswith(".xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_objetivos.to_excel(w, sheet_name="objetivos", index=False)
            df_paradas.to_excel(w, sheet_name="paradas_por_rota", index=False)
            df_capacidade.to_excel(w, sheet_name="capacidade", index=False)
            df_demanda.to_excel(w, sheet_name="atribuicao_demanda", index=False)
            df_pontos.to_excel(w, sheet_name="pontos_ativos", index=False)
        print(f"Solucao exportada: {path}")
    else:
        df_objetivos.to_csv(path, index=False)
        if not path.endswith(".csv"):
            path += ".csv"
        print(f"CSV exportado: {path}")

    return sol