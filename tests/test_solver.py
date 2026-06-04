"""Tests for the solver — model building, solving, and export."""

import json
import os
import tempfile

import pytest
from gurobipy import GRB


class TestSolver:
    def test_build_model_returns_correct_types(self, built_model):
        model, vars, domains, obj_exprs = built_model
        assert model.ModelName == "soubuz"
        assert isinstance(vars, dict)
        for key in ("x", "x_k", "a", "Cap", "Cad", "s_k"):
            assert key in vars, f"Missing variable: {key}"
        for key in ("f1", "f2", "f3", "f4"):
            assert key in obj_exprs, f"Missing objective: {key}"

    def test_build_and_solve_optimal(self, solved_results):
        assert solved_results["status"] == "optimal"
        assert solved_results["obj"] > 0

    def test_rebuild_and_resolve(self, loaded_data):
        from src.model.solver import build_and_solve
        r1 = build_and_solve(loaded_data, verbose=False)
        r2 = build_and_solve(loaded_data, verbose=False)
        assert r1["status"] == "optimal"
        assert r2["status"] == "optimal"
        assert abs(r1["obj"] - r2["obj"]) < 1e-6

    def test_model_has_model(self, solved_results):
        assert "model" in solved_results
        assert solved_results["model"].status == GRB.OPTIMAL

    def test_export_solution_json(self, solved_results, loaded_data):
        from src.utils.export_solution import export_solution
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            p = f.name
        try:
            sol = export_solution(solved_results, loaded_data, path=p)
            assert sol["status"] == "optimal"
            assert sol["num_pontos_ativos"] > 0
            assert len(sol["pontos_ativos"]) == sol["num_pontos_ativos"]
            assert len(sol["paradas_por_rota"]) > 0
            assert len(sol["capacidade_rotas"]) > 0
            with open(p) as f:
                loaded = json.load(f)
            assert loaded["valor_objetivo"] == sol["valor_objetivo"]
        finally:
            os.unlink(p)

    def test_export_solution_csv(self, solved_results, loaded_data):
        from src.utils.export_solution import export_solution_csv
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            p = f.name
        try:
            sol = export_solution_csv(solved_results, loaded_data, path=p)
            assert sol["num_pontos_ativos"] > 0
        finally:
            os.unlink(p)

    def test_solve_with_normalization(self, loaded_data):
        from src.model.solver import build_and_solve
        r = build_and_solve(loaded_data, verbose=False)
        assert r["status"] == "optimal"