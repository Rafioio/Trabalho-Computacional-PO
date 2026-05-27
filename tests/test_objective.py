"""Tests for objective function expressions (f1-f4)."""

import gurobipy as gp


class TestObjective:
    def test_f1_is_lin_expr(self, built_model):
        model, vars, domains, obj_exprs = built_model
        assert isinstance(obj_exprs["f1"], gp.LinExpr), "f1 should be LinExpr"

    def test_f2_is_lin_expr(self, built_model):
        model, vars, domains, obj_exprs = built_model
        assert isinstance(obj_exprs["f2"], gp.LinExpr), "f2 should be LinExpr"

    def test_f3_is_lin_expr(self, built_model):
        model, vars, domains, obj_exprs = built_model
        assert isinstance(obj_exprs["f3"], gp.LinExpr), "f3 should be LinExpr"

    def test_f4_is_lin_expr(self, built_model):
        model, vars, domains, obj_exprs = built_model
        assert isinstance(obj_exprs["f4"], gp.LinExpr), "f4 should be LinExpr"

    def test_all_four_objectives_present(self, built_model):
        model, vars, domains, obj_exprs = built_model
        for name in ("f1", "f2", "f3", "f4"):
            assert name in obj_exprs, f"Missing objective: {name}"

    def test_objective_values_after_solve(self, solved_results):
        obj_exprs = solved_results["obj_exprs"]
        for name in ("f1", "f2", "f3", "f4"):
            val = obj_exprs[name].getValue()
            assert val >= 0, f"{name} = {val} < 0"