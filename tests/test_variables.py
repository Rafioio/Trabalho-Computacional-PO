"""Tests for decision variable creation — types, bounds, domains."""

from gurobipy import GRB


class TestVariables:
    def test_x_is_binary(self, built_model):
        model, vars, domains, obj_exprs = built_model
        for v in vars["x"].values():
            assert v.vtype == GRB.BINARY, f"x[{v.varName}] not binary"

    def test_x_k_is_binary(self, built_model):
        model, vars, domains, obj_exprs = built_model
        for v in vars["x_k"].values():
            assert v.vtype == GRB.BINARY, f"x_k[{v.varName}] not binary"

    def test_a_is_continuous(self, built_model):
        model, vars, domains, obj_exprs = built_model
        for v in vars["a"].values():
            assert v.vtype == GRB.CONTINUOUS, f"a[{v.varName}] not continuous"
            assert v.lb == 0.0
            assert v.ub == 1.0

    def test_Cap_is_integer(self, built_model):
        model, vars, domains, obj_exprs = built_model
        for v in vars["Cap"].values():
            assert v.vtype == GRB.INTEGER, f"Cap[{v.varName}] not integer"

    def test_Cad_is_integer(self, built_model):
        model, vars, domains, obj_exprs = built_model
        v = vars["Cad"]
        assert v.vtype == GRB.INTEGER
        assert v.lb == 0

    def test_s_k_is_continuous(self, built_model):
        model, vars, domains, obj_exprs = built_model
        for v in vars["s_k"].values():
            assert v.vtype == GRB.CONTINUOUS, f"s_k[{v.varName}] not continuous"
            assert v.lb == 0.0

    def test_x_count_matches_N(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        assert len(vars["x"]) == loaded_data["NumN"]

    def test_x_k_count_matches_I(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        assert len(vars["x_k"]) == len(loaded_data["I"])

    def test_a_count_matches_a_domain(self, built_model, domains):
        model, vars, domains_b, obj_exprs = built_model
        assert len(vars["a"]) == len(domains["a_domain"])

    def test_Cap_count_matches_K(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        assert len(vars["Cap"]) == loaded_data["NumK"]

    def test_s_k_count_matches_S_Indices(self, built_model, domains):
        model, vars, domains_b, obj_exprs = built_model
        assert len(vars["s_k"]) == len(domains["S_Indices"])