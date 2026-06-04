"""Tests for all 8 constraint groups."""

import gurobipy as gp
from gurobipy import GRB


class TestConstraints:
    def test_c1_x_k_leq_x(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        for n, k in loaded_data["I"][:5]:
            c = model.getConstrByName(f"c1_{n}_{k}")
            assert c is not None, f"c1_{n}_{k} not found"
            assert c.Sense == GRB.LESS_EQUAL

    def test_c2_a_leq_x_k(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        for q, n, k in domains["a_domain"][:3]:
            c = model.getConstrByName(f"c2_{q}_{n}_{k}")
            assert c is not None, f"c2_{q}_{n}_{k} not found"
            assert c.Sense == GRB.LESS_EQUAL

    def test_c3_demand_conservation(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        for q in loaded_data["Q"]:
            c = model.getConstrByName(f"c3_{q}")
            assert c is not None, f"c3_{q} not found"

    def test_c4_terminal_activated(self, built_model, loaded_data, domains):
        model, vars, domains_b, obj_exprs = built_model
        for k_idx, k in enumerate(loaded_data["K"]):
            first = loaded_data["V"][k_idx][0]
            last = loaded_data["V"][k_idx][loaded_data["V_tamanho"][k_idx] - 1]
            cinicio = model.getConstrByName(f"c4_inicio_{k}")
            cfim = model.getConstrByName(f"c4_fim_{k}")
            assert cinicio is not None, f"c4_inicio_{k} not found"
            assert cfim is not None, f"c4_fim_{k} not found"

    def test_c5_spacing_with_slack(self, built_model, loaded_data, domains):
        model, vars, domains_b, obj_exprs = built_model
        for k_idx, k in enumerate(loaded_data["K"]):
            for idx in range(2, loaded_data["V_tamanho"][k_idx] + 1):
                cname = f"c5_{k}_{idx}"
                c = model.getConstrByName(cname)
                if c is not None:
                    assert c.Sense == GRB.LESS_EQUAL

    def test_c6_route_capacity(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        for k in loaded_data["K"]:
            c = model.getConstrByName(f"c6_{k}")
            assert c is not None, f"c6_{k} not found"

    def test_c7_routes_per_stop(self, built_model, loaded_data):
        model, vars, domains, obj_exprs = built_model
        for n in loaded_data["N"]:
            c = model.getConstrByName(f"c7_{n}")
            assert c is not None, f"c7_{n} not found"

    def test_c8_system_capacity(self, built_model):
        model, vars, domains, obj_exprs = built_model
        c = model.getConstrByName("c8")
        assert c is not None, "c8 not found"

    def test_total_constraint_count(self, built_model, loaded_data, domains):
        model, vars, domains_b, obj_exprs = built_model
        names = {c.ConstrName for c in model.getConstrs()}
        N_c1 = len(loaded_data["I"])
        N_c2 = len(domains["a_domain"])
        N_c3 = loaded_data["NumQ"]
        N_c4 = loaded_data["NumK"] * 2
        N_c5 = sum(max(0, vt - 1) for vt in loaded_data["V_tamanho"])
        N_c6 = loaded_data["NumK"]
        N_c7 = loaded_data["NumN"]
        N_c8 = 1
        expected = N_c1 + N_c2 + N_c3 + N_c4 + N_c5 + N_c6 + N_c7 + N_c8
        assert len(names) == expected, (
            f"Expected {expected} constraints, got {len(names)}")