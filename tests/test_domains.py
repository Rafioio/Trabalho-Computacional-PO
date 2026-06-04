"""Tests for sparse domain construction (a_domain, S_Indices)."""


class TestDomains:
    def test_a_domain_size(self, loaded_data, domains):
        a_domain = domains["a_domain"]
        assert len(a_domain) > 0, "a_domain should not be empty"
        for q, n, k in a_domain:
            assert q in loaded_data["Q"], f"Invalid demand {q}"
            assert n in loaded_data["N"], f"Invalid node {n}"
            assert k in loaded_data["K"], f"Invalid route {k}"

    def test_a_domain_distance_filter(self, loaded_data, domains):
        d = loaded_data["d"]
        d_walk_max = loaded_data["d_walk_max"]
        for q, n, k in domains["a_domain"]:
            assert d[q - 1][n - 1] <= d_walk_max, (
                f"Distance {d[q-1][n-1]:.1f} > {d_walk_max} for {q},{n}")

    def test_a_domain_route_membership(self, loaded_data, domains):
        I_set = set(loaded_data["I"])
        L_set = set(loaded_data["L"])
        for q, n, k in domains["a_domain"]:
            assert (n, k) in I_set, f"({n},{k}) not in I"
            assert (q, k) in L_set, f"({q},{k}) not in L"

    def test_S_Indices_structure(self, loaded_data, domains):
        S = domains["S_Indices"]
        assert len(S) > 0
        for k, idx in S:
            assert k in loaded_data["K"], f"Invalid route {k}"
            assert 2 <= idx, f"Index {idx} should be >= 2"

    def test_S_Indices_count(self, loaded_data, domains):
        S = domains["S_Indices"]
        expected = sum(max(0, vt - 1) for vt in loaded_data["V_tamanho"])
        assert len(S) == expected, (
            f"S_Indices count {len(S)} != expected {expected}")

    def test_a_domain_covers_at_least_one_demand(self, loaded_data, domains):
        covered = {q for q, n, k in domains["a_domain"]}
        assert len(covered) > 0, "No demand zones covered by a_domain"

    def test_a_domain_no_invalid_node_indices(self, loaded_data, domains):
        N_set = set(loaded_data["N"])
        for q, n, k in domains["a_domain"]:
            assert n in N_set, f"Node {n} not in N"