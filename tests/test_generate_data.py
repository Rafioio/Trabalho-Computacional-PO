"""Tests for the synthetic data generator."""

import numpy as np
import pytest

from src.utils.generate_data import parse_args, generate_single, generate_quality


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def default_args():
    return parse_args([
        "--num-n", "30", "--num-k", "3", "--num-q", "5",
        "--seed", "42", "--output", "/dev/null", "--quiet",
    ])


class TestGenerateData:
    def test_default_output_shape(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        assert data["NumN"] == 30
        assert data["NumK"] == 3
        assert data["NumQ"] == 5
        assert len(data["N"]) == 30
        assert len(data["K"]) == 3
        assert len(data["Q"]) == 5

    def test_centroid_sets_are_disjoint(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        assert len(data["C"]) + len(data["T"]) == data["NumN"]
        assert set(data["C"]).isdisjoint(set(data["T"]))

    def test_route_structure(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        assert len(data["V"]) == data["NumK"]
        assert len(data["V_tamanho"]) == data["NumK"]
        for k in range(data["NumK"]):
            assert data["V_tamanho"][k] == len(data["V"][k])
            assert data["V_tamanho"][k] >= 2

    def test_I_pairs_integrity(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        for n, k in data["I"]:
            assert 1 <= n <= data["NumN"]
            assert 1 <= k <= data["NumK"]
            assert n in data["N"]
            assert k in data["K"]

    def test_L_pairs_integrity(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        for q, k in data["L"]:
            assert 1 <= q <= data["NumQ"]
            assert 1 <= k <= data["NumK"]
            assert q in data["Q"]
            assert k in data["K"]

    def test_d_matrix_shape(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        d = np.array(data["d"])
        assert d.shape == (data["NumQ"], data["NumN"])

    def test_D_matrix_shape(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        D = np.array(data["D"])
        assert D.shape == (data["NumK"], data["NumN"], data["NumN"])

    def test_de_and_w_lengths(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        assert len(data["de"]) == data["NumQ"]
        assert len(data["w"]) == data["NumN"]

    def test_quality_bounds(self, rng):
        q = generate_quality(1000, rng)
        assert all(0.2 <= v <= 1.0 for v in q)
        assert len(q) == 1000

    def test_metadata_present(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        assert "metadata" in data
        assert "visualizacao" in data
        assert "estatisticas" in data
        assert data["metadata"]["semente"] == 42

    def test_scalars_present(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        for key in ("P", "omega", "W1", "W2", "W3", "W4", "Capt", "m_max",
                     "d_route_max", "d_walk_max", "NumN", "NumK", "NumQ"):
            assert key in data, f"Missing scalar: {key}"

    def test_reproducible_seed(self):
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        args = parse_args(["--num-n", "10", "--num-k", "2", "--num-q", "3",
                            "--seed", "99", "--output", "/dev/null", "--quiet"])
        d1 = generate_single(args, 99, rng1, quiet=True)
        d2 = generate_single(args, 99, rng2, quiet=True)
        assert d1["de"] == d2["de"]
        assert d1["d"] == d2["d"]
        assert d1["V"] == d2["V"]
        assert d1["I"] == d2["I"]

    def test_route_nodes_in_I(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        for k_idx, route in enumerate(data["V"]):
            for n in route:
                assert (n, k_idx + 1) in data["I"], (
                    f"Node {n} on route {k_idx+1} not in I")

    def test_terminal_nodes_in_I(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        for k_idx, route in enumerate(data["V"]):
            k = k_idx + 1
            assert (route[0], k) in data["I"], f"First terminal {route[0]} not in I"
            assert (route[-1], k) in data["I"], f"Last terminal {route[-1]} not in I"

    def test_no_duplicate_nodes_in_route(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        for route in data["V"]:
            assert len(route) == len(set(route)), "Duplicate node in route V"

    def test_each_demand_zone_in_at_least_one_L(self, default_args):
        rng = np.random.default_rng(42)
        data = generate_single(default_args, 42, rng, quiet=True)
        served = {q for q, k in data["L"]}
        assert served == set(data["Q"]), "Not all demand zones in L"