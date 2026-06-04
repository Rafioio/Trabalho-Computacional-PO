"""Tests for the data loader — .dat parsing, JSON loading, numpy conversion."""

import json
import os
import tempfile

import numpy as np
import pytest


OPL_SAMPLE = """
NumN = 6;
NumK = 2;
NumQ = 2;
P = 1000;
W1 = 0.35;
W2 = 0.15;
W3 = 0.30;
W4 = 0.20;
omega = 50;
d_route_max = 800;
d_walk_max = 500;
m_max = 3;
Capt = 800;
C = {1, 6};
I = {<1,1> <2,1> <3,1> <4,2> <5,2>};
L = {<1,1> <1,2> <2,1>};
de = [100, 200];
w = [0.8, 0.6, 0.9, 0.7, 0.5, 0.4];
V_tamanho = [3, 3];
d = [[0, 50, 100, 200, 300, 400],
     [400, 300, 200, 100, 50, 0]];
D = [[[0, 50, 0, 0, 0, 0],
      [50, 0, 60, 0, 0, 0],
      [0, 60, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 80, 0],
      [0, 0, 0, 80, 0, 90],
      [0, 0, 0, 0, 90, 0]]];
V = [[1, 2, 3], [4, 5, 6]];
"""


@pytest.fixture(scope="session")
def opl_dat_path():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write(OPL_SAMPLE)
        path = f.name
    yield path
    os.unlink(path)


class TestLoadDat:
    def test_load_scalars(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert data["NumN"] == 6
        assert data["NumK"] == 2
        assert data["NumQ"] == 2
        assert data["P"] == 1000.0
        assert data["d_walk_max"] == 500.0
        assert data["m_max"] == 3

    def test_load_sets(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert data["C"] == [1, 6]
        assert data["T"] == [2, 3, 4, 5]

    def test_load_tuple_sets(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert (1, 1) in data["I"]
        assert (5, 2) in data["I"]
        assert (1, 2) in data["L"]

    def test_load_arrays(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert data["de"] == [100.0, 200.0]
        assert data["V_tamanho"] == [3, 3]

    def test_load_d_as_numpy(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert isinstance(data["d"], np.ndarray)
        assert data["d"].shape == (2, 6)

    def test_load_D_as_numpy(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert isinstance(data["D"], np.ndarray)
        assert data["D"].shape == (2, 6, 6)

    def test_V_stripped_to_V_tamanho(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert data["V"] == [[1, 2, 3], [4, 5, 6]]

    def test_derived_ranges(self, opl_dat_path):
        from src.data.loader import load_dat
        data = load_dat(opl_dat_path)
        assert data["Q"] == [1, 2]
        assert data["N"] == [1, 2, 3, 4, 5, 6]
        assert data["K"] == [1, 2]


class TestLoadJson:
    def test_load_json_basic(self, small_json_path):
        from src.data.loader import load_json
        data = load_json(small_json_path)
        assert data["NumN"] == 10
        assert data["NumK"] == 2
        assert data["NumQ"] == 3

    def test_load_json_numpy_conversion(self, small_json_path):
        from src.data.loader import load_json
        data = load_json(small_json_path)
        assert isinstance(data["d"], np.ndarray)
        assert isinstance(data["D"], np.ndarray)
        assert data["d"].ndim == 2
        assert data["D"].ndim == 3

    def test_load_json_I_normalization(self, small_json_path):
        from src.data.loader import load_json
        data = load_json(small_json_path)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in data["I"])

    def test_load_json_L_normalization(self, small_json_path):
        from src.data.loader import load_json
        data = load_json(small_json_path)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in data["L"])


class TestSaveJson:
    def test_save_and_reload(self, small_data):
        from src.data.loader import save_json, load_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            p = f.name
        try:
            save_json(small_data, p)
            reloaded = load_json(p)
            assert reloaded["NumN"] == small_data["NumN"]
            assert reloaded["NumK"] == small_data["NumK"]
            assert reloaded["de"] == small_data["de"]
        finally:
            os.unlink(p)