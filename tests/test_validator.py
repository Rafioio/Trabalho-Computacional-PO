"""Tests for the data validator — pass and fail cases."""

import json
import os
import tempfile

import pytest


class TestValidator:
    def test_valid_data_passes(self, small_json_path):
        from src.data.loader import load_json
        from src.utils.validator import validate
        data = load_json(small_json_path)
        assert validate(data) is True

    def test_missing_N_fails(self, small_data):
        from src.utils.validator import validate
        bad = dict(small_data)
        bad["N"] = bad["N"][:-1]
        assert validate(bad) is False

    def test_overlapping_CT_fails(self, small_data):
        from src.utils.validator import validate
        bad = dict(small_data)
        bad["C"] = [1]
        bad["T"] = [1, 2, 3]
        assert validate(bad) is False

    def test_bad_d_shape_fails(self, small_data):
        from src.utils.validator import validate
        bad = dict(small_data)
        bad["d"] = [[0.0] * (bad["NumN"] - 1) for _ in range(bad["NumQ"])]
        assert validate(bad) is False

    def test_bad_w_negative_fails(self, small_data):
        from src.utils.validator import validate
        bad = dict(small_data)
        bad["w"] = [0.0] * bad["NumN"]
        assert validate(bad) is False

    def test_V_duplicate_fails(self, small_data):
        from src.utils.validator import validate
        bad = dict(small_data)
        bad["V"] = [[1, 1, 2] for _ in bad["V"]]
        assert validate(bad) is False

    def test_empty_a_domain_fails(self, small_data):
        from src.utils.validator import validate
        bad = dict(small_data)
        bad["d_walk_max"] = 0.001
        assert validate(bad) is False