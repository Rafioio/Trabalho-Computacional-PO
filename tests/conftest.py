"""Shared fixtures for SouBuz tests."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.utils.generate_data import generate_single, parse_args


@pytest.fixture(scope="session")
def small_args():
    return parse_args([
        "--num-n", "10",
        "--num-k", "2",
        "--num-q", "3",
        "--seed", "42",
        "--output", "/dev/null",
        "--quiet",
    ])


@pytest.fixture(scope="session")
def small_data(small_args):
    """Small deterministic dataset (10 nodes, 2 routes, 3 zones)."""
    import numpy as np
    rng = np.random.default_rng(42)
    return generate_single(small_args, 42, rng, quiet=True)


@pytest.fixture(scope="session")
def small_json_path(small_data):
    """Write small_data to a temp JSON file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(small_data, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="session")
def loaded_data(small_json_path):
    """small_data loaded back via load_json (numpy arrays applied)."""
    from src.data.loader import load_json
    return load_json(small_json_path)


@pytest.fixture(scope="session")
def domains(loaded_data):
    from src.model.domains import build_a_domain, build_S_indices
    return {
        "a_domain": build_a_domain(loaded_data),
        "S_Indices": build_S_indices(loaded_data),
    }


@pytest.fixture(scope="session")
def built_model(loaded_data, domains):
    """Full Gurobi model (built but not solved)."""
    from src.model.solver import build_model
    return build_model(loaded_data, domains)


@pytest.fixture(scope="session")
def solved_results(loaded_data):
    """Solve a tiny instance and return results."""
    from src.model.solver import build_and_solve
    return build_and_solve(loaded_data, verbose=False)