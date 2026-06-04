import json
import os
import re
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_DAT_PATH = os.path.join(_DATA_DIR, "dados.dat")
_JSON_PATH = os.path.join(_DATA_DIR, "dados.json")


def load_data(filepath=None):
    """
    Load data from JSON or DAT file with automatic format detection.
    
    Args:
        filepath: Path to data file. If None, tries default locations.
        
    Returns:
        Dictionary with loaded data
    """
    if filepath is None:
        # Try JSON first, then DAT
        if os.path.exists(_JSON_PATH):
            return load_json(_JSON_PATH)
        elif os.path.exists(_DAT_PATH):
            return load_dat(_DAT_PATH)
        else:
            raise FileNotFoundError(f"No data file found. Expected {_JSON_PATH} or {_DAT_PATH}")
    
    filepath = Path(filepath)
    
    # Detect format by file extension or content
    if filepath.suffix.lower() == '.json':
        return load_json(filepath)
    elif filepath.suffix.lower() == '.dat':
        return load_dat(filepath)
    else:
        # Try to detect by reading first few characters
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '{':
                print(f"Detected JSON format for {filepath}")
                return load_json(filepath)
            else:
                print(f"Detected DAT format for {filepath}")
                return load_dat(filepath)


def load_dat(path=None):
    """Load data from legacy DAT format."""
    if path is None:
        path = _DAT_PATH
    
    print(f"Loading DAT file: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    data = {}

    # Scalars (with optional values)
    scalar_names = ["P", "W1", "W2", "W3", "W4", "omega", "NumN", "NumK", "NumQ", 
                    "d_route_max", "d_walk_max", "m_max", "Capt"]
    
    for name in scalar_names:
        try:
            value = _parse_scalar(text, name)
            if name in ["NumN", "NumK", "NumQ", "m_max", "Capt"]:
                data[name] = int(value)
            else:
                data[name] = float(value)
        except ValueError:
            # Set default values if not found
            defaults = {
                "P": 1000.0,
                "W1": 0.35,
                "W2": 0.15,
                "W3": 0.30,
                "W4": 0.20,
                "omega": 50.0,
                "NumN": 50,
                "NumK": 5,
                "NumQ": 35,
                "d_route_max": 800.0,
                "d_walk_max": 500.0,
                "m_max": 3,
                "Capt": 800
            }
            data[name] = defaults.get(name, 0)
            print(f"Warning: '{name}' not found in DAT file, using default: {data[name]}")

    # Sets
    try:
        data["C"] = _parse_set(text, "C")
    except ValueError:
        data["C"] = []
    
    NumN = data["NumN"]
    data["T"] = sorted([i for i in range(1, NumN + 1) if i not in data["C"]])

    # Tuple sets
    try:
        data["I"] = _parse_tuple_set(text, "I")
    except ValueError:
        data["I"] = []
    
    try:
        data["L"] = _parse_tuple_set(text, "L")
    except ValueError:
        data["L"] = []

    # 1D arrays
    try:
        data["de"] = _parse_array(text, "de")
    except ValueError:
        data["de"] = [50] * data["NumQ"]
    
    try:
        data["w"] = _parse_array(text, "w")
    except ValueError:
        data["w"] = [0.5] * data["NumN"]
    
    try:
        data["V_tamanho"] = [int(v) for v in _parse_array(text, "V_tamanho")]
    except ValueError:
        route_len = max(2, data["NumN"] // 10)
        data["V_tamanho"] = [route_len] * data["NumK"]

    # 2D arrays
    try:
        data["d"] = np.array(_parse_2d_array(text, "d"), dtype=np.float64)
    except ValueError:
        data["d"] = np.random.rand(data["NumQ"], data["NumN"]) * 1000

    try:
        data["D"] = _parse_flat_D(text, "D", data["NumK"], data["NumN"])
    except ValueError:
        data["D"] = np.zeros((data["NumK"], data["NumN"], data["NumN"]), dtype=np.float64)
        for k in range(data["NumK"]):
            for i in range(data["NumN"]):
                for j in range(data["NumN"]):
                    data["D"][k, i, j] = abs(i - j) * 100

    # V matrix
    try:
        V_raw = _parse_2d_array(text, "V")
        data["V"] = [
            [int(v) for v in row[: data["V_tamanho"][k]]]
            for k, row in enumerate(V_raw)
        ]
    except ValueError:
        data["V"] = []
        for k in range(data["NumK"]):
            start = k * (data["NumN"] // data["NumK"]) + 1
            end = min(start + data["V_tamanho"][k], data["NumN"] + 1)
            data["V"].append(list(range(start, end)))

    # Derived lists
    data["Q"] = list(range(1, data["NumQ"] + 1))
    data["N"] = list(range(1, data["NumN"] + 1))
    data["K"] = list(range(1, data["NumK"] + 1))
    
    # Ensure all arrays have correct lengths
    if len(data["de"]) < data["NumQ"]:
        data["de"].extend([50] * (data["NumQ"] - len(data["de"])))
    if len(data["w"]) < data["NumN"]:
        data["w"].extend([0.5] * (data["NumN"] - len(data["w"])))

    return data


def load_json(path=None):
    """
    Load data from JSON format generated by generate_data.py.
    """
    if path is None:
        path = _JSON_PATH
    
    print(f"Loading JSON file: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Debug: print top-level keys
    print(f"JSON top-level keys: {list(raw_data.keys())}")
    
    data = {}
    
    # Case 1: Format from generate_data.py (has 'parametros' key)
    if "parametros" in raw_data:
        print("Detected new JSON format (with 'parametros')")
        params = raw_data["parametros"]
        
        data["NumN"] = params.get("NumN", 0)
        data["NumK"] = params.get("NumK", 0)
        data["NumQ"] = params.get("NumQ", 0)
        data["d_route_max"] = params.get("d_route_max", 800.0)
        data["d_walk_max"] = params.get("d_walk_max", 500.0)
        data["P"] = params.get("P", 1000.0)
        data["Capt"] = params.get("Capt", 800)
        data["m_max"] = params.get("m_max", 3)
        data["omega"] = params.get("omega", 50.0)
        data["W1"] = params.get("W1", 0.35)
        data["W2"] = params.get("W2", 0.15)
        data["W3"] = params.get("W3", 0.30)
        data["W4"] = params.get("W4", 0.20)
        
        # Sets
        data["C"] = raw_data.get("C", [])
        data["T"] = [i for i in range(1, data["NumN"] + 1) if i not in data["C"]]
        
        # Tuple sets - IMPORTANT: I and L are lists of lists or tuples
        raw_I = raw_data.get("I", [])
        data["I"] = []
        for item in raw_I:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                data["I"].append((item[0], item[1]))
            elif isinstance(item, dict):
                data["I"].append((item.get("n"), item.get("k")))
        
        raw_L = raw_data.get("L", [])
        data["L"] = []
        for item in raw_L:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                data["L"].append((item[0], item[1]))
            elif isinstance(item, dict):
                data["L"].append((item.get("q"), item.get("k")))
        
        # Arrays
        data["de"] = raw_data.get("de", [])
        data["w"] = raw_data.get("w", [])
        
        # Distance matrix d
        if "d" in raw_data:
            data["d"] = np.array(raw_data["d"], dtype=np.float64)
        else:
            data["d"] = np.zeros((data["NumQ"], data["NumN"]), dtype=np.float64)
        
        # D matrix (3D distances per route)
        if "D" in raw_data:
            data["D"] = np.array(raw_data["D"], dtype=np.float64)
        elif "D_list" in raw_data:
            data["D"] = np.array(raw_data["D_list"], dtype=np.float64)
        else:
            data["D"] = np.zeros((data["NumK"], data["NumN"], data["NumN"]), dtype=np.float64)
        
        # V (routes) and V_tamanho
        data["V"] = raw_data.get("V", [])
        data["V_tamanho"] = raw_data.get("V_tamanho", [len(v) for v in data["V"]])
        
        # If V is empty, try to get from visualizacao.rotas
        if not data["V"] and "visualizacao" in raw_data:
            vis = raw_data["visualizacao"]
            if "rotas" in vis:
                data["V"] = vis["rotas"]
                data["V_tamanho"] = [len(v) for v in data["V"]]
                print(f"Loaded routes from visualizacao.rotas: {len(data['V'])} routes")
        
        # Derived lists
        data["Q"] = list(range(1, data["NumQ"] + 1))
        data["N"] = list(range(1, data["NumN"] + 1))
        data["K"] = list(range(1, data["NumK"] + 1))
        
    # Case 2: Old JSON format (direct keys)
    elif "NumN" in raw_data:
        print("Detected old JSON format (direct keys)")
        
        for key in ["NumN", "NumK", "NumQ", "d_route_max", "d_walk_max", "P", "Capt", "m_max", "omega", "W1", "W2", "W3", "W4"]:
            data[key] = raw_data.get(key, 0)
        
        data["C"] = raw_data.get("C", [])
        data["T"] = raw_data.get("T", [])
        data["I"] = raw_data.get("I", [])
        data["L"] = raw_data.get("L", [])
        data["de"] = raw_data.get("de", [])
        data["w"] = raw_data.get("w", [])
        data["d"] = np.array(raw_data.get("d", []), dtype=np.float64)
        data["D"] = np.array(raw_data.get("D", []), dtype=np.float64)
        data["V"] = raw_data.get("V", [])
        data["V_tamanho"] = raw_data.get("V_tamanho", [])
        data["Q"] = raw_data.get("Q", list(range(1, data.get("NumQ", 0) + 1)))
        data["N"] = raw_data.get("N", list(range(1, data.get("NumN", 0) + 1)))
        data["K"] = raw_data.get("K", list(range(1, data.get("NumK", 0) + 1)))
        
    else:
        raise ValueError(f"Unknown JSON format. Expected 'parametros' or 'NumN' key. Got: {list(raw_data.keys())}")
    
    # Post-processing: ensure types
    data["de"] = [float(d) if isinstance(d, (int, float)) else 50.0 for d in data["de"]]
    data["w"] = [float(w) if isinstance(w, (int, float)) else 0.5 for w in data["w"]]
    
    # Ensure I and L are properly formatted
    data["I"] = [(int(n), int(k)) for (n, k) in data["I"] if n is not None and k is not None]
    data["L"] = [(int(q), int(k)) for (q, k) in data["L"] if q is not None and k is not None]
    
    # Ensure V_tamanho matches V
    if len(data["V_tamanho"]) != len(data["V"]):
        data["V_tamanho"] = [len(v) for v in data["V"]]
    
    print(f"Data loaded: {data['NumN']} nodes, {data['NumK']} routes, {data['NumQ']} demand zones")
    print(f"  - I set size: {len(data['I'])}")
    print(f"  - L set size: {len(data['L'])}")
    print(f"  - V routes: {len(data['V'])}")
    
    return data


def _parse_scalar(text, name):
    """Parse scalar value from DAT file."""
    patterns = [
        rf"(?m)^{name}\s*=\s*([^;]+);",
        rf"(?m)^{name}\s+=\s+([^;]+);",
        rf"(?m)^{name}\s*:=\s*([^;]+);",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            val = m.group(1).strip()
            try:
                return float(val)
            except ValueError:
                return val
    
    raise ValueError(f"Scalar '{name}' not found")


def _parse_set(text, name):
    """Parse set from DAT file."""
    patterns = [
        rf"(?m)^{name}\s*=\s*\{{([^}}]+)\}};",
        rf"(?m)^{name}\s+=\s+\{{([^}}]+)\}};",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            raw = m.group(1)
            vals = []
            for part in raw.split(","):
                part = part.strip()
                if part:
                    vals.append(int(part))
            return vals
    
    raise ValueError(f"Set '{name}' not found")


def _parse_tuple_set(text, name):
    """Parse tuple set from DAT file."""
    patterns = [
        rf"(?m)^{name}\s*=\s*\{{([^}}]+)\}};",
        rf"(?m)^{name}\s+=\s+\{{([^}}]+)\}};",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            raw = m.group(1)
            pairs = re.findall(r"<(\d+),(\d+)>", raw)
            if pairs:
                return [(int(a), int(b)) for a, b in pairs]
    
    raise ValueError(f"Tuple set '{name}' not found")


def _parse_array(text, name):
    """Parse 1D array from DAT file."""
    patterns = [
        rf"(?m)^{name}\s*=\s*\[([^\]]*)\];",
        rf"(?m)^{name}\s+=\s+\[([^\]]*)\];",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            raw = m.group(1)
            vals = []
            for part in raw.split(","):
                part = part.strip()
                if part:
                    try:
                        vals.append(float(part))
                    except ValueError:
                        pass
            return vals
    
    raise ValueError(f"Array '{name}' not found")


def _parse_2d_array(text, name):
    """Parse 2D array from DAT file."""
    patterns = [
        rf"(?m)^{name}\s*=\s*\[",
        rf"(?m)^{name}\s+=\s+\[",
    ]
    
    m = None
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            break
    
    if not m:
        raise ValueError(f"2D array '{name}' not found")

    start = m.end()
    depth = 1
    end = start
    while end < len(text) and depth > 0:
        if text[end] == "[":
            depth += 1
        elif text[end] == "]":
            depth -= 1
        end += 1
    body = text[start : end - 1]

    rows = []
    for match in re.finditer(r"\[([^\[\]]*)\]", body):
        row_text = match.group(1)
        vals = [float(x.strip()) for x in row_text.split(",") if x.strip()]
        rows.append(vals)
    return rows


def _parse_flat_D(text, name, NumK, NumN):
    """Parse 3D D matrix from DAT file."""
    rows = _parse_2d_array(text, name)
    D_3d = []
    rows_per_k = NumN
    for k in range(NumK):
        start = k * rows_per_k
        end = start + rows_per_k
        if end <= len(rows):
            D_3d.append(rows[start:end])
        else:
            D_3d.append([[0.0] * NumN for _ in range(NumN)])
    return np.array(D_3d, dtype=np.float64)


def save_json(data, path=None):
    """Save data to JSON file."""
    if path is None:
        path = _JSON_PATH
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_data = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            json_data[key] = value.tolist()
        elif isinstance(value, np.integer):
            json_data[key] = int(value)
        elif isinstance(value, np.floating):
            json_data[key] = float(value)
        else:
            json_data[key] = value
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {path}")


def convert_dat_to_json(dat_path=None, json_path=None):
    """Convert DAT file to JSON format."""
    data = load_dat(dat_path)
    save_json(data, json_path)
    return data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        data = load_data(filepath)
        print(f"\nData loaded successfully!")
        print(f"  NumN: {data['NumN']}")
        print(f"  NumK: {data['NumK']}")
        print(f"  NumQ: {data['NumQ']}")
        print(f"  Total demand: {sum(data['de']):.0f}")
    else:
        convert_dat_to_json()