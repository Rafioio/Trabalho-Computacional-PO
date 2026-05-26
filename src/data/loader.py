import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_DAT_PATH = os.path.join(_DATA_DIR, "dados.dat")
_JSON_PATH = os.path.join(_DATA_DIR, "dados.json")


def load_dat(path=None):
    if path is None:
        path = _DAT_PATH
    with open(path) as f:
        text = f.read()

    data = {}

    # Scalars
    data["P"] = _parse_scalar(text, "P")
    data["W1"] = _parse_scalar(text, "W1")
    data["W2"] = _parse_scalar(text, "W2")
    data["W3"] = _parse_scalar(text, "W3")
    data["W4"] = _parse_scalar(text, "W4")
    data["omega"] = _parse_scalar(text, "omega")
    data["NumN"] = int(_parse_scalar(text, "NumN"))
    data["NumK"] = int(_parse_scalar(text, "NumK"))
    data["NumQ"] = int(_parse_scalar(text, "NumQ"))
    data["d_route_max"] = _parse_scalar(text, "d_route_max")
    data["d_walk_max"] = _parse_scalar(text, "d_walk_max")
    data["m_max"] = int(_parse_scalar(text, "m_max"))
    data["Capt"] = int(_parse_scalar(text, "Capt"))

    # Sets
    data["C"] = _parse_set(text, "C")
    NumN = data["NumN"]
    NumK = data["NumK"]
    NumQ = data["NumQ"]
    data["T"] = sorted([i for i in range(1, NumN + 1) if i not in data["C"]])

    # Tuple sets
    data["I"] = _parse_tuple_set(text, "I")
    data["L"] = _parse_tuple_set(text, "L")

    # 1D arrays
    data["de"] = _parse_array(text, "de")
    data["w"] = _parse_array(text, "w")
    data["V_tamanho"] = [int(v) for v in _parse_array(text, "V_tamanho")]

    # 2D arrays
    data["d"] = _parse_2d_array(text, "d")
    data["D"] = _parse_flat_D(text, "D", NumK, NumN)

    # V (padded with zeros in .dat, strip to V_tamanho)
    V_raw = _parse_2d_array(text, "V")
    data["V"] = [
        [int(v) for v in row[: data["V_tamanho"][k]]]
        for k, row in enumerate(V_raw)
    ]

    # Derived lists
    data["Q"] = list(range(1, NumQ + 1))
    data["N"] = list(range(1, NumN + 1))
    data["K"] = list(range(1, NumK + 1))

    return data


def _parse_scalar(text, name):
    m = re.search(rf"(?m)^{name}\s*=\s*([^;]+);", text)
    if not m:
        raise ValueError(f"Scalar '{name}' not found")
    val = m.group(1).strip()
    try:
        return float(val)
    except ValueError:
        return val


def _parse_set(text, name):
    m = re.search(rf"(?m)^{name}\s*=\s*\{{([^}}]+)\}};", text)
    if not m:
        raise ValueError(f"Set '{name}' not found")
    raw = m.group(1)
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def _parse_tuple_set(text, name):
    m = re.search(rf"(?m)^{name}\s*=\s*\{{([^}}]+)\}};", text, re.DOTALL)
    if not m:
        raise ValueError(f"Tuple set '{name}' not found")
    raw = m.group(1)
    pairs = re.findall(r"<(\d+),(\d+)>", raw)
    return [(int(a), int(b)) for a, b in pairs]


def _parse_array(text, name):
    m = re.search(rf"(?m)^{name}\s*=\s*\[([^\]]*)\];", text, re.DOTALL)
    if not m:
        raise ValueError(f"Array '{name}' not found")
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


def _parse_2d_array(text, name):
    """Parse a 2D array that may be flat (d, V) or triple-nested (D: [ [ [...], ... ], ... ])."""
    m = re.search(rf"(?m)^{name}\s*=\s*\[", text)
    if not m:
        raise ValueError(f"2D array '{name}' not found")

    # Start past the opening bracket, depth = 1
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
    rows = _parse_2d_array(text, name)
    D_3d = []
    for k in range(NumK):
        start = k * NumN
        D_3d.append(rows[start : start + NumN])
    return D_3d


def load_json(path=None):
    if path is None:
        path = _JSON_PATH
    with open(path) as f:
        data = json.load(f)
    _normalize_I_L(data)
    return data


def _normalize_I_L(data):
    if isinstance(data.get("I"), list):
        if data["I"] and isinstance(data["I"][0], (list, tuple)):
            data["I"] = [(x[0], x[1]) for x in data["I"]]
    if isinstance(data.get("L"), list):
        if data["L"] and isinstance(data["L"][0], (list, tuple)):
            data["L"] = [(x[0], x[1]) for x in data["L"]]


def save_json(data, path=None):
    if path is None:
        path = _JSON_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON: {path}")


def convert_dat_to_json(dat_path=None, json_path=None):
    data = load_dat(dat_path)
    save_json(data, json_path)
    return data


if __name__ == "__main__":
    convert_dat_to_json()