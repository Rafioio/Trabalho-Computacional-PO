import numpy as np


def validate(data):
    errors = []

    N_list = data.get("N", [])
    K_list = data.get("K", [])
    Q_list = data.get("Q", [])

    NumN = data.get("NumN", len(N_list))
    NumK = data.get("NumK", len(K_list))
    NumQ = data.get("NumQ", len(Q_list))

    if len(N_list) != NumN:
        errors.append(f"N size mismatch: {len(N_list)} vs {NumN}")
    if len(K_list) != NumK:
        errors.append(f"K size mismatch: {len(K_list)} vs {NumK}")
    if len(Q_list) != NumQ:
        errors.append(f"Q size mismatch: {len(Q_list)} vs {NumQ}")

    C = data.get("C", [])
    T = data.get("T", [])

    if len(C) + len(T) != NumN:
        errors.append(f"C ({len(C)}) + T ({len(T)}) != N ({NumN})")
    if any(n in C for n in T):
        errors.append("C and T overlap")

    I = data.get("I", [])
    L = data.get("L", [])

    for n, k in I:
        if n not in N_list:
            errors.append(f"I contains invalid node {n}")
        if k not in K_list:
            errors.append(f"I contains invalid route {k}")

    for q, k in L:
        if q not in Q_list:
            errors.append(f"L contains invalid demand {q}")
        if k not in K_list:
            errors.append(f"L contains invalid route {k}")

    d = data.get("d", [])
    if len(d) != NumQ:
        errors.append(f"d rows ({len(d)}) != Q ({NumQ})")
    elif len(d) > 0 and len(d[0]) != NumN:
        errors.append(f"d cols ({len(d[0])}) != N ({NumN})")

    D = data.get("D", [])
    D_len = len(D) if not isinstance(D, np.ndarray) else D.shape[0]
    if D_len != NumK:
        errors.append(f"D outer ({D_len}) != K ({NumK})")
    elif D_len > 0:
        d1 = len(D[0]) if not isinstance(D, np.ndarray) else D.shape[1]
        if d1 != NumN:
            errors.append(f"D mid ({d1}) != N ({NumN})")
        else:
            if not isinstance(D, np.ndarray):
                d2 = len(D[0][0]) if D[0] else 0
            else:
                d2 = D.shape[2]
            if d2 != NumN:
                errors.append(f"D inner ({d2}) != N ({NumN})")

    V = data.get("V", [])
    V_tamanho = data.get("V_tamanho", [])

    if len(V) != NumK:
        errors.append(f"V rows ({len(V)}) != K ({NumK})")
    if len(V_tamanho) != NumK:
        errors.append(f"V_tamanho len ({len(V_tamanho)}) != K ({NumK})")

    for k_idx, k in enumerate(K_list):
        vt = V_tamanho[k_idx] if k_idx < len(V_tamanho) else 0
        if vt < 2:
            errors.append(f"V_tamanho[{k_idx}] = {vt}, need >= 2")
        if k_idx < len(V):
            route = V[k_idx][:vt]
            if len(route) != vt:
                errors.append(f"V[{k_idx}] len ({len(route)}) != V_tamanho ({vt})")
            if len(set(route)) != vt:
                errors.append(f"V[{k_idx}] contains duplicates")

    for k_idx, k in enumerate(K_list):
        if k_idx >= len(V_tamanho) or k_idx >= len(V):
            break
        vt = V_tamanho[k_idx]
        if vt < 1:
            continue
        first = V[k_idx][0]
        last = V[k_idx][vt - 1]
        if (first, k_idx + 1) not in I:
            errors.append(f"Terminal {first} not in I for route {k_idx + 1}")
        if (last, k_idx + 1) not in I:
            errors.append(f"Terminal {last} not in I for route {k_idx + 1}")

    has_domain = False
    for q in Q_list:
        for n, k in I:
            if (q, k) in L and q - 1 < len(d) and n - 1 < len(d[0]) and d[q - 1][n - 1] <= data.get("d_walk_max", 100):
                has_domain = True
                break
        if has_domain:
            break
    if not has_domain:
        errors.append("a_domain is empty — no feasible demand assignments")

    de = data.get("de", [])
    w = data.get("w", [])

    if len(de) != NumQ:
        errors.append(f"de len ({len(de)}) != Q ({NumQ})")
    if any(v < 0 for v in de):
        errors.append("de has negative values")
    if len(w) != NumN:
        errors.append(f"w len ({len(w)}) != N ({NumN})")
    if any(v <= 0 for v in w):
        errors.append("w has non-positive values")

    if errors:
        for e in errors:
            print(f"[VALIDATION ERROR] {e}")
        return False
    return True