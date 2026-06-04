def build_a_domain(data):
    domain = []
    d = data["d"]
    d_walk_max = data["d_walk_max"]
    for q in data["Q"]:
        for n, k in data["I"]:
            if (q, k) in data["L"] and d[q - 1][n - 1] <= d_walk_max:
                domain.append((q, n, k))
    return domain


def build_S_indices(data):
    indices = []
    for k_idx, k in enumerate(data["K"]):
        for idx in range(2, data["V_tamanho"][k_idx] + 1):
            indices.append((k, idx))
    return indices