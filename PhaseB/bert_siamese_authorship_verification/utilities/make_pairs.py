def make_pairs(impostor_names):
    """
    Generate all possible pairs of impostor names from the given list.
    """
    pairs = []
    for i in range(len(impostor_names)):
        for j in range(i + 1, len(impostor_names)):
            pairs.append((impostor_names[i], impostor_names[j]))
    return pairs
