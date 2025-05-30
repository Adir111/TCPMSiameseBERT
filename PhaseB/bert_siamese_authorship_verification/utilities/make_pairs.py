def make_pairs(list):
    """
    Generate all possible pairs of a given list.
    """
    pairs = []
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            pairs.append((list[i], list[j]))
    return pairs
