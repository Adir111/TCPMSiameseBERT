"""
Utility function to generate all unique pairs from a list.
"""


def make_pairs(list):
    """
    Generate all possible pairs of a given list.

    Parameters:
    - list: A list of items

    Returns:
    - A list of tuple pairs, each containing two distinct items from the input list
    """
    pairs = []
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            pairs.append((list[i], list[j]))
    return pairs
