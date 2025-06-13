"""
Provides a function to increment iteration counters in the impostor pairs JSON.

Supports incrementing either the training iteration or the signal iteration counter,
and persists the updated data to the pairs JSON file.
"""

from .dataset_manager import save_to_json
from .load_json_data import load_json_data
from pathlib import Path


def increment_last_iteration(config, is_training=True):
    """
    Increment the 'last_iteration' field in the pairs JSON file.

    Parameters:
    - config: Configuration dictionary with data paths
    - is_training: Boolean flag, True to increment training iteration, False for signal iteration

    Returns:
    - The updated iteration number (int)
    """
    data_path = (Path(__file__).parent.parent / config['data']['organised_data_folder_path']).resolve()
    pairs = config['data']['pairs']
    pairs_file = data_path / pairs
    data = load_json_data(data_path, pairs)

    if is_training:
        data["last_iteration_training"] += 1
        ret = data["last_iteration_training"]
    else:
        data["last_iteration_signal"] += 1
        ret = data["last_iteration_signal"]

    save_to_json(data, pairs_file, "Updated impostor pairs iteration")
    return ret
