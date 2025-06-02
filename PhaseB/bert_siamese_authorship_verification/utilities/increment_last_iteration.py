from .dataset_manager import save_to_json
from .load_json_data import load_json_data
from pathlib import Path


def increment_last_iteration(config, is_training=True):
    """
    Increment the 'last_iteration' field in the pairs JSON file.
    Set is_training to True (default) for training iteration, or to False for signal iteration.
    Returns the updated iteration number.
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
