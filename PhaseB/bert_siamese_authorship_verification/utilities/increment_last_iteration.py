from .dataset_manager import save_to_json
from .load_json_data import load_json_data
from pathlib import Path


def increment_last_iteration(config):
    """
    Increment the 'last_iteration' field in the pairs JSON file.
    Returns the updated iteration number.
    """
    data_path = (Path(__file__).parent.parent / config['data']['organised_data_folder_path']).resolve()
    pairs = config['data']['pairs']
    pairs_file = data_path / pairs
    data = load_json_data(data_path, pairs)

    if "last_iteration" not in data:
        data["last_iteration"] = 0

    data["last_iteration"] += 1
    save_to_json(data, pairs_file, "Updated impostor pairs iteration")
    return data["last_iteration"]
