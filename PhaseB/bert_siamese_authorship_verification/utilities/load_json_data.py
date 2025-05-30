import json


def load_json_data(data_path, file_name):
    """
    Utility method to load data from a JSON file.
    """
    path = data_path / file_name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
