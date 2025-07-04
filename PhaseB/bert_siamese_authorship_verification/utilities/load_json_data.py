"""
Utility function for loading JSON data from a specified file path.
"""

import json


def load_json_data(data_path, file_name):
    """
    Load data from a JSON file.

    Parameters:
    - data_path: Path object or string representing the directory of the JSON file
    - file_name: Name of the JSON file to load

    Returns:
    - Parsed JSON data as a Python object (dict or list)
    """
    path = data_path / file_name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
