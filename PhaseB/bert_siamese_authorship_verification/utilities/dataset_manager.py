"""
Dataset Manager module for processing and converting text datasets into JSON format.

This module supports:
- Handling impostor text datasets organized in folders per author
- Handling Shakespeare texts including a special "text to classify.txt"
- Generating pairs data for impostor authors using make_pairs function
- Saving datasets and metadata to JSON files
- Logging progress and errors using configured logger
"""

import json
from pathlib import Path

from .make_pairs import make_pairs
from .config_loader import get_config
from .logger import get_logger

config = get_config()
logger = get_logger(config)


def save_to_json(data, output_path, data_name):
    """
    Save the given data to a JSON file and log the status.

    Parameters:
    - data: The data to be saved (dict or list)
    - output_path: The path to the JSON file where data will be saved
    - data_name: A name or description for the data being saved (used for logging)
    """
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"{data_name} saved to {output_path}.")


def handle_impostor_texts(impostor_dir, impostor_size=None):
    """
    Process impostor dataset directory structure and load text data.

    Parameters:
    - impostor_dir: Path to directory containing impostor author folders
    - impostor_size: Optional maximum number of impostor authors to process

    Returns:
    - List of dicts, each with keys "author" (folder name) and "texts" (list of strings)
    """
    impostor_dataset = []

    # Process impostors
    impostor_folders = [f for f in impostor_dir.iterdir() if f.is_dir()]
    count_impostors_texts = 0

    for i in range(len(impostor_folders)):
        if impostor_size is not None and len(impostor_dataset) >= impostor_size:
            break

        impostor_folder = impostor_folders[i]
        impostor_files = impostor_folder.iterdir()

        impostor_texts = []

        for file in impostor_files:
            if file.is_file():
                with file.open("r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    impostor_texts.append(text)
            else:
                raise ValueError(f"Expected a file, but found a directory or unsupported file type: {file}")

        impostor_dataset.append({
            "author": impostor_folder.name,
            "texts": impostor_texts
        })
        logger.info(f"Handled {len(impostor_texts)} {impostor_folder.name} texts.")
        count_impostors_texts += len(impostor_texts)

    logger.info(f"Handled {len(impostor_dataset)} impostor authors, with a total of {count_impostors_texts} texts.")
    return impostor_dataset


def handle_shakespeare_texts(shakespeare_dir, shakespeare_collection_size=None):
    """
    Load Shakespeare texts and a special classification text.

    Parameters:
    - shakespeare_dir: Path to Shakespeare dataset directory
    - shakespeare_collection_size: Optional max number of Shakespeare texts to load

    Returns:
    - Tuple: (list of Shakespeare text dicts, dict with classification text)
      Each Shakespeare dict has keys "text_name" and "text"
      Classification text dict has keys "text_name" and "text"
    """
    classify_text = {}
    shakespeare_collection = []
    classify_text_file_name = "text to classify.txt"

    # Process the 'text to classify.txt'
    classify_file_path = shakespeare_dir / classify_text_file_name
    if not classify_file_path.exists():
        error_msg = f"Error: {classify_text_file_name} is missing in the directory {shakespeare_dir}"
        raise FileNotFoundError(error_msg)

    with classify_file_path.open("r", encoding="utf-8", errors="ignore") as f:
        classify_text["text_name"] = classify_text_file_name
        classify_text["text"] = f.read()

    # Process Shakespeare texts
    for filename in shakespeare_dir.iterdir():
        if shakespeare_collection_size is not None and len(shakespeare_collection) == shakespeare_collection_size:
            break
        if filename.name == classify_text_file_name:
            continue

        with filename.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            shakespeare_collection.append({
                "text_name": filename.name,
                "text": text
            })

    logger.info(f"Handled {len(shakespeare_collection)} Shakespeare texts.")
    return shakespeare_collection, classify_text


def __generate_and_save_pairs(impostor_dataset):
    """
    Generate impostor author pairs and save them to JSON.

    Parameters:
    - impostor_dataset: List of impostor author dicts (from handle_impostor_texts)

    Uses configuration paths to determine output location.
    """
    data_sources_folder = Path(config['data']['organised_data_folder_path'])
    pairs_output_path = data_sources_folder / config['data']['pairs']
    impostor_names = [author["author"] for author in impostor_dataset]
    pairs_data = {
        "last_iteration_training": 0,
        "last_iteration_signal": 0,
        "models_to_skip": [],
        "pairs": make_pairs(impostor_names)
    }
    save_to_json(pairs_data, pairs_output_path, "Impostor pairs")


def convert_texts_to_json(shakespeare_dir, impostor_dir, shakespeare_collection_size=None, impostor_size=None):
    """
    Convert Shakespeare and impostor datasets from raw texts to JSON files.

    Parameters:
    - shakespeare_dir: Directory path of Shakespeare texts
    - impostor_dir: Directory path of impostor texts
    - shakespeare_collection_size: Optional max number of Shakespeare texts to process
    - impostor_size: Optional max number of impostor authors to process

    Saves multiple JSON files including pairs, impostor dataset,
    Shakespeare collection, and classification text.
    """
    data_sources_folder = Path(config['data']['organised_data_folder_path'])
    shakespeare_collection_output_path = data_sources_folder / config['data']['shakespeare_data_source']
    impostors_output_path = data_sources_folder / config['data']['impostors_data_source']
    classify_text_output_path = data_sources_folder / config['data']['classify_text_data_source']

    shakespeare_dir = Path(shakespeare_dir)
    impostor_dir = Path(impostor_dir)
    impostor_output_path = Path(impostors_output_path)
    shakespeare_collection_output_path = Path(shakespeare_collection_output_path)
    classify_text_output_path = Path(classify_text_output_path)

    # Ensure directory exists
    impostor_output_path.parent.mkdir(parents=True, exist_ok=True)
    shakespeare_collection_output_path.parent.mkdir(parents=True, exist_ok=True)
    classify_text_output_path.parent.mkdir(parents=True, exist_ok=True)

    shakespeare_collection, classify_text = handle_shakespeare_texts(shakespeare_dir, shakespeare_collection_size)
    impostor_dataset = handle_impostor_texts(impostor_dir, impostor_size)

    # Save to JSON
    save_to_json(impostor_dataset, impostor_output_path, "Impostor dataset")
    save_to_json(shakespeare_collection, shakespeare_collection_output_path, "Shakespeare collection")
    save_to_json(classify_text, classify_text_output_path, "Text to classify data")
    __generate_and_save_pairs(impostor_dataset)


def convert_all_impostor_texts_to_json(impostor_dir):
    """
    Convert all impostor texts to JSON without limiting size.

    Parameters:
    - impostor_dir: Directory path containing impostor author folders

    Saves the entire impostor dataset to JSON.
    """
    data_sources_folder = Path(config['data']['organised_data_folder_path'])
    impostors_output_path = data_sources_folder / config['data']['all_impostors_data_source']

    impostor_dir = Path(impostor_dir)
    impostor_output_path = Path(impostors_output_path)
    impostor_output_path.parent.mkdir(parents=True, exist_ok=True)
    impostor_dataset = handle_impostor_texts(impostor_dir)

    save_to_json(impostor_dataset, impostor_output_path, "All Impostors Dataset")
