import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.config_loader import get_config


def convert_texts_to_json(shakespeare_dir, impostor_dir, tested_collection_size=None, impostor_size=None):
    config = get_config()

    impostor_dataset = []
    tested_collection = []

    tested_collection_output_path = config['data']['processed_tested_path']
    impostors_output_path = config['data']['processed_impostors_path']

    shakespeare_dir = os.path.join(os.getcwd(), shakespeare_dir)
    impostor_dir = os.path.join(os.getcwd(), impostor_dir)
    impostor_output_path = os.path.join(os.getcwd(), impostors_output_path)
    tested_collection_output_path = os.path.join(os.getcwd(), tested_collection_output_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(impostor_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(tested_collection_output_path), exist_ok=True)

    # Process Shakespeare texts
    for filename in os.listdir(shakespeare_dir):
        with open(os.path.join(shakespeare_dir, filename), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            tested_collection.append({
                "text_name": filename,
                "text": text
            })
            if tested_collection_size is not None and len(tested_collection) == tested_collection_size:
                break

    # Process impostors
    impostor_folders = [f for f in os.listdir(impostor_dir) if os.path.isdir(os.path.join(impostor_dir, f))]

    for i in range(len(impostor_folders)):
        if impostor_size is not None and len(impostor_dataset) >= impostor_size:
            break

        impostor_folder = impostor_folders[i]
        impostor_path = os.path.join(impostor_dir, impostor_folder)
        impostor_files = os.listdir(impostor_path)

        impostor_texts = []

        for file in impostor_files:
            path = os.path.join(impostor_path, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                impostor_texts.append(text)

        impostor_dataset.append({
            "author": impostor_folder,
            "texts": impostor_texts
        })

    # Save to JSON
    with open(impostor_output_path, "w", encoding="utf-8") as json_file:
        json.dump(impostor_dataset, json_file, indent=4)
    with open(tested_collection_output_path, "w", encoding="utf-8") as json_file:
        json.dump(tested_collection, json_file, indent=4)


if __name__ == "__main__":
    # Load config
    config = get_config()

    # Paths
    TESTED_COLLECTION_PATH = '../' + config['data']['shakespeare_path']
    IMPOSTORS_PATH = '../' + config['data']['impostors_path']
    convert_texts_to_json(
        TESTED_COLLECTION_PATH,
        IMPOSTORS_PATH,
        tested_collection_size=2,
        impostor_size=2)
