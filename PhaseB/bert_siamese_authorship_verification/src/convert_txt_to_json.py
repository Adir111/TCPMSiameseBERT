import os
import json

from config.get_config import get_config


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
            tested_collection.append(text)
            if tested_collection_size is not None and len(tested_collection) == tested_collection_size:
                break

    # Process impostors
    for impostor_folder in os.listdir(impostor_dir):
        impostor_path = os.path.join(impostor_dir, impostor_folder)
        if os.path.isdir(impostor_path):
            for filename in os.listdir(impostor_path):
                with open(os.path.join(impostor_path, filename), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    impostor_dataset.append({"text1": text, "text2": text})
                    if impostor_size is not None and len(impostor_dataset) == impostor_size:
                        break
        if impostor_size is not None and len(impostor_dataset) == impostor_size:
            break

    # Save to JSON
    with open(impostor_output_path, "w", encoding="utf-8") as json_file:
        json.dump(impostor_dataset, json_file, indent=4)
    with open(tested_collection_output_path, "w", encoding="utf-8") as json_file:
        json.dump(tested_collection, json_file, indent=4)


if __name__ == "__main__":
    convert_texts_to_json("../data/raw/shakespeare", "../data/raw/impostors", "../data/processed/dataset.json")
