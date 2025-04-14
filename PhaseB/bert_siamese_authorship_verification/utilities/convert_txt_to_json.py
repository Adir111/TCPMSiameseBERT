import os
import json

from config_loader import get_config


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
        for j in range(i + 1, len(impostor_folders)):
            if impostor_size is not None and len(impostor_dataset) >= impostor_size:
                break

            impostor_1_folder = impostor_folders[i]
            impostor_2_folder = impostor_folders[j]

            impostor_1_path = os.path.join(impostor_dir, impostor_1_folder)
            impostor_2_path = os.path.join(impostor_dir, impostor_2_folder)

            impostor_1_files = os.listdir(impostor_1_path)
            impostor_2_files = os.listdir(impostor_2_path)

            for file1 in impostor_1_files:
                for file2 in impostor_2_files:
                    if impostor_size is not None and len(impostor_dataset) >= impostor_size:
                        break

                    path1 = os.path.join(impostor_1_path, file1)
                    path2 = os.path.join(impostor_2_path, file2)

                    with open(path1, "r", encoding="utf-8", errors="ignore") as f1, \
                            open(path2, "r", encoding="utf-8", errors="ignore") as f2:

                        text1 = f1.read()
                        text2 = f2.read()

                        impostor_dataset.append({
                            "text1": text1,
                            "text2": text2,
                            "pair_name": f"{file1}_vs_{file2}"
                        })
                if impostor_size is not None and len(impostor_dataset) >= impostor_size:
                    break

    # Save to JSON
    with open(impostor_output_path, "w", encoding="utf-8") as json_file:
        json.dump(impostor_dataset, json_file, indent=4)
    with open(tested_collection_output_path, "w", encoding="utf-8") as json_file:
        json.dump(tested_collection, json_file, indent=4)


if __name__ == "__main__":
    convert_texts_to_json(
        "../data/raw/shakespeare",
        "../data/raw/impostors",
        "../data/processed/dataset.json")
