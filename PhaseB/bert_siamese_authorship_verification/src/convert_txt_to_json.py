import os
import json


def convert_texts_to_json(shakespeare_dir, impostor_dir, output_path):
    dataset = []

    # Process Shakespeare texts
    for filename in os.listdir(shakespeare_dir):
        with open(os.path.join(shakespeare_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
            dataset.append({"text1": text, "text2": text, "label": 1})  # Same author

    # Process impostors
    for impostor_folder in os.listdir(impostor_dir):
        impostor_path = os.path.join(impostor_dir, impostor_folder)
        if os.path.isdir(impostor_path):
            for filename in os.listdir(impostor_path):
                with open(os.path.join(impostor_path, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    dataset.append({"text1": text, "text2": text, "label": 0})  # Different authors

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, indent=4)


if __name__ == "__main__":
    convert_texts_to_json("../data/raw/shakespeare", "../data/raw/impostors", "../data/processed/dataset.json")
