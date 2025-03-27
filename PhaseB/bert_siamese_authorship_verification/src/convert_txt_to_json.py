import os
import json
import random


def convert_texts_to_json(shakespeare_dir, impostor_dir, output_path):
    dataset = []
    shakespeare_dir = os.path.join(os.getcwd(), shakespeare_dir)
    impostor_dir = os.path.join(os.getcwd(), impostor_dir)
    output_path = os.path.join(os.getcwd(), output_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process Shakespeare texts
    for filename in os.listdir(shakespeare_dir):
        with open(os.path.join(shakespeare_dir, filename), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            dataset.append({"text1": text, "text2": text, "label": 1})  # Same author

    # Process impostors
    for impostor_folder in os.listdir(impostor_dir):
        impostor_path = os.path.join(impostor_dir, impostor_folder)
        if os.path.isdir(impostor_path):
            for filename in os.listdir(impostor_path):
                with open(os.path.join(impostor_path, filename), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    dataset.append({"text1": text, "text2": text, "label": 0})  # Different authors

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, indent=4)


def convert_texts_to_json_with_limits(shakespeare_dir, impostor_dir, output_path, shakespeare_size, impostor_size):
    dataset = []
    shakespeare_dir = os.path.join(os.getcwd(), shakespeare_dir)
    impostor_dir = os.path.join(os.getcwd(), impostor_dir)
    output_path = os.path.join(os.getcwd(), output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Process Shakespeare texts ---
    shakespeare_files = os.listdir(shakespeare_dir)
    random.shuffle(shakespeare_files)
    shakespeare_files = shakespeare_files[:shakespeare_size]

    for filename in shakespeare_files:
        file_path = os.path.join(shakespeare_dir, filename)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            dataset.append({"text1": text, "text2": text, "label": 1})  # Same author

    # --- Process impostor texts ---
    all_impostor_files = []
    for impostor_folder in os.listdir(impostor_dir):
        folder_path = os.path.join(impostor_dir, impostor_folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                all_impostor_files.append(os.path.join(folder_path, filename))

    random.shuffle(all_impostor_files)
    selected_impostor_files = all_impostor_files[:impostor_size]

    for file_path in selected_impostor_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            dataset.append({"text1": text, "text2": text, "label": 0})  # Different authors

    # --- Save to JSON ---
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, indent=4)


if __name__ == "__main__":
    convert_texts_to_json("../data/raw/shakespeare", "../data/raw/impostors", "../data/processed/dataset.json")
