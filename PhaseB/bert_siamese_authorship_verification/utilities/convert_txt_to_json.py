import json
from pathlib import Path

def save_to_json(data, output_path, data_name):
    """
    Save the given data to a JSON file and print the status.

    Parameters:
    - data: The data to be saved (dict or list)
    - output_path: The path to the JSON file where data will be saved
    - data_name: A name or description for the data being saved (used for logging)
    """
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"{data_name} saved to {output_path}.")

def convert_texts_to_json(config, shakespeare_dir, impostor_dir, shakespeare_collection_size=None, impostor_size=None):
    impostor_dataset = []
    shakespeare_collection = []
    classify_text = {}

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

    # Process the 'text to classify.txt'
    classify_file_path = shakespeare_dir / 'text to classify.txt'
    if not classify_file_path.exists():
        raise FileNotFoundError(f"Error: 'text to classify.txt' is missing in the directory {shakespeare_dir}")

    with classify_file_path.open("r", encoding="utf-8", errors="ignore") as f:
        classify_text["text_name"] = 'text to classify.txt'
        classify_text["text"] = f.read()

    # Process Shakespeare texts
    for filename in shakespeare_dir.iterdir():
        if shakespeare_collection_size is not None and len(shakespeare_collection) == shakespeare_collection_size:
            break
        if filename.name == 'text to classify.txt':
            continue

        with filename.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            shakespeare_collection.append({
                "text_name": filename.name,
                "text": text
            })

    print(f"Handled {len(shakespeare_collection)} Shakespeare texts.")

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
        print(f"Handled {len(impostor_texts)} {impostor_folder.name} texts.")
        count_impostors_texts += len(impostor_texts)

    print(f"Handled {len(impostor_dataset)} impostor authors, with a total of {count_impostors_texts} texts.")

    # Save to JSON
    save_to_json(impostor_dataset, impostor_output_path, "Impostor dataset")
    save_to_json(shakespeare_collection, shakespeare_collection_output_path, "Shakespeare collection")
    save_to_json(classify_text, classify_text_output_path, "Text to classify data")

