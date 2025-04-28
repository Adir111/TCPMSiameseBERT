import json
from pathlib import Path

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

    # Process impostors
    impostor_folders = [f for f in impostor_dir.iterdir() if f.is_dir()]

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

    # Save to JSON
    with impostor_output_path.open("w", encoding="utf-8") as json_file:
        json.dump(impostor_dataset, json_file, indent=4)
    with shakespeare_collection_output_path.open("w", encoding="utf-8") as json_file:
        json.dump(shakespeare_collection, json_file, indent=4)
    with classify_text_output_path.open("w", encoding="utf-8") as classify_file:
        json.dump(classify_text, classify_file, indent=4)
