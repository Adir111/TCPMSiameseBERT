import json
import os

from src.preprocess import TextPreprocessor


class DataLoader:
    def __init__(self, data_path, preprocessor: TextPreprocessor):
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_path))
        self.preprocessor = preprocessor

    def load_impostors(self):
        try:
            is_dir = os.path.isdir(self.data_path)
            if is_dir:
                raise Exception("Cannot load pair from directory, please provide a json file path")

            with open(self.data_path, "r") as f:
                data = json.load(f)
                data_len = len(data)

                cleaned_dataset = []
                for impostor in data:
                    cleaned_texts = []
                    for text in impostor["texts"]:
                        cleaned_texts.append(self.preprocessor.clean_text(text))

                    cleaned_dataset.append({
                        "author": impostor["author"],
                        "texts": cleaned_texts
                    })

                impostor_pairs = []
                for i in range(data_len):
                    for j in range(i + 1, data_len):
                        pair_name = f'{cleaned_dataset[i]["author"]}_vs_{cleaned_dataset[j]["author"]}'
                        impostor_pairs.append((
                            cleaned_dataset[i]["texts"],
                            cleaned_dataset[j]["texts"],
                            pair_name
                        ))

                return impostor_pairs
        except KeyError as e:
            raise KeyError(f"Missing expected key in JSON data while loading impostors: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading impostors: {e}")

    def load_tested_collection_text(self):
        is_dir = os.path.isdir(self.data_path)
        if is_dir:
            data = []
            for i, file in enumerate(os.listdir(self.data_path)):
                file_path = os.path.join(self.data_path, file)
                with open(file_path, "r") as f:
                    data.extend(json.load(f))
        else:
            with open(self.data_path, "r") as f:
                data = json.load(f)
        return [(entry["text_name"], self.preprocessor.clean_text(entry["text"])) for entry in data]
