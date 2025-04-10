import json
import os

from src.preprocess import TextPreprocessor


class DataLoader:
    def __init__(self, data_path, preprocessor: TextPreprocessor):
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', data_path))
        self.preprocessor = preprocessor

    def load_cleaned_text_pair(self):
        is_dir = os.path.isdir(self.data_path)
        if is_dir:
            raise Exception("Cannot load pair from directory, please provide a json file path")

        with open(self.data_path, "r") as f:
            data = json.load(f)
        return [(self.preprocessor.clean_text(entry["text1"]), self.preprocessor.clean_text(entry["text2"]), entry["pair_name"]) for entry in data]

    def load_cleaned_text(self):
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
