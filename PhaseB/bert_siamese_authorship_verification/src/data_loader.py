import json
from bert_siamese_authorship_verification.src.preprocess import TextPreprocessor


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        return [(self.preprocessor.clean_text(entry["text1"]), self.preprocessor.clean_text(entry["text2"]),
                 entry["label"]) for entry in data]
