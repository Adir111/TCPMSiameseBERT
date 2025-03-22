import json
from transformers import BertTokenizer


class DataLoader:
    def __init__(self, data_path, max_length=512):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        pairs = [(entry["text1"], entry["text2"], entry["label"]) for entry in data]
        return pairs

    def tokenize_pairs(self, text1, text2):
        tokens = self.tokenizer(text1, text2, padding="max_length",
                                truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokens["input_ids"], tokens["attention_mask"]
