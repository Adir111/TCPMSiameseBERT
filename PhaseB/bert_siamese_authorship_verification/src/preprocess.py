import re
from transformers import BertTokenizer


class TextPreprocessor:
    def __init__(self, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
        return text

    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True,
                              max_length=self.max_length, return_tensors="pt")
