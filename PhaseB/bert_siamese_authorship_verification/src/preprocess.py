import re
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
from bert_siamese_authorship_verification.config.get_config import get_config

# Load config
config = get_config()


class TextPreprocessor:
    def __init__(self, max_length=config['bert']['maximum_sequence_length']):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    def tokenize_pairs(self, text1, text2):
        tokens = self.tokenizer(
            text1, text2,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
            return_overflowing_tokens=False
        )
        return tokens["input_ids"], tokens["attention_mask"]
