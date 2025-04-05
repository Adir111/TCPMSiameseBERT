import random
import re
import nltk
import numpy as np
from transformers import BertTokenizer
from config.get_config import get_config

# Load config
config = get_config()


class TextPreprocessor:
    def __init__(self, max_length=config['bert']['maximum_sequence_length']):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        try:
            from nltk.corpus import wordnet, stopwords
            wordnet.ensure_loaded()  # This checks if it can be used
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('stopwords')

    def clean_text(self, text):
        # Split text into words, convert to lowercase, remove punctuation from each word, remove non-alphabetic words and filter out stop words
        text = text.lower()
        text = text.replace("\n", " ")  # Replace newlines with spaces
        text = text.replace("\r", " ")  # Replace carriage returns with spaces
        text = text.replace("\t", " ")  # Replace tabs with spaces
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space

        words = text.split()
        # Remove non-alphabetic characters
        words = [word for word in words if word.isalpha()]

        # Remove Stopwords
        stop_words = set(nltk.corpus.stopwords.words("english"))
        words = [word for word in words if word not in stop_words]

        # Re-concatenate words into a text
        return " ".join(words)

    @staticmethod
    def divide_into_chunk_pair(text_1, text_2, chunk_size=20):
        config = get_config()
        chunk_ratio = config['training']['impostor_chunk_ratio']
        words_1 = text_1.split()
        words_2 = text_2.split()

        # Find minimum length to avoid index errors
        min_length = min(len(words_1), len(words_2))
        words_1 = chunk_ratio * words_1[:min_length]
        words_2 = chunk_ratio * words_2[:min_length]

        # Create chunks
        chunk_pair = []
        for i in range(0, min_length, chunk_size):
            chunk_1 = " ".join(words_1[i:i + chunk_size])
            chunk_2 = " ".join(words_2[i:i + chunk_size])
            chunk_pair.append((chunk_1, chunk_2))
        return chunk_pair

    @staticmethod
    def create_model_x_y(chunks_imp_1, chunks_imp_2):
        config = get_config()
        chunk_ratio = config['training']['impostor_chunk_ratio']
        chunks = [chunks_imp_1, chunks_imp_2]

        lens = [len(chunks[0]), len(chunks[1])]
        max_idx = lens.index(max(lens))
        min_idx = lens.index(min(lens))

        even_sized_chunks = []
        for i in range(lens[max_idx] // lens[min_idx]):
            even_sized_chunks = even_sized_chunks + chunks_imp_2[min_idx]
        chunks[min_idx] = even_sized_chunks + random.sample(chunks[min_idx], lens[max_idx] - len(even_sized_chunks))
        chunks[0] *= chunk_ratio
        chunks[1] *= chunk_ratio

        y_labels = [0] * len(chunks[0]) + [1] * len(chunks[1])
        x_labels = [y for x in [chunks[0], chunks[1]] for y in x]

        return np.asarray(x_labels), np.asarray(y_labels), chunks

    @staticmethod
    def divide_into_chunk(text, chunk_size=20):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def divide_tokens_into_chunks(tokens, chunk_size):
        tokens = np.asarray(tokens)
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)
        chunks = np.asarray(chunks)
        return chunks

    def tokenize_chunk(self, chunk):
        token = self.tokenizer(
            chunk,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
            return_overflowing_tokens=False
        )
        return token["input_ids"], token["attention_mask"]

    def tokenize_text(self, text):
        tokens = self.tokenizer.tokenize(
            text,
            max_length=self.max_length
        )
        return tokens
