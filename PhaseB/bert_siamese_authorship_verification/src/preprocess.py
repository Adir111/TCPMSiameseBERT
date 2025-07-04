"""
Provides the Preprocessor class for preparing text data for Siamese BERT models.

Features include:
- Downloading necessary NLTK resources silently
- Tokenizing and chunking long texts into fixed-size chunks suitable for BERT
- Balancing and equalizing chunk lists between impostor datasets
- Creating paired input arrays (X) and labels (y) for training/testing Siamese models
- Supports singleton pattern to reuse tokenizer instances efficiently

Designed for use in text similarity and verification tasks with BERT-based Siamese networks.
"""

import nltk
import contextlib
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
import random

with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)

class Preprocessor:
    """
    Preprocessor class for preparing text data for Siamese BERT models.

    Supports tokenization, chunking long texts, balancing chunk lists,
    and creating paired input data for model training/testing.

    Implements a singleton pattern when no tokenizer is provided,
    otherwise creates a fresh instance.
    """

    _singleton_instance = None

    def __new__(cls, config, tokenizer=None):
        """
        Creates a singleton instance unless a tokenizer is explicitly provided.

        Parameters:
        - config (dict): Configuration dictionary
        - tokenizer (BertTokenizer or None): Optional tokenizer to use

        Returns:
        - instance (Preprocessor): Preprocessor instance
        """
        if tokenizer is None:
            if cls._singleton_instance is None:
                cls._singleton_instance = super(Preprocessor, cls).__new__(cls)
                cls._singleton_instance._initialized = False
            return cls._singleton_instance
        else:
            # Bypass singleton — return a fresh instance
            instance = super(Preprocessor, cls).__new__(cls)
            instance._initialized = False
            return instance


    def __init__(self, config, tokenizer=None):
        """
        Initializes the Preprocessor.

        Parameters:
        - config (dict): Configuration dictionary
        - tokenizer (BertTokenizer or None): Optional tokenizer to use
        """
        if self._initialized:
            return

        self.config = config
        self.chunk_size = config['model']['chunk_size']
        self.test_split = config['training']['test_split']

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(config['bert']['model'])
        else:
            self.tokenizer = tokenizer

        self._initialized = True


    def __tokenize_text(self, text):
        """
        Tokenize text into BERT input format.

        Parameters:
        - text (str): Input text

        Returns:
        - dict: Tokenized input with 'input_ids', 'attention_mask', 'token_type_ids'
        """
        encoded_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.chunk_size,
            return_tensors='tf'
        )
        return encoded_input


    def __handle_chunk(self, chunk_text, preprocessed_collection):
        """
        Tokenize a chunk of text and append it to the collection.

        Parameters:
        - chunk_text (str): Text chunk to tokenize
        - preprocessed_collection (list): List to append tokenized outputs to

        Returns:
        - int: Number of tokens in the chunk
        """
        tokenized = self.__tokenize_text(chunk_text)
        tokens_count = tokenized.data['input_ids'].shape[1]
        preprocessed_collection.append(tokenized)
        return tokens_count


    def preprocess(self, collection):
        """
        Preprocess a collection of text data.

        Tokenizes text data into BERT format. Splits text into chunks if longer
        than configured chunk size. Returns tokenized chunks and total token count.

        Parameters:
        - collection (list of str): List of input text strings

        Returns:
        - tuple:
          - preprocessed_collection (list): List of tokenized text dicts
          - tokens_count (int): Total number of tokens processed
        """
        preprocessed_collection = []
        tokens_count = 0

        for text in collection:
            # Tokenize the text first
            tokens = self.tokenizer.tokenize(text)

            if len(tokens) > self.chunk_size:
                # Split text into chunks of chunk_size words
                num_chunks = len(tokens) // self.chunk_size
                if len(tokens) % self.chunk_size != 0:
                    num_chunks += 1

                chunks = [tokens[i * self.chunk_size: (i + 1) * self.chunk_size] for i in range(num_chunks)]
                for chunk in chunks:
                    chunk_text = self.tokenizer.convert_tokens_to_string(chunk)
                    tokens_count += self.__handle_chunk(chunk_text, preprocessed_collection)
            else:
                # If no chunking is needed, directly tokenize the text
                tokens_count += self.__handle_chunk(text, preprocessed_collection)
        return preprocessed_collection, tokens_count


    def equalize_chunks(self, chunks_list):
        """
        Equalize the lengths of two chunk lists by repeating and sampling.

        This helps to balance the number of chunks between two impostor datasets
        based on a configured chunk ratio.

        Parameters:
        - chunks_list (list of lists): Two lists of tokenized chunks

        Returns:
        - list of lists: Balanced chunk lists of equal length
        """
        chunk_ratio = self.config['training']['impostor_chunk_ratio']

        lengths = [len(chunks_list[0]), len(chunks_list[1])]
        i1 = lengths.index(max(lengths))  # index of longer list
        i2 = lengths.index(min(lengths))  # index of shorter list

        # Repeat the smaller list enough times
        repeated_chunks = chunks_list[i2] * (lengths[i1] // lengths[i2])

        # Randomly sample the remaining elements needed
        remainder = lengths[i1] - len(repeated_chunks)
        if remainder > 0:
            repeated_chunks += random.sample(chunks_list[i2], remainder)

        chunks_list[i2] = repeated_chunks

        chunks_list[0] *= chunk_ratio
        chunks_list[1] *= chunk_ratio

        return chunks_list


    def create_xy(self, impostor_1_chunks, impostor_2_chunks, num_pairs=None):
        """
        Create paired input data (X) and labels (y) for training/testing.

        Generates positive pairs from chunks of the same impostor and
        negative pairs from chunks of different impostors. Splits data into
        train and test sets based on configuration.

        Parameters:
        - impostor_1_chunks (list): List of tokenized chunks for impostor 1
        - impostor_2_chunks (list): List of tokenized chunks for impostor 2
        - num_pairs (int or None): Optional limit on number of positive pairs to generate

        Returns:
        - tuple:
          - x_train (dict): Training inputs dict with keys for each BERT input
          - y_train (np.ndarray): Training labels (float32, shape [-1,1])
          - x_test (dict): Testing inputs dict
          - y_test (np.ndarray): Testing labels (float32, shape [-1,1])
        """
        if len(impostor_1_chunks) != len(impostor_2_chunks):
            raise ValueError("Chunk lists must be equal length for pairing.")

        # Prepare each input group
        ids_1, masks_1, types_1 = [], [], []
        ids_2, masks_2, types_2 = [], [], []
        labels = []

        def append_pair(c1, c2, label):
            ids_1.append(np.squeeze(c1["input_ids"].numpy(), axis=0))
            masks_1.append(np.squeeze(c1["attention_mask"].numpy(), axis=0))
            types_1.append(np.squeeze(c1["token_type_ids"].numpy(), axis=0))

            ids_2.append(np.squeeze(c2["input_ids"].numpy(), axis=0))
            masks_2.append(np.squeeze(c2["attention_mask"].numpy(), axis=0))
            types_2.append(np.squeeze(c2["token_type_ids"].numpy(), axis=0))

            labels.append(label)

        def generate_positive_pairs(chunks):
            n = len(chunks)
            if n < 2:
                return
            indices = list(range(n))
            random.shuffle(indices)
            limit = min(num_pairs or n // 2, n - 1)
            for i in range(limit):
                c1 = chunks[indices[i]]
                c2 = chunks[indices[i + 1]]
                append_pair(c1, c2, label=1)

        def generate_negative_pairs(chunks_a, chunks_b, count):
            for _ in range(count):
                c1 = random.choice(chunks_a)
                c2 = random.choice(chunks_b)
                append_pair(c1, c2, label=0)

        # Balanced generation
        generate_positive_pairs(impostor_1_chunks)
        generate_positive_pairs(impostor_2_chunks)

        total_positives = len(labels)
        generate_negative_pairs(impostor_1_chunks, impostor_2_chunks, total_positives)

        # Convert to arrays
        x_dict = {
            "input_ids_1": np.stack(ids_1),
            "attention_mask_1": np.stack(masks_1),
            "token_type_ids_1": np.stack(types_1),
            "input_ids_2": np.stack(ids_2),
            "attention_mask_2": np.stack(masks_2),
            "token_type_ids_2": np.stack(types_2),
        }
        y = np.array(labels, dtype=np.int32)

        # Split each field in x_dict independently
        x_train, x_test, y_train, y_test = {}, {}, None, None

        (
            x_train["input_ids_1"], x_test["input_ids_1"],
            x_train["attention_mask_1"], x_test["attention_mask_1"],
            x_train["token_type_ids_1"], x_test["token_type_ids_1"],
            x_train["input_ids_2"], x_test["input_ids_2"],
            x_train["attention_mask_2"], x_test["attention_mask_2"],
            x_train["token_type_ids_2"], x_test["token_type_ids_2"],
            y_train, y_test
        ) = train_test_split(
            x_dict["input_ids_1"], x_dict["attention_mask_1"], x_dict["token_type_ids_1"],
            x_dict["input_ids_2"], x_dict["attention_mask_2"], x_dict["token_type_ids_2"],
            y,
            test_size=self.test_split,
            random_state=42
        )

        y_train = y_train.astype(np.float32).reshape(-1, 1)
        y_test = y_test.astype(np.float32).reshape(-1, 1)

        return x_train, y_train, x_test, y_test
