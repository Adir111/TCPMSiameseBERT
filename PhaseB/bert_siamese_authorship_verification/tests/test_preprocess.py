import sys
import os
import unittest
from tokenizers import BertWordPieceTokenizer
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import TextPreprocessor


class DummyConfig:
    def __init__(self):
        self.config = {
            'bert': {
                'maximum_sequence_length': 8,
                "vocab_path": "tokenizer_vocab/bert-base-uncased.txt"
            },
            'training': {
                'impostor_chunk_ratio': 1
            }
        }


class TextPreprocessorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DummyConfig().config
        cls.preprocessor = TextPreprocessor(cls.config)

    def test_tokenize_text(self):
        text = "Hello, world! This is a test."
        tokens = self.preprocessor.tokenize_text(text)
        # Strip punctuation from expected tokens for BERT tokenization
        expected_tokens_clean = self.preprocessor.tokenizer.encode("Hello, world! This is a test.").tokens
        self.assertEqual(tokens, expected_tokens_clean)

    def test_divide_tokens_into_chunks(self):
        tokens = [f"token{i}" for i in range(10)]
        chunk_size = 4
        chunks = self.preprocessor.divide_tokens_into_chunks(tokens, chunk_size)
        expected = [
            ['token0', 'token1', 'token2', 'token3'],
            ['token4', 'token5', 'token6', 'token7'],
            ['token8', 'token9', '[PAD]', '[PAD]']
        ]
        self.assertEqual(len(chunks), 3)
        self.assertListEqual(chunks[-1].tolist(), expected[-1])

    def test_encode_tokenized_chunks(self):
        token_chunks = [
            ['hello', 'world'],
            ['test', 'sentence', 'two']
        ]
        max_len = self.config['bert']['maximum_sequence_length']
        encoded = self.preprocessor.encode_tokenized_chunks(token_chunks)

        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)

        input_ids = encoded['input_ids'].numpy()
        attention_mask = encoded['attention_mask'].numpy()

        self.assertEqual(input_ids.shape, (2, max_len))
        self.assertEqual(attention_mask.shape, (2, max_len))

        # Padding check (assuming max_len = 8)
        self.assertTrue(np.all(attention_mask[0][4:] == 0))
        self.assertTrue(np.all(attention_mask[1][5:] == 0))


if __name__ == '__main__':
    unittest.main()
