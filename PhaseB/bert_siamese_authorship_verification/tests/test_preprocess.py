import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import TextPreprocessor


def test_chunking_basic():
    text = "This is a simple sentence used for chunk testing."
    preprocessor = TextPreprocessor()
    chunks = preprocessor.divide_into_chunk(text, chunk_size=4)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk.split()) <= 4 for chunk in chunks), "Each chunk should have <= 4 words"


def test_chunking_exact_length():
    text = "one two three four five six"
    preprocessor = TextPreprocessor()
    chunks = preprocessor.divide_into_chunk(text, chunk_size=2)

    assert len(chunks) == 3, "Should split into 3 chunks of size 2"
    assert chunks == ['one two', 'three four', 'five six']


def test_chunking_with_remainder():
    text = "one two three four five"
    preprocessor = TextPreprocessor()
    chunks = preprocessor.divide_into_chunk(text, chunk_size=2)

    assert len(chunks) == 3
    assert chunks[-1] == "five", "Final chunk should contain the remainder word"


def test_chunk_pair_splits_equally():
    text1 = "A B C D E F"
    text2 = "1 2 3 4 5 6"
    preprocessor = TextPreprocessor()
    chunk_pairs = preprocessor.divide_into_chunk_pair(text1, text2, chunk_size=2)

    assert isinstance(chunk_pairs, list)
    assert all(isinstance(pair, tuple) for pair in chunk_pairs)
    assert all(len(pair) == 2 for pair in chunk_pairs)
    assert chunk_pairs[0] == ('A B', '1 2')
