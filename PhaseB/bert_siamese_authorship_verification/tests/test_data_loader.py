import sys
import os
import json
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DataLoader

def write_temp_file(data, suffix=".json"):
    temp = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=suffix)
    json.dump(data, temp)
    temp.close()
    return temp.name


def test_load_cleaned_text_pair_with_complex_text():
    complex_json = [
        {
            "text1": "Author1's text:\nThis is line one!\nThis is line two...\n",
            "text2": "Author2's text:\n\tIndentations? Punctuation? Yes!",
            "pair_name": "author1_vs_author2"
        }
    ]
    file_path = write_temp_file(complex_json)
    loader = DataLoader(file_path)

    pairs = loader.load_impostors()
    os.unlink(file_path)

    assert len(pairs) == 1
    t1, t2, pair_name = pairs[0]
    assert isinstance(t1, str) and isinstance(t2, str)
    assert isinstance(pair_name, str)
    assert "\n" not in t1 and "\n" not in t2  # cleaned!
    assert "line one" in t1.lower()
    assert "punctuation" in t2.lower()


def test_load_cleaned_text_single_file_with_formatting():
    complex_texts = [
        "Shakespeare wrote:\n\nTo be, or not to be... 💭\nThat is the question!",
        "Some prose with... dashes -- and ellipses...\nNewlines and 🐍 emojis!"
    ]
    file_path = write_temp_file(complex_texts)
    loader = DataLoader(file_path)

    cleaned = loader.load_tested_collection_text()
    os.unlink(file_path)

    assert len(cleaned) == 2
    assert all(isinstance(c, str) for c in cleaned)
    assert "to be or not to be" in cleaned[0].lower().replace(",", "").replace(".", "")
    assert "emojis" in cleaned[1].lower()


def test_load_cleaned_text_directory_with_complex_entries(tmp_path):
    content1 = ["A\nB\nC -- line one. Line two!!"]
    content2 = ["Another file.\nWith more… lines, and emoji 🤖"]

    (tmp_path / "f1.json").write_text(json.dumps(content1))
    (tmp_path / "f2.json").write_text(json.dumps(content2))

    loader = DataLoader(str(tmp_path))
    cleaned = loader.load_tested_collection_text()

    assert len(cleaned) == 2
    assert all(isinstance(c, str) for c in cleaned)
    assert "line one" in cleaned[0].lower()
    assert "emoji" in cleaned[1].lower()