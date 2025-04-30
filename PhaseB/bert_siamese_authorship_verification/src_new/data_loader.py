import json
from pathlib import Path
import re


def _clean_text(text):
    """
    Utility function to clean text by replacing unwanted characters.
    - Replaces newline, carriage return, and tab with spaces
    - Removes multiple spaces and replaces them with a single space
    """
    text = text.replace("\n", " ")  # Replace newlines with spaces
    text = text.replace("\r", " ")  # Replace carriage returns with spaces
    text = text.replace("\t", " ")  # Replace tabs with spaces
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()  # Strip leading and trailing spaces


class DataLoader:
    def __init__(self, data_path, shakespeare_dataset_name, impostor_dataset_name, text_to_classify_name):
        self.data_path = (Path(__file__).parent.parent / data_path).resolve()
        self.shakespeare_dataset_name = shakespeare_dataset_name
        self.impostor_dataset_name = impostor_dataset_name
        self.text_to_classify_name = text_to_classify_name

    def __load_json_data(self, file_name):
        """
        Utility method to load data from a JSON file.
        """
        path = self.data_path / file_name
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_shakespeare_data(self):
        """
        Load and return the Shakespeare dataset from JSON.
        """
        data = self.__load_json_data(self.shakespeare_dataset_name)
        return [_clean_text(item["text"]) for item in data]

    def get_impostor_texts_by_name(self, name):
        """
        Load and return the texts of a specific impostor by name.
        """
        impostors = self.__load_json_data(self.impostor_dataset_name)
        for impostor in impostors:
            if impostor["author"] == name:
                return [_clean_text(text) for text in impostor["texts"]]
        raise ValueError(f"Impostor with name '{name}' not found.")

    def get_impostors_name_list(self):
        """
        Return a list of all impostor names.
        """
        impostors = self.__load_json_data(self.impostor_dataset_name)
        return [impostor['author'] for impostor in impostors]

    def get_text_to_classify(self):
        """
        Load and return the text to classify from JSON.
        """
        data = self.__load_json_data(self.text_to_classify_name)
        return _clean_text(data.get('text', ''))
