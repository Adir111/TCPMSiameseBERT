from pathlib import Path
import re

from PhaseB.bert_siamese_authorship_verification.utilities import load_json_data


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
    def __init__(self, config):
        self._config = config
        self.data_path = (Path(__file__).parent.parent / config['data']['organised_data_folder_path']).resolve()
        self.shakespeare_dataset_name = config['data']['shakespeare_data_source']
        self.impostor_dataset_name = config['data']['impostors_data_source']
        self.text_to_classify_name = config['data']['classify_text_data_source']
        self.all_impostors_dataset_name = config['data']['all_impostors_data_source']
        self.pairs = config['data']['pairs']

    def get_shakespeare_data(self):
        """
        Load and return the Shakespeare dataset from JSON.
        """
        data = load_json_data(self.data_path, self.shakespeare_dataset_name)
        return [{
            "text_name": item["text_name"],
            "text": _clean_text(item["text"])
        } for item in data]

    def get_impostor_texts_by_name(self, name):
        """
        Load and return the texts of a specific impostor by name.
        """
        impostors = load_json_data(self.data_path, self.impostor_dataset_name)
        for impostor in impostors:
            if impostor["author"] == name:
                return [_clean_text(text) for text in impostor["texts"]]
        raise ValueError(f"Impostor with name '{name}' not found.")

    def get_impostors_name_list(self):
        """
        Return a list of all impostor names.
        """
        impostors = load_json_data(self.data_path, self.impostor_dataset_name)
        return [impostor['author'] for impostor in impostors]

    def get_all_impostors_data(self):
        """
        Load and return all impostor data from JSON.
        """
        impostors = load_json_data(self.data_path, self.all_impostors_dataset_name)
        all_impostors = []

        for impostor in impostors:
            texts = []
            for text in impostor["texts"]:
                texts.append(_clean_text(text))
            all_impostors.append({
                "author": impostor["author"],
                "texts": texts
            })
        return all_impostors

    def get_text_to_classify(self):
        """
        Load and return the text to classify from JSON.
        """
        data = load_json_data(self.data_path, self.text_to_classify_name)
        return _clean_text(data.get('text', ''))

    def get_pairs(self, impostor_names):
        """
        Load and return pairs of impostor names.
        - If the pairs file exists, load and return it.
        - Otherwise, generate the pairs, save with last_iteration=0, and return the object.
        """
        data = load_json_data(self.data_path, self.pairs)
        return data
