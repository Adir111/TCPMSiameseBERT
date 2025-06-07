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
    _instance = None

    def __new__(cls, config):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance._initialized = False  # Track initialization
        return cls._instance

    def __init__(self, config):
        if self._initialized:
            return  # Avoid reinitialization on repeated calls

        self._config = config
        self.data_path = (Path(__file__).parent.parent / config['data']['organised_data_folder_path']).resolve()
        self.shakespeare_dataset_name = config['data']['shakespeare_data_source']
        self.impostor_dataset_name = config['data']['impostors_data_source']
        self.text_to_classify_name = config['data']['classify_text_data_source']
        self.all_impostors_dataset_name = config['data']['all_impostors_data_source']
        self.pairs = config['data']['pairs']
        self.signals_folder = config['data']['signals_folder_name']
        self.distance_folder = config['data']['dtw']['output_distance_folder']
        self.dtw_file_name = config['data']['dtw']['dtw_file_name']
        self.included_text_names_file_name = config['data']['dtw']['included_text_names_file_name']
        self.all_isolation_forest_scores = config['data']['isolation_forest']['all_models_scores_file_name']
        self.signals_file_name = config['data']['dtw']['signals_file_name']

        self._initialized = True  # Prevent reinitialization

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
        Load the text from JSON and return it split by lines (raw, uncleaned).
        This is used for matching anomaly line names.
        """
        data = load_json_data(self.data_path, self.text_to_classify_name)
        raw_text = data.get('text', '')
        return [line.strip() for line in raw_text.split('\n') if line.strip()]

    def get_pairs(self):
        """
        Load and return pairs of impostor names.
        """
        data = load_json_data(self.data_path, self.pairs)
        return data

    def get_model_signals(self, model_name):
        """
        Load signal data for a specific model from its JSON file.
        """
        file_name = f"{model_name}-signals.json"
        path = self.data_path / self.signals_folder

        data = load_json_data(path, file_name)
        return data

    def get_shakespeare_included_text_names(self, model_name):
        """
        Load shakespeare included (in DTW) text names from JSON.
        """
        file_name = self.included_text_names_file_name
        path = self.data_path / self.distance_folder / model_name

        data = load_json_data(path, file_name)
        return data

    def get_dtw(self, model_name):
        """
        Load DTW distance matrix from JSON for given model.
        """
        file_name = self.dtw_file_name
        path = self.data_path / self.distance_folder / model_name

        data = load_json_data(path, file_name)
        return data

    def get_isolation_forest_results(self):
        """
        Load all models isolation forest score
        """
        data = load_json_data(self.data_path, self.all_isolation_forest_scores)
        return data
