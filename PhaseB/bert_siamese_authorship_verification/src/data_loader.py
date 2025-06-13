"""
Provides singleton DataLoader class for loading and cleaning
various text datasets used in authorship verification.
Includes utility function for cleaning raw text input.
"""

from pathlib import Path
import re

from PhaseB.bert_siamese_authorship_verification.utilities import load_json_data


def _clean_text(text):
    """
    Cleans the input text by removing unwanted whitespace characters.

    Replaces newline, carriage return, and tab characters with a space,
    and collapses multiple spaces into a single space.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text string.
    """
    text = text.replace("\n", " ")  # Replace newlines with spaces
    text = text.replace("\r", " ")  # Replace carriage returns with spaces
    text = text.replace("\t", " ")  # Replace tabs with spaces
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.strip()  # Strip leading and trailing spaces


class DataLoader:
    """
    Singleton class responsible for loading and cleaning various datasets
    needed for authorship verification and clustering analysis.
    """

    _instance = None

    def __new__(cls, config):
        """
        Implements the singleton pattern to ensure only one instance of DataLoader exists.
        Initializes the instance if it doesn't exist yet, otherwise returns the existing one.
        """
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
            cls._instance._initialized = False  # Track initialization
        return cls._instance

    def __init__(self, config):
        """
        Initializes the DataLoader singleton with config and logger.

        Args:
            config (dict): Configuration dictionary with paths and file names.
        """
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
        Loads and returns the cleaned Shakespeare dataset.

        Returns:
            list: A list of dictionaries with 'text_name' and cleaned 'text'.
        """
        data = load_json_data(self.data_path, self.shakespeare_dataset_name)
        return [{
            "text_name": item["text_name"],
            "text": _clean_text(item["text"])
        } for item in data]

    def get_impostor_texts_by_name(self, name):
        """
        Loads and returns the cleaned texts of a specific impostor by author name.

        Args:
            name (str): The name of the impostor author.

        Returns:
            list: List of cleaned text strings for the given author.

        Raises:
            ValueError: If the given author name is not found.
        """
        impostors = load_json_data(self.data_path, self.impostor_dataset_name)
        for impostor in impostors:
            if impostor["author"] == name:
                return [_clean_text(text) for text in impostor["texts"]]
        raise ValueError(f"Impostor with name '{name}' not found.")

    def get_impostors_name_list(self):
        """
        Retrieves all impostor author names.

        Returns:
            list: List of impostor names.
        """
        impostors = load_json_data(self.data_path, self.impostor_dataset_name)
        return [impostor['author'] for impostor in impostors]

    def get_all_impostors_data(self):
        """
        Loads and returns cleaned texts for all impostor authors.

        Returns:
            list: A list of dicts, each containing an author and their texts.
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
        Loads the target text and splits it by line.

        Returns:
            list: A list of stripped lines from the raw text.
        """
        data = load_json_data(self.data_path, self.text_to_classify_name)
        raw_text = data.get('text', '')
        return [line.strip() for line in raw_text.split('\n') if line.strip()]

    def get_pairs(self):
        """
        Lads and returns the impostor name pairs.

        Returns:
            list: List of impostor pairs.
        """
        data = load_json_data(self.data_path, self.pairs)
        return data

    def get_model_signals(self, model_name):
        """
        Loads signal data for the given model.

        Args:
            model_name (str): Name of the model to load signals for.

        Returns:
            dict: Signal data loaded from the model's file.
        """
        file_name = f"{model_name}-signals.json"
        path = self.data_path / self.signals_folder

        data = load_json_data(path, file_name)
        return data

    def get_shakespeare_included_text_names(self, model_name):
        """
        Loads the text names used in DTW for the given model.

        Args:
            model_name (str): Name of the model.

        Returns:
            list: Included text names for the model.
        """
        file_name = self.included_text_names_file_name
        path = self.data_path / self.distance_folder / model_name

        data = load_json_data(path, file_name)
        return data

    def get_dtw(self, model_name):
        """
        Loads the DTW (Dynamic Time Warping) distance matrix for a model.

        Args:
            model_name (str): Name of the model.

        Returns:
            Any: DTW matrix loaded from JSON.
        """
        file_name = self.dtw_file_name
        path = self.data_path / self.distance_folder / model_name

        data = load_json_data(path, file_name)
        return data

    def get_isolation_forest_results(self):
        """
        Loads the Isolation Forest anomaly scores for all models.

        Returns:
            dict: Dictionary with model names and their corresponding scores.
        """
        data = load_json_data(self.data_path, self.all_isolation_forest_scores)
        return data
