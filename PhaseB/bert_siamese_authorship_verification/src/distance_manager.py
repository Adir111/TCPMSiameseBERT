import numpy as np
from pathlib import Path
from dtaidistance import dtw

from .data_loader import DataLoader

from PhaseB.bert_siamese_authorship_verification.utilities import save_to_json


class SignalDistanceManager:
    _instance = None

    def __new__(cls, config, logger):
        if cls._instance is None:
            cls._instance = super(SignalDistanceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config, logger):
        if self._initialized:
            return  # Avoid reinitialization
        self._initialized = True

        self.logger = logger
        self.data_loader = DataLoader(config)
        self.output_path = Path(config['data']['organised_data_folder_path']) / config['data']['dtw']['output_distance_folder']
        self.dtw_file_name = config['data']['dtw']['dtw_file_name']
        self.included_text_names_file_name = config['data']['dtw']['included_text_names_file_name']
        self.signals_file_name = config['data']['dtw']['signals_file_name']

        self.output_path.mkdir(parents=True, exist_ok=True)


    def compute_distance_matrix_for_model(self, model_name):
        self.logger.info(f"Computing distance matrix for model: {model_name}")

        # Load signal data
        model_signals = self.data_loader.get_model_signals(model_name)

        # Filter and batch-average signals
        signals, included_text_names = self.__get_filtered_signals_and_text_names(model_signals)

        if not signals:
            self.logger.warn("No valid signals to process.")
            return

        # Create distance matrix
        distance_matrix = self.__create_dtw_distance_matrix(signals)

        # Save results
        self.__save_results(distance_matrix, included_text_names, model_name)
        self.logger.info(f"Finished computing distance matrix for {model_name}")


    @staticmethod
    def __get_filtered_signals_and_text_names(signals_dict):
        filtered_signals = {}
        included_text_names = []

        for text_name, signal in signals_dict.items():
            # Skip signals with fewer than 2 batches
            if len(signal) < 2:
                continue
            included_text_names.append(text_name)

            filtered_signals[text_name] = signal

        return filtered_signals, included_text_names

    @staticmethod
    def __create_dtw_distance_matrix(signals_dict):
        keys = list(signals_dict.keys())
        m = len(keys)
        dist_mat = np.zeros((m, m))

        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                dist_mat[i, j] = dtw.distance(signals_dict[key1], signals_dict[key2])

        return dist_mat

    def __get_dtw_file_paths(self, model_name):
        """
        Returns the paths for the DTW matrix and included text names JSON files.
        """
        model_output_dir = self.output_path / model_name
        matrix_file_path = model_output_dir / self.dtw_file_name
        names_file_path = model_output_dir / self.included_text_names_file_name
        return model_output_dir, matrix_file_path, names_file_path


    def dtw_results_already_exist(self, model_name):
        """
        Check if the DTW results for the given model already exist.
        """
        _, matrix_file_path, names_file_path = self.__get_dtw_file_paths(model_name)
        return matrix_file_path.exists() and names_file_path.exists()


    def __save_results(self, distance_matrix, included_text_names, model_name):
        model_output_dir, matrix_file_path, names_file_path = self.__get_dtw_file_paths(model_name)
        # Create subfolder for this model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        save_to_json(distance_matrix.tolist(), matrix_file_path, f"DTW ({model_name})")
        save_to_json(included_text_names, names_file_path, f"Included Text Names ({model_name})")
