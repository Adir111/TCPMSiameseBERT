import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from pathlib import Path

from .data_loader import DataLoader
from .preprocess import Preprocessor
from .trainer import Trainer
from .model import SiameseBertModel
from PhaseB.bert_siamese_authorship_verification.utilities import make_pairs, DataVisualizer

# from src.dtw import compute_dtw_distance
# from src.isolation_forest import AnomalyDetector
# from src.clustering import perform_kmedoids_clustering

tf.get_logger().setLevel('ERROR')


class Procedure:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.data_visualizer = DataVisualizer(logger)
        self.preprocessor = Preprocessor(config=config)
        self.max_length = config['bert']['maximum_sequence_length']
        self.batch_size = config['training']['batch_size']
        self.data_loader = DataLoader(config=config)
        self.trained_networks = []
        self.model_creator = SiameseBertModel(config=config, logger=self.logger)

    # def __preprocessing_stage(self, impostor_1_name, impostor_2_name, shakespeare_data):
    def __preprocessing_stage(self, impostor_1_name, impostor_2_name):
        print("----------------------")
        self.logger.info("Starting preprocessing stage...")

        def __load_and_preprocess(impostor_name):
            impostor_texts = self.data_loader.get_impostor_texts_by_name(impostor_name)
            impostor_chunks, impostor_tokens_count = self.preprocessor.preprocess(impostor_texts)
            self.logger.info(
                f"Before equalization: {impostor_name} - {len(impostor_chunks)} chunks with {impostor_tokens_count} tokens")
            return impostor_chunks, impostor_tokens_count

        impostor_1_chunks, impostor_1_tokens_count = __load_and_preprocess(impostor_1_name)
        impostor_2_chunks, impostor_2_tokens_count = __load_and_preprocess(impostor_2_name)
        # shakespeare_chunks, shakespeare_tokens_count = self.preprocessor.preprocess(shakespeare_data)

        impostor_1_chunks, impostor_2_chunks = self.preprocessor.equalize_chunks([impostor_1_chunks, impostor_2_chunks])

        # Log after stabilizing
        self.logger.info(f"After equalization: {impostor_1_name} - {len(impostor_1_chunks)} chunks")
        self.logger.info(f"After equalization: {impostor_2_name} - {len(impostor_2_chunks)} chunks")

        self.logger.info("✅ Preprocessing stage has been completed!")
        print("----------------------")
        return impostor_1_chunks, impostor_2_chunks

    def __training_stage(self, model_name, impostor_1_preprocessed, impostor_2_preprocessed):
        print("----------------------")
        self.logger.info("Starting training stage...")

        model = self.model_creator.build_model(model_name)
        trainer = Trainer(self.config, self.logger, self.model_creator, model, self.batch_size)

        x_train, y_train, x_test, y_test = self.preprocessor.create_xy(impostor_1_preprocessed, impostor_2_preprocessed)
        history = trainer.train(x_train, y_train, x_test, y_test)

        self.logger.info("✅ Training stage has been completed!")
        print("----------------------")
        return history

    def run(self):
        # shakespeare_data = self.data_loader.get_shakespeare_data()
        impostors_names = self.data_loader.get_impostors_name_list()
        impostor_pairs = make_pairs(impostors_names)
        self.logger.info(f"Batch size is {self.batch_size}")

        for idx, impostor_pair in enumerate(impostor_pairs):
            self.logger.info(
                f"Training model number {idx + 1} for impostor pair: {impostor_pair[0]} and {impostor_pair[1]}")
            impostor_1_preprocessed, impostor_2_preprocessed = self.__preprocessing_stage(impostor_pair[0],
                                                                                          impostor_pair[1])
            model_name = f"{impostor_pair[0]}_{impostor_pair[1]}"
            history = self.__training_stage(model_name, impostor_1_preprocessed, impostor_2_preprocessed)
            self.logger.info(f"Model {idx + 1} training complete.")
