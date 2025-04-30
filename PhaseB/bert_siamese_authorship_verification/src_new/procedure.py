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
from PhaseB.bert_siamese_authorship_verification.utilities import env_handler, make_pairs

# from src.dtw import compute_dtw_distance
# from src.isolation_forest import AnomalyDetector
# from src.clustering import perform_kmedoids_clustering

tf.get_logger().setLevel('ERROR')

if env_handler:
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow_addons.optimizers import AdamW
else:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import AdamW


class Procedure:
    def __init__(self, config, logger, data_visualizer):
        self.config = config
        self.logger = logger
        self.data_visualizer = data_visualizer
        self.preprocessor = Preprocessor(config)
        self.max_length = config['bert']['maximum_sequence_length']
        self.chunk_size = config['training']['chunk_size']
        self.batch_factor = config['training']['batch_factor']
        self.batch_size = self.chunk_size // self.batch_factor
        self.data_loader = DataLoader(
            data_path=self.config['data']['organised_data_folder_path'],
            shakespeare_dataset_name=self.config['data']['shakespeare_data_source'],
            impostor_dataset_name=self.config['data']['impostors_data_source'],
            text_to_classify_name=self.config['data']['classify_text_data_source']
        )
        self.trained_networks = []
        self.model_creator = SiameseBertModel(config=config, logger=logger)

    def preprocessing_stage(self, impostor_1_name, impostor_2_name, shakespeare_data):
        def _create_dataset_from_chunks(chunks):
            """Helper to create a TensorFlow dataset from already preprocessed chunks."""
            dataset = tf.data.Dataset.from_generator(
                lambda: (chunk for chunk in chunks),
                output_signature={
                    'input_ids': tf.TensorSpec(shape=(self.max_length,), dtype=tf.int32),
                    'attention_mask': tf.TensorSpec(shape=(self.max_length,), dtype=tf.int32),
                }
            ).batch(self.batch_size)

            return dataset, len(chunks), (len(chunks) // self.batch_size) + (
                1 if len(chunks) % self.batch_size != 0 else 0)

        print("----------------------")
        self.logger.log("[INFO] Starting preprocessing stage...")

        # Load data
        impostor_1_texts = self.data_loader.get_impostor_texts_by_name(impostor_1_name)
        impostor_2_texts = self.data_loader.get_impostor_texts_by_name(impostor_2_name)

        # Preprocess
        impostor_1_chunks, impostor_1_tokens_count = self.preprocessor.preprocess(impostor_1_texts)
        impostor_2_chunks, impostor_2_tokens_count = self.preprocessor.preprocess(impostor_2_texts)
        shakespeare_chunks, shakespeare_tokens_count = self.preprocessor.preprocess(shakespeare_data)

        # Log before stabilizing
        self.logger.log(f"[INFO] Before equalization: {impostor_1_name} - {len(impostor_1_chunks)} chunks, {impostor_1_tokens_count} tokens")
        self.logger.log(f"[INFO] Before equalization: {impostor_2_name} - {len(impostor_2_chunks)} chunks, {impostor_2_tokens_count} tokens")

        impostor_1_chunks, impostor_2_chunks = self.preprocessor.equalize_chunks([impostor_1_chunks, impostor_2_chunks])

        # Create datasets
        impostor_1_dataset, impostor_1_chunks_count, impostor_1_batches_count = _create_dataset_from_chunks(impostor_1_chunks)
        impostor_2_dataset, impostor_2_chunks_count, impostor_2_batches_count = _create_dataset_from_chunks(impostor_2_chunks)
        shakespeare_dataset, shakespeare_chunks_count, shakespeare_batches_count = _create_dataset_from_chunks(shakespeare_chunks)

        # Log after stabilizing
        self.logger.log(f"[INFO] After equalization: {impostor_1_name} - {impostor_1_batches_count} batches, {impostor_1_chunks_count} chunks")
        self.logger.log(f"[INFO] After equalization: {impostor_2_name} - {impostor_2_batches_count} batches, {impostor_2_chunks_count} chunks")
        self.logger.log(f"[INFO] Processed shakespeare - {shakespeare_batches_count} batches, {shakespeare_chunks_count} chunks, {shakespeare_tokens_count} tokens")

        self.logger.log("[INFO] ✅ Preprocessing stage has been completed!")
        print("----------------------")
        return (impostor_1_dataset, impostor_1_chunks_count), (impostor_2_dataset, impostor_2_chunks_count), shakespeare_dataset

    def training_stage(self, impostor_1_preprocessed, impostor_2_preprocessed):
        print("----------------------")
        self.logger.log("[INFO] Starting training stage...")
        model = self.model_creator.build_model()
        model.summary()
        trainer = Trainer(self.config, model, self.batch_size)
        train_dataset = self.preprocessor.create_xy(impostor_1_preprocessed, impostor_2_preprocessed)
        try:
            history = trainer.train(train_dataset)
        except Exception as e:
            print(str(e))
            exit(-1)
        self.logger.log("[INFO] ✅ Training stage has been completed!")
        print("----------------------")
        return history

    def run(self):
        shakespeare_data = self.data_loader.get_shakespeare_data()
        impostors_names = self.data_loader.get_impostors_name_list()
        impostor_pairs = make_pairs(impostors_names)
        self.logger.log(f"[INFO] Batch size is {self.batch_size}")

        for idx, impostor_pair in enumerate(impostor_pairs):
            self.logger.log(f"[INFO] Training model number {idx + 1} for impostor pair: {impostor_pair[0]} and {impostor_pair[1]}")
            impostor_1_preprocessed, impostor_2_preprocessed, shakespeare_data = self.preprocessing_stage(impostor_pair[0], impostor_pair[1], [])
            history = self.training_stage(impostor_1_preprocessed, impostor_2_preprocessed)
            self.logger.log(f"[INFO] Model {idx + 1} training complete.")
