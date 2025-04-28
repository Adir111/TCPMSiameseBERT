import random
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys
from pathlib import Path

from src_new.data_loader import DataLoader
from src_new.preprocess import Preprocessor
from src.dtw import compute_dtw_distance
from src.isolation_forest import AnomalyDetector
from src.clustering import perform_kmedoids_clustering
from src.model import SiameseBertModel

tf.get_logger().setLevel('ERROR')

if tf.__version__.startswith('2.10.'):
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow_addons.optimizers import AdamW
else:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.optimizers import AdamW

def make_pairs(impostor_names):
    """
    Generate all possible pairs of impostor names from the given list.
    """
    pairs = []
    for i in range(len(impostor_names)):
        for j in range(i + 1, len(impostor_names)):
            pairs.append((impostor_names[i], impostor_names[j]))
    return pairs

def equalize_chunks(chunks_list):
    """
    Helper to balance two lists of chunks to the same length.
    """
    lengths = [len(chunks_list[0]), len(chunks_list[1])]
    i1 = lengths.index(max(lengths))
    i2 = lengths.index(min(lengths))

    temp = []
    for _ in range(lengths[i1] // lengths[i2]):
        temp += chunks_list[i2]

    chunks_list[i2] = temp + random.sample(chunks_list[i2], lengths[i1] - len(temp))

    return chunks_list

class Procedure:
    def __init__(self, config, logger, data_visualizer, preprocessor):
        self.config = config
        self.logger = logger
        self.data_visualizer = data_visualizer
        self.preprocessor = preprocessor
        self.chunk_size = config['training']['chunk_size']
        self.batch_size = config['training']['batch_size']
        self.data_loader = DataLoader(
            data_path=self.config['data']['organised_data_folder_path'],
            shakespeare_dataset_name=self.config['data']['shakespeare_data_source'],
            impostor_dataset_name=self.config['data']['impostors_data_source'],
            text_to_classify_name=self.config['data']['classify_text_data_source']
        )
        self.trained_networks = []

    def _create_dataset_from_chunks(self, chunks):
        """Helper to create a TensorFlow dataset from already preprocessed chunks."""
        batches = len(chunks) // self.batch_size + (1 if len(chunks) % self.batch_size != 0 else 0)
        dataset = tf.data.Dataset.from_tensor_slices(chunks).batch(batches)
        return dataset, len(chunks), batches


    def preprocessing(self, impostor_1_name, impostor_2_name, shakespeare_data):
        print("----------------------")
        self.logger.log("[INFO] Starting preprocessing data...")

        # Load data
        impostor_1_texts = self.data_loader.get_impostor_texts_by_name(impostor_1_name)
        impostor_2_texts = self.data_loader.get_impostor_texts_by_name(impostor_2_name)

        # Preprocess
        impostor_1_chunks, impostor_1_tokens_count = self.preprocessor.preprocess(impostor_1_texts, chunk_size=self.chunk_size)
        impostor_2_chunks, impostor_2_tokens_count = self.preprocessor.preprocess(impostor_2_texts, chunk_size=self.chunk_size)
        shakespeare_chunks, shakespeare_tokens_count = self.preprocessor.preprocess(shakespeare_data, chunk_size=self.chunk_size)

        # Log before stabilizing
        self.logger.log(
            f"[INFO] Before equalization: {impostor_1_name} - {len(impostor_1_chunks)} chunks, {impostor_1_tokens_count} tokens")
        self.logger.log(
            f"[INFO] Before equalization: {impostor_2_name} - {len(impostor_2_chunks)} chunks, {impostor_2_tokens_count} tokens")

        impostor_1_chunks, impostor_2_chunks = equalize_chunks([impostor_1_chunks, impostor_2_chunks])

        # Create datasets
        impostor_1_dataset, impostor_1_chunks_count, impostor_1_batches_count = self._create_dataset_from_chunks(
            impostor_1_chunks)
        impostor_2_dataset, impostor_2_chunks_count, impostor_2_batches_count = self._create_dataset_from_chunks(
            impostor_2_chunks)
        shakespeare_dataset, shakespeare_chunks_count, shakespeare_batches_count = self._create_dataset_from_chunks(
            shakespeare_chunks)

        # Log after stabilizing
        self.logger.log(f"[INFO] After equalization: {impostor_1_name} - {impostor_1_batches_count} batches, "
                        f"{impostor_1_chunks_count} chunks")
        self.logger.log(f"[INFO] After equalization: {impostor_2_name} - {impostor_2_batches_count} batches, "
                        f"{impostor_2_chunks_count} chunks")
        self.logger.log(f"[INFO] Processed shakespeare - {shakespeare_batches_count} batches, "
                        f"{shakespeare_chunks_count} chunks, {shakespeare_tokens_count} tokens")

        self.logger.log("[INFO] ✅ Preprocessing data has been completed!")
        print("----------------------")
        return impostor_1_dataset, impostor_2_dataset, shakespeare_dataset


    def full_procedure(self):
        shakespeare_data = self.data_loader.get_shakespeare_data()
        impostors_names = self.data_loader.get_impostors_name_list()
        impostor_pairs = make_pairs(impostors_names)

        for idx, impostor_pair in enumerate(impostor_pairs):
            self.logger.log(f"[INFO] Training model number {idx + 1} for impostor pair: {impostor_pair[0]} and {impostor_pair[1]}")
            impostor_1_preprocessed, impostor_2_preprocessed, shakespeare_data = self.preprocessing(impostor_pair[0], impostor_pair[1], shakespeare_data)

