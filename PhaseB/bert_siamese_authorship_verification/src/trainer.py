"""
This module defines the Trainer class which manages the training lifecycle of a
TensorFlow Keras model. It handles model compilation, training with early stopping,
and saving model weights.

Key features:
- Configurable training parameters such as epochs, batch size, learning rate,
  and early stopping criteria.
- Custom callbacks to log early stopping events and optionally stop training
  when a performance threshold is reached.
- Encapsulates model compilation with Adam optimizer and binary crossentropy loss.

Intended to be used with a model creator object that provides the Keras model
and saving mechanism.
"""

import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, Callback


class Trainer:
    def __init__(self, config, logger, model_creator, batch_size):
        """
        Initialize Trainer with config, logger, model creator, and batch size.

        Args:
            config (dict): Configuration dictionary with training parameters.
            logger (Logger): Logger instance for logging messages.
            model_creator (object): Object providing the Keras model and save method.
            batch_size (int): Batch size for training.
        """
        self.logger = logger
        self.model_creator = model_creator
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.learning_rate = float(config['training']['optimizer']['initial_learning_rate'])
        self.monitor = config['training']['early_stopping']['monitor']
        self.patience = int(config['training']['early_stopping']['patience'])
        self.baseline = float(config['training']['early_stopping']['baseline'])

    def __compile_model(self):
        """
        Compile the Keras model with Adam optimizer, binary crossentropy loss,
        and accuracy metric.
        """
        self.model_creator.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info(f"Model compiled, learning rate: {self.learning_rate}")

    class EarlyStoppingLogger(tf.keras.callbacks.Callback):
        """
        Initialize callback to log when early stopping triggers.

        Args:
            logger (Logger): Logger instance.
        """
        def __init__(self, logger):
            super().__init__()
            self.logger = logger

        def on_train_end(self, logs=None):
            """
            Log message if training was stopped early.
            """
            if self.model.stop_training:
                self.logger.info("Early stopping was triggered.")

    class EarlyStoppingAtThreshold(Callback):
        def __init__(self, monitor='val_accuracy', threshold=0.97):
            """
            Initialize callback to stop training when monitored metric reaches threshold.

            Args:
                monitor (str): Metric name to monitor.
                threshold (float): Threshold value to stop training.
            """
            super().__init__()
            self.monitor = monitor
            self.threshold = threshold

        def on_epoch_end(self, epoch, logs=None):
            """
            Check metric value at epoch end and stop training if threshold met.
            """
            logs = logs or {}
            val_acc = logs.get(self.monitor)
            if val_acc is not None and val_acc >= self.threshold:
                print(f"\nEpoch {epoch + 1}: Reached {self.monitor} >= {self.threshold:.2f}, stopping training.")
                self.model.stop_training = True

    def train(self, x_train, y_train, x_test, y_test):
        """
        Compile and train the model with early stopping callbacks.

        Args:
            x_train (np.array or tf.Tensor): Training features.
            y_train (np.array or tf.Tensor): Training labels.
            x_test (np.array or tf.Tensor): Validation features.
            y_test (np.array or tf.Tensor): Validation labels.

        Returns:
            History: Keras History object with training metrics.
        """
        self.__compile_model()

        early_stopping = EarlyStopping(
            monitor=self.monitor,
            mode='max',
            patience=self.patience,
            baseline=self.baseline,
            restore_best_weights=True
        )

        # early_stopping = self.EarlyStoppingAtThreshold(monitor=self.monitor, threshold=self.baseline)
        early_stopping_logger = self.EarlyStoppingLogger(self.logger)

        callbacks = [
            early_stopping,
            early_stopping_logger
        ]

        history = self.model_creator.model.fit(
            x=x_train, y=y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks
        )

        self.model_creator.save_weights()

        return history
