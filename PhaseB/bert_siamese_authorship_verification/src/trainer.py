import io
from contextlib import redirect_stdout
import tensorflow as tf


class Trainer:
    def __init__(self, config, logger, model, batch_size):
        self.logger = logger
        self.model = model
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.loss = config['training']['loss']
        self.learning_rate = float(config['training']['optimizer']['initial_learning_rate'])

    def __compile_model(self):
        """Compile the model with an optimizer, loss function, and metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info(f"Model compiled with loss: {self.loss}, learning rate: {self.learning_rate}")

    def train(self, x_train, y_train, x_test, y_test):
        """Train the model on the dataset."""
        self.__compile_model()

        # score_before = self.model.evaluate_generator(x_train, y_train, batch_size=self.batch_size)
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        # score_after = self.model.evaluate(x_train, y_train, batch_size=64)
        # print("score_before: ", score_before, ", score_after: ", score_after)
        return history
