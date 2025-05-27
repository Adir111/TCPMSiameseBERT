import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping


class Trainer:
    def __init__(self, config, logger, model_creator, batch_size):
        self.config = config
        self.logger = logger
        self.model_creator = model_creator
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.learning_rate = float(config['training']['optimizer']['initial_learning_rate'])

    def __compile_model(self):
        """Compile the model with an optimizer, loss function, and metrics."""
        self.model_creator.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info(f"Model compiled, learning rate: {self.learning_rate}")

    def train(self, x_train, y_train, x_test, y_test):
        """Train the model on the dataset."""
        self.__compile_model()

        early_stopping = EarlyStopping(patience=self.config['training']['early_stopping_patience'], restore_best_weights=True, mode='min', baseline=0.4, monitor='val_loss')

        history = self.model_creator.model.fit(
            x=x_train, y=y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                early_stopping
            ]
        )

        self.model_creator.save_weights()

        return history
