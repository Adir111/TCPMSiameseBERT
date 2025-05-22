import tensorflow as tf


class Trainer:
    def __init__(self, config, logger, model_creator, model, batch_size):
        self.logger = logger
        self.model_creator = model_creator
        self.model = model
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.loss = config['training']['loss']
        self.learning_rate = float(config['training']['optimizer']['initial_learning_rate'])

    def __compile_model(self):
        """Compile the model with an optimizer, loss function, and metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.loss,
            metrics=['accuracy']
        )
        self.logger.info(f"Model compiled with loss: {self.loss}, learning rate: {self.learning_rate}")

    def train(self, x_train, y_train, x_test, y_test):
        """Train the model on the dataset."""
        self.__compile_model()

        history = self.model.fit(
            x=x_train, y=y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        return history
