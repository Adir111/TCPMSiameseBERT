import tensorflow as tf

class Trainer:
    def __init__(self, config, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.loss=config['training']['loss']
        self.learning_rate = config['training']['optimizer']['initial_learning_rate']

    def __compile_model(self):
        """Compile the model with an optimizer, loss function, and metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_dataset):
        """Train the model on the dataset."""
        self.__compile_model()
        history = self.model.fit(
            train_dataset.batch(self.batch_size),
            epochs=self.epochs
        )
        return history
