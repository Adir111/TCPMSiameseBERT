import tensorflow as tf

class Trainer:
    def __init__(self, config, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.loss = config['training']['loss']
        self.learning_rate = config['training']['optimizer']['initial_learning_rate']

    def __compile_model(self):
        """Compile the model with an optimizer, loss function, and metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

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
