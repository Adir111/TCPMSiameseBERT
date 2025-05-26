import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbModelCheckpoint


class Trainer:
    def __init__(self, config, logger, model_creator, bert_model1, bert_model2, batch_size):
        self.config = config
        self.logger = logger
        self.model_creator = model_creator
        self.model = model_creator.build_siamese_model(bert_model1, bert_model2)
        self.batch_size = batch_size
        self.epochs = config['training']['epochs']
        self.learning_rate = float(config['training']['optimizer']['initial_learning_rate'])

    def __compile_model(self):
        """Compile the model with an optimizer, loss function, and metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.logger.info(f"Model compiled, learning rate: {self.learning_rate}")

    def train(self, x_train, y_train, x_test, y_test):
        """Train the model on the dataset."""
        self.__compile_model()

        early_stopping = EarlyStopping(patience=self.config['training']['early_stopping_patience'], restore_best_weights=True, mode='min', baseline=0.5, monitor='val_loss')
        wandb_model_checkpoint = WandbModelCheckpoint(
            filepath=f"weights-{self.model_creator.model_name}.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            save_freq="epoch"
        )

        history = self.model.fit(
            x=x_train, y=y_train,
            validation_data=(x_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[
                early_stopping,
                wandb_model_checkpoint
            ]
        )

        return history
