import tensorflow as tf


class PredictionLogger(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, limit=5):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.limit = limit

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.x_val, verbose=0)
        print(f"\nEpoch {epoch+1} Predictions:")
        for i in range(min(self.limit, len(preds))):
            print(f"x[{i}]: prediction = {preds[i][0]:.4f}, true = {self.y_val[i][0]}")
