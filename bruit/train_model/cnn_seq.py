import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

class CNNSequenceClassifier:
    def __init__(self, input_shape, num_classes, params: dict):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = params
        self.architecture = params["architecture"]
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        filters = self.architecture.get("filters", [32, 64])
        kernel_size = tuple(self.architecture.get("kernel_size", [3, 3]))
        pool_size = tuple(self.architecture.get("pool_size", [2, 2]))
        activation = self.architecture.get("activation", "relu")
        dropout_rate = self.architecture.get("dropout_rate", 0.5)
        dense_units = self.architecture.get("dense_units", [128])
        output_activation = self.architecture.get("architecture",{}).get("output_activation", "softmax")
        rnn_units = self.architecture.get("rnn_units", 64)
        rnn_type = self.architecture.get("rnn_type", "lstm").lower()  # "lstm" or "gru"

        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))

        # CNN stack
        for i in range(self.params.get("conv_layers", len(filters))):
            model.add(layers.Conv2D(filters[i], kernel_size, activation=activation, padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D(pool_size))

        # Reshape for RNN: (batch, time, features)
        shape = model.output_shape  # (None, H, W, C)
        time_steps = shape[1]
        features = shape[2] * shape[3]
        model.add(layers.Reshape((time_steps, features)))

        # Recurrent Layer
        if rnn_type == "gru":
            model.add(layers.Bidirectional(layers.GRU(rnn_units, return_sequences=False)))
        else:
            model.add(layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=False)))

        # Dense layers
        for units in dense_units:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.num_classes, activation=output_activation))

        return model

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.params["learning_rate"]),
            loss=self.params["loss_function"],
            metrics=['accuracy']
        )

    def fit(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, patience=5,class_weight=None):
        callbacks = []

        # Add EarlyStopping if patience is specified
        if patience > 0:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ))

        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks
        )

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.export(path)

    def summary(self):
        self.model.summary()
