import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW

class EfficientNet_Classifier:
    def __init__(
        self,
        base_model, 
        num_classes:int,
        img_size:int=240,
        dropout:float=0.1,
        use_augmentation:bool=False,
    ):
        self.img_size = img_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_augmentation = use_augmentation
        self.base_model = base_model
        self.model = self._build_model()

    def _build_model(self):
        # initial inputs
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))

        # Optional augmentation
        x = inputs
        if self.use_augmentation:
            x = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.2),
                layers.RandomContrast(0.2),
            ])(x)
        
        # preprocess for efficientnet_v2
        x = tf.keras.applications.efficientnet.preprocess_input(x)

        # base model
        # base_model = tf.keras.applications.EfficientNetB0(
        # include_top=False,
        # weights="imagenet",
        # input_shape=(self.img_size, self.img_size, 3),
        # )

        self.base_model.trainable = False    # freeze layers

        x = self.base_model(x, training=False)

        # classifier head
        x = layers.GlobalAveragePooling2D()(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dense(1024, activation="relu")(x)
        # x = layers.Dropout(self.dropout)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Dense(512, activation="relu")(x)
        # x = layers.Dropout(self.dropout)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs) # create model

        return model

    def compile_head(self, learning_rate:float=1e-3, label_smoothing:float=0.1):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),],
        )

        return

    def train_head(self, train_ds, val_ds, epochs_head:int=10, callbacks:list=None):
        print("\n🔹 Training Classifier Head")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head,
            callbacks=callbacks
        )

    def fine_tune(
        self, 
        train_ds, 
        val_ds,
        epochs_fine:int=20, 
        percent_unfreeze:float=0.2, 
        learning_rate:float=1e-5,
        weight_decay:float=1e-4,
        callbacks:list=None,
    ):

        self.base_model.trainable = True    # make base model trainable

        n_layers = len(self.base_model.layers)
        num_unfreeze = int(n_layers * percent_unfreeze)

        # keep batch norm layers frozen
        for layer in self.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # Freeze early layers (safe default)
        for layer in self.base_model.layers[:-num_unfreeze]:
            layer.trainable = False

        self.model.compile(
            optimizer=AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),],
        )

        print("\n🔹 Fine-tuning model...")
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_fine,
            callbacks=callbacks
        )