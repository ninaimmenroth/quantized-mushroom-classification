import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW

class ResNet50Classifier:
    def __init__(self, img_size: int = 240, num_classes: int = 10, use_augmentation: bool = True):
        self.img_size = img_size
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation
        self.model = self._build_model()

    def _build_model(self):
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

        # Preprocessing for ResNet50
        x = tf.keras.applications.resnet50.preprocess_input(x)

        # Base model
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',     # pretrained on ImageNet
            include_top=False,      # exclude the original classifier
            input_shape=(self.img_size, self.img_size, 3)   # adjust to your image size
        )

        base_model.trainable = False    # freeze layers

        x = base_model(x, training=False)

        # Classifier head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs) # create model

        return model


