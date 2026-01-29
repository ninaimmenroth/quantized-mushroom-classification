import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os

# =====================
# CONFIG
# =====================
DATASET_DIR = "data"
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
SEED = 42
UNFREEZE_LAYERS = 50

print(tf.config.list_physical_devices())

# =====================
# DATA LOADING
# =====================
def resize_with_pad(image, label):
    # resizes the image while maintaining the aspect ratio, padding with zeros
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    return image, label

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=None,        # <-- keep original and later resize/pad
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=None,        # <-- keep original and later resize/pad
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

# automatic detection of number of classes
NUM_CLASSES = train_ds.element_spec[1].shape[-1]

# performance optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(resize_with_pad, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(resize_with_pad, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# =====================
# DATA AUGMENTATION
# =====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# =====================
# MODEL BUILDER (for tuning)
# =====================
def build_model(hp):
    dense_units = hp.Choice("dense_units", [256, 512, 768])
    dropout_rate = hp.Float("dropout", 0.3, 0.6, step=0.1)
    lr = hp.Float("lr", 1e-4, 3e-3, sampling="log")
    label_smoothing = hp.Float("label_smoothing", 0.0, 0.15, step=0.05)

    base_model = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )

    return model


# =====================
# HYPERPARAMETER SEARCH
# =====================
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=15,   # small but effective
    directory="tuning",
    project_name="mushroom_efficientnet_v2m",
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best hyperparameters:", best_hp.values)




# =====================
# REBUILD BEST MODEL
# =====================
model = tuner.hypermodel.build(best_hp)
model.summary()

# =====================
# PHASE 1: TRAIN HEAD
# =====================
print("\n Training classifier head with best hyperparameters...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
)

# =====================
# PHASE 2: FINE-TUNING
# =====================
base_model = model.layers[3]  # EfficientNet backbone

base_model.trainable = True

# Unfreeze last N layers (fixed for reproducibility)
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=AdamW(
        learning_rate=1e-5,
        weight_decay=1e-4,
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=best_hp.get("label_smoothing")
    ),
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ],
)

print("\n Fine-tuning model...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
)

# =====================
# SAVE MODEL
# =====================
model.save("output/efficientnet_v2_m_mushrooms")
print("\n Model saved!")
