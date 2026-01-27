import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os

# =====================
# CONFIG
# =====================
DATASET_DIR = "data"
IMG_SIZE = 256          # change later if needed
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
NUM_CLASSES = None      # auto-detected
SEED = 42
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
# MODEL
# =====================
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
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

# =====================
# PHASE 1: TRAIN HEAD
# =====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

print("\n🔹 Training classifier head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
)

# =====================
# PHASE 2: FINE-TUNING
# =====================
base_model.trainable = True

# Freeze early layers (safe default)
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=AdamW(
        learning_rate=1e-5,
        weight_decay=1e-4,
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

print("\n🔹 Fine-tuning model...")
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
