import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os
import wandb
from dataloader import build_dataset


# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
IMG_SIZE = 240          # final size for model input
BATCH_SIZE = 16
EPOCHS = 1
NUM_CLASSES = None      # auto-detected
SEED = 42
OUTPUT_DIR = "output"
MODEL_NAME = "ResNet50_Mushrooms.keras"

print(tf.config.list_physical_devices())

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

wandb.init(
    project="mushroom-classification",
    name="resnet50_head",
    config={
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_head": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
    },
    reinit=True
)

AUTOTUNE = tf.data.AUTOTUNE

# =====================
# WANDB LOGGER
# =====================
class WandbLogger(tf.keras.callbacks.Callback):
    """Logs training & validation metrics to WandB per epoch."""
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log(logs, step=epoch)

# =====================
# BUILD DATASETS
# =====================

# build train dataset
train_ds, class_names = build_dataset(
    dataset_dir=DATASET_DIR,
    fold = "train",
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    shuffle=True,
    seed = SEED,
)

# build validation dataset
val_ds, class_names = build_dataset(
    dataset_dir=DATASET_DIR,
    fold = "val",
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    shuffle=False,
    seed = SEED,
)

NUM_CLASSES = len(class_names)

# =====================
# DATA AUGMENTATION
# =====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])


# =========================================
# MODEL: ResNet with Classifier Head
# =========================================
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',       # pretrained on ImageNet
    include_top=False,        # exclude the original classifier
    input_shape=(IMG_SIZE, IMG_SIZE, 3)  # adjust to your image size
)

base_model.trainable = False # freeze layers

# Create the classification head
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
# x = layers.Dense(512, activation="relu")(x)
# x = layers.Dropout(0.4)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.summary()


# =====================
# COMPILE
# =====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

# =====================
# CALLBACKS
# =====================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    min_delta=1e-3,
    restore_best_weights=True,
    verbose=1
)

wandb_cb = WandbLogger()

# =====================
# TRAIN
# =====================
print("\n🔹 Training classifier head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, wandb_cb]
)


# =====================
# SAVE MODEL
# =====================
model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
model.save(model_path)
print("\n Model saved!")

# Finish W&B run
wandb.finish()