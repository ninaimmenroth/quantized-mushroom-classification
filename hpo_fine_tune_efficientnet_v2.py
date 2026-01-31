import tensorflow as tf
import keras_tuner as kt
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os
import wandb
import logging
import sys
from datetime import datetime
from pathlib import Path

# =====================
# CONFIG
# =====================
DATASET_DIR = "images"
IMG_SIZE = 240
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
SEED = 42
UNFREEZE_LAYERS = 50
SMOKE_TEST = True

if SMOKE_TEST:
    EPOCHS_HEAD = 2
    EPOCHS_FINE = 2
    BATCH_SIZE = 4

AUTOTUNE = tf.data.AUTOTUNE


# =====================
# PLOTTING & LOGGING
# =====================
class WandbEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            wandb.log({f"hpo/{k}": v for k, v in logs.items()}, step=epoch)


class ConfidenceLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        log_confidence_stats(self.model, self.val_ds)


wandb.init(
    project="test", #mushroom-classification
    name= "SMOKE_TEST-efficientnetv2m",#"efficientnetv2m-hparam-search",
    tags=["smoke-test"],
    config={
        "smoke_test": SMOKE_TEST,
        "architecture": "EfficientNetV2-M",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_head": EPOCHS_HEAD,
        "epochs_fine": EPOCHS_FINE,
        "dataset": "custom_mushrooms",
        "task": "species + edibility",
    },
)


def setup_logger(log_dir="logs", run_name="run"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_path


logger, log_file = setup_logger(
    run_name=wandb.run.name if wandb.run else "local_run"
)

logger.info("Logger initialized")
logger.info(f"Log file: {log_file}")

logger.info(tf.config.list_physical_devices())

def log_exceptions(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

sys.excepthook = log_exceptions

def log_confidence_stats(model, val_ds, logger=None):
    confidences = []
    entropies = []

    for images, _ in val_ds:
        preds = model.predict(images, verbose=0)
        confidences.extend(tf.reduce_max(preds, axis=1).numpy())
        entropies.extend(
            tfp.distributions.Categorical(probs=preds).entropy().numpy()
        )

    stats = {
        "confidence/mean": float(tf.reduce_mean(confidences)),
        "confidence/std": float(tf.math.reduce_std(confidences)),
        "entropy/mean": float(tf.reduce_mean(entropies)),
    }

    if logger:
        logger.info(f"Confidence stats: {stats}")

    wandb.log(stats)

# =====================
# MANUAL DATA LOADING + RESIZE WITH PADDING
# =====================
def load_and_pad_image(path, label):
    """Loads a JPEG image, converts to float32, pads to IMG_SIZE while preserving aspect ratio."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def get_file_paths_and_labels(root_dir):
    """Scans directory and returns sorted file paths and integer labels."""
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])
    return paths, labels, class_names

# =====================
# GET TRAIN + VAL FILES
# =====================
train_paths, train_labels, class_names = get_file_paths_and_labels(os.path.join(DATASET_DIR, "train"))
val_paths, val_labels, _ = get_file_paths_and_labels(os.path.join(DATASET_DIR, "val"))

NUM_CLASSES = len(class_names)

# =====================
# BUILD TF.DATA DATASETS
# =====================
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(1000, seed=SEED)
train_ds = train_ds.map(load_and_pad_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_pad_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

if SMOKE_TEST:
    logger.info("!!! SMOKE TEST MODE: Using tiny dataset")

    train_ds = train_ds.take(2)   # 2 batches total
    val_ds = val_ds.take(2)

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
    max_trials=2 if SMOKE_TEST else 15,
    directory="tuning",
    project_name="mushroom_efficientnet_v2m",
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[
        WandbEpochLogger(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
)

best_hp = tuner.get_best_hyperparameters(1)[0]
logger.info("Best hyperparameters:", best_hp.values)

wandb.config.update(best_hp.values)


# =====================
# REBUILD BEST MODEL
# =====================
model = tuner.hypermodel.build(best_hp)
model.summary()

# =====================
# PHASE 1: TRAIN HEAD
# =====================
logger.info("\n Training classifier head with best hyperparameters...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[
        WandbEpochLogger(),
        ConfidenceLogger(val_ds),
    ],
)


# =====================
# PHASE 2: FINE-TUNING
# =====================
# find EfficientNet backbone
base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

assert base_model is not None, "EfficientNet backbone not found"

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

logger.info("\n Fine-tuning model...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[
        WandbEpochLogger(),
        ConfidenceLogger(val_ds),
    ],
)


final_metrics = model.evaluate(val_ds, return_dict=True)
logger.info(f"Final validation metrics: {final_metrics}")

wandb.log({f"final/{k}": v for k, v in final_metrics.items()})


# =====================
# SAVE MODEL
# =====================
model.save("output/efficientnet_v2_m_mushrooms")
logger.info("\n Model saved!")

wandb.save(str(log_file))