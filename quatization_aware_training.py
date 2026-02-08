import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import random
from evaluate import evaluate_model
import wandb
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
MODEL_WEIGHTS_PATH = "output/efficientnet_v2_m_mushrooms.h5"
MODEL_PATH = "output/efficientnet_v2_m_mushrooms"
OUTPUT_DIR = "output"
MODEL_CONFIG = "output/efficientnet_v2_config/model_config.json"
IMG_SIZE = 240
BATCH_SIZE = 64
EPOCHS_QAT = 5
SEED = 42
SMOKE_TEST = True

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

wandb.init(
    project="mushroom-classification",
    name="qat",
    config={
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_qat": EPOCHS_QAT,
        "learning_rate": 1e-5,
        "dataset": "custom_mushrooms",
    }
)

# =====================
# PLOTTING & LOGGING
# =====================

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

logger.info("\n Logger initialized")
logger.info(f"\n Log file: {log_file}")

logger.info(tf.config.list_physical_devices())

def log_exceptions(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

sys.excepthook = log_exceptions


# =====================
# REPRODUCIBILITY
# =====================
if SMOKE_TEST:
    EPOCHS_QAT = 2
    BATCH_SIZE = 4
    # =====================
    # REPRODUCIBILITY - Only for smoke tests
    # =====================
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    logger.info(f"\n Deterministic mode enabled for smoke test (seed={SEED})")
else:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


# =====================
# DATA LOADING (same as training)
# =====================
def load_and_pad_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def get_file_paths_and_labels(root_dir):
    class_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    paths, labels = [], []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])

    return paths, labels, class_names


train_paths, train_labels, class_names = get_file_paths_and_labels(
    os.path.join(DATASET_DIR, "train")
)
val_paths, val_labels, _ = get_file_paths_and_labels(
    os.path.join(DATASET_DIR, "val")
)

NUM_CLASSES = len(class_names)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths), seed=SEED)
train_ds = train_ds.map(load_and_pad_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_pad_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

if SMOKE_TEST:
    logger.info("\n !!! SMOKE TEST MODE: Using tiny dataset")

    train_ds = train_ds.take(2)  
    val_ds = val_ds.take(2)

# =====================
# LOAD FLOAT MODEL
# =====================

def build_model_from_config(cfg):
    base_model = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights=None,
        input_shape=(cfg["img_size"], cfg["img_size"], 3),
    )

    contrast_enabled = cfg["data_augmentation"]["contrast"]

    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(cfg["data_augmentation"]["rotation"]),
        tf.keras.layers.RandomZoom(cfg["data_augmentation"]["zoom"]),
        tf.keras.layers.RandomContrast(
            0.2 if contrast_enabled else 1e-6
        ),
    ], name="sequential")

    inputs = tf.keras.Input(shape=(cfg["img_size"], cfg["img_size"], 3))
    x = aug(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if cfg["head"]["batch_norm"]:
        x = tf.keras.layers.BatchNormalization()(x)

    if cfg["head"]["type"] == "dense":
        x = tf.keras.layers.Dense(cfg["head"]["units"], activation="relu")(x)

    x = tf.keras.layers.Dropout(cfg["head"]["dropout"])(x)

    outputs = tf.keras.layers.Dense(cfg["num_classes"], activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


logger.info("\n Loading trained float model weights and config...")
with open(MODEL_CONFIG) as f:
    cfg = json.load(f)
float_model = build_model_from_config(cfg)
float_model.summary()
float_model.load_weights(MODEL_WEIGHTS_PATH)


# Apply QAT graph rewrite - insert FakeQuant nodes into the network
qat_model = tfmot.quantization.keras.quantize_model(float_model)

# FREEZE BATCHNORM *HERE*
for layer in qat_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Compile AFTER changing trainable flags
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5),
    ],
)

qat_model.summary()

# =====================
# QAT FINETUNING
# =====================
logger.info("\n Starting QAT fine-tuning...")
qat_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_QAT,
)

# =====================
# EVALUATION
# =====================
metrics = qat_model.evaluate(val_ds, return_dict=True)
logger.info("\n QAT validation metrics:", metrics)

# =====================
# EXPORT INT8 TFLITE
# =====================
logger.info("\n Converting to fully INT8 TFLite...")

def representative_dataset():
    for images, _ in train_ds.take(100):
        yield [images]

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

os.makedirs("output", exist_ok=True)
with open("output/efficientnet_v2_m_mushrooms_int8.tflite", "wb") as f:
    f.write(tflite_model)

logger.info("\n INT8 TFLite model saved to:")
logger.info("\n output/efficientnet_v2_m_mushrooms_int8.tflite")
