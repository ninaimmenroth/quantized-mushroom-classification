import tensorflow as tf
import numpy as np
import os
import random
import logging
import sys
from pathlib import Path
from datetime import datetime
import wandb

# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
OUTPUT_DIR = "output"
MODEL_NAME = "efficientnet_v2_m_mushrooms"
MODEL_PATH = OUTPUT_DIR + "/" + MODEL_NAME
IMG_SIZE = 240
BATCH_SIZE = 64
SEED = 42
SMOKE_TEST = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

wandb.init(
    project="mushroom-classification",
    name="ptq-int8"+MODEL_NAME,
    config={
        "model_name": MODEL_NAME,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "quantization": "post-training-int8",
        "dataset": "custom_mushrooms",
    }
)

# =====================
# LOGGING
# =====================
def setup_logger(log_dir="logs", run_name="run"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path

logger, log_file = setup_logger(run_name=wandb.run.name if wandb.run else "local_run")
logger.info("Logger initialized")
logger.info(f"Log file: {log_file}")

# =====================
# REPRODUCIBILITY
# =====================
if SMOKE_TEST:
    BATCH_SIZE = 4
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    logger.info(f"Smoke test mode enabled (seed={SEED})")
else:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# =====================
# DATA LOADING
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

# Load training data for representative dataset
train_paths, train_labels, class_names = get_file_paths_and_labels(
    os.path.join(DATASET_DIR, "train")
)
val_paths, val_labels, _ = get_file_paths_and_labels(
    os.path.join(DATASET_DIR, "val")
)

NUM_CLASSES = len(class_names)

# Create dataset for calibration
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths), seed=SEED)
train_ds = train_ds.map(load_and_pad_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Create validation dataset
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_pad_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

if SMOKE_TEST:
    logger.info("SMOKE TEST MODE: Using tiny dataset")
    train_ds = train_ds.take(2)  
    val_ds = val_ds.take(2)

# =====================
# LOAD FLOAT MODEL
# =====================
logger.info(f"Loading trained model from {MODEL_PATH}...")
float_model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# Evaluate float model
logger.info("Evaluating float model...")
float_metrics = float_model.evaluate(val_ds, return_dict=True)
logger.info(f"Float model metrics: {float_metrics}")

# =====================
# POST-TRAINING QUANTIZATION TO INT8
# =====================
logger.info("Starting Post-Training Quantization to INT8...")

def representative_dataset():
    """
    Generator that yields representative data for calibration.
    Uses a subset of training data to determine quantization parameters.
    """
    num_calibration_batches = 100 if not SMOKE_TEST else 2
    for i, (images, _) in enumerate(train_ds.take(num_calibration_batches)):
        logger.info(f"Calibration batch {i+1}/{num_calibration_batches}")
        yield [images]

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(float_model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set representative dataset for full integer quantization
converter.representative_dataset = representative_dataset

# Enforce full integer quantization (INT8)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

logger.info("Converting model to TFLite INT8...")
tflite_model_quant = converter.convert()

# Save quantized model
output_path = os.path.join(OUTPUT_DIR, "efficientnet_v2_m_mushrooms_int8.tflite")
with open(output_path, "wb") as f:
    f.write(tflite_model_quant)

logger.info(f"INT8 TFLite model saved to: {output_path}")

# =====================
# EVALUATE QUANTIZED MODEL
# =====================
logger.info("Evaluating quantized model...")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=output_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logger.info(f"Input details: {input_details}")
logger.info(f"Output details: {output_details}")

# Get quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

logger.info(f"Input quantization - scale: {input_scale}, zero_point: {input_zero_point}")
logger.info(f"Output quantization - scale: {output_scale}, zero_point: {output_zero_point}")

# Evaluate on validation set
correct_top1 = 0
correct_top5 = 0
total = 0

for images, labels in val_ds:
    for i in range(images.shape[0]):
        # Get single image
        img = images[i:i+1].numpy()
        
        # Quantize input
        img_int8 = (img / input_scale + input_zero_point).astype(np.int8)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_int8)
        interpreter.invoke()
        
        # Get output and dequantize
        output_int8 = interpreter.get_tensor(output_details[0]['index'])
        output = (output_int8.astype(np.float32) - output_zero_point) * output_scale
        
        # Get predictions
        pred_class = np.argmax(output[0])
        top5_classes = np.argsort(output[0])[-5:]
        
        # Get true label
        true_class = np.argmax(labels[i].numpy())
        
        # Update metrics
        if pred_class == true_class:
            correct_top1 += 1
        if true_class in top5_classes:
            correct_top5 += 1
        total += 1

top1_accuracy = correct_top1 / total
top5_accuracy = correct_top5 / total

logger.info(f"\nQuantized model evaluation results:")
logger.info(f"Top-1 Accuracy: {top1_accuracy:.4f}")
logger.info(f"Top-5 Accuracy: {top5_accuracy:.4f}")

# Log to wandb
wandb.log({
    "float_accuracy": float_metrics.get('accuracy', 0),
    "float_top5": float_metrics.get('top5', 0),
    "quant_top1_accuracy": top1_accuracy,
    "quant_top5_accuracy": top5_accuracy,
    "accuracy_drop": float_metrics.get('accuracy', 0) - top1_accuracy,
})

# =====================
# MODEL SIZE COMPARISON
# =====================
float_model_size = os.path.getsize(MODEL_PATH + "/saved_model.pb") / (1024 * 1024)
quant_model_size = os.path.getsize(output_path) / (1024 * 1024)

logger.info(f"\nModel size comparison:")
logger.info(f"Float model: {float_model_size:.2f} MB")
logger.info(f"Quantized model: {quant_model_size:.2f} MB")
logger.info(f"Compression ratio: {float_model_size / quant_model_size:.2f}x")

wandb.log({
    "float_model_size_mb": float_model_size,
    "quant_model_size_mb": quant_model_size,
    "compression_ratio": float_model_size / quant_model_size,
})

logger.info("\nQuantization complete!")
wandb.finish()