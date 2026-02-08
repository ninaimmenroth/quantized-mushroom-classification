import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import os


# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
IMG_SIZE = 240
BATCH_SIZE = 16
SEED = 42

# Which dataset to evaluate on: 'val' or 'test'
EVAL_SPLIT = "test"  # Change to "val" if you want to use validation set

# Model paths - compare float vs quantized
MODEL_NAME = "efficientnet_v2_m_mushrooms" # change here based on the model to evaluate
EVAL_OUT_DIR = MODEL_NAME + "_eval"
FLOAT_MODEL_PATH = "output/" + MODEL_NAME
TFLITE_MODEL_PATH = "output/" + MODEL_NAME + "_int8.tflite"

# Create output folder if it doesn't exist
os.makedirs("output/" + EVAL_OUT_DIR, exist_ok=True)

# SMOKE TEST MODE - Set to True for quick testing with minimal data
SMOKE_TEST = False  # Change to True for smoke test


def softmax(x):
    """Apply softmax to convert logits to probabilities."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


def load_and_pad_image(path, label, num_classes):
    """Load image for TFLite evaluation."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    label_onehot = tf.one_hot(label, num_classes)
    return image, label_onehot


def get_file_paths_and_labels(root_dir):
    """Get file paths and labels from directory structure."""
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


def build_dataset(dataset_dir, fold, batch_size, img_size, num_classes, shuffle=False, seed=42):
    """Build TensorFlow dataset."""
    fold_dir = os.path.join(dataset_dir, fold)
    paths, labels, class_names = get_file_paths_and_labels(fold_dir)
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)
    
    ds = ds.map(
        lambda p, l: load_and_pad_image(p, l, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds, class_names


def evaluate_keras_model(
    model,
    dataset,
    class_names,
    edibility_csv=None,
    species_to_edible=None,
    top_k=(3, 5),
    prefix="float"
):
    """
    Evaluate Keras model (original evaluation function).
    """
    print(f"\nEvaluating {prefix} model...")
    
    # Limit batches in smoke test mode
    eval_dataset = dataset.take(2) if SMOKE_TEST else dataset
    if SMOKE_TEST:
        print("SMOKE TEST MODE: Processing only 2 batches")
    
    # Collect predictions
    y_true, y_pred, y_prob = [], [], []
    for images, labels in eval_dataset:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    return compute_metrics(
        y_true, y_pred, y_prob, class_names,
        edibility_csv, species_to_edible, top_k, prefix
    )


def evaluate_tflite_model(
    tflite_path,
    dataset,
    class_names,
    edibility_csv=None,
    species_to_edible=None,
    top_k=(3, 5),
    prefix="tflite"
):
    """
    Evaluate TFLite INT8 quantized model.
    Handles quantization/dequantization for INT8 input/output.
    """
    print(f"\nEvaluating {prefix} model from {tflite_path}...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    print(f"Input quantization - scale: {input_scale}, zero_point: {input_zero_point}")
    print(f"Output quantization - scale: {output_scale}, zero_point: {output_zero_point}")

    # Collect predictions
    y_true, y_pred, y_prob = [], [], []
    
    # Limit batches in smoke test mode
    eval_dataset = dataset.take(2) if SMOKE_TEST else dataset
    
    print("Running inference on validation set...")
    if SMOKE_TEST:
        print("SMOKE TEST MODE: Processing only 2 batches")
    
    batch_num = 0
    for images, labels in eval_dataset:
        batch_num += 1
        if batch_num % 10 == 0 and not SMOKE_TEST:
            print(f"Processing batch {batch_num}...")
        
        batch_probs = []
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            # Get single image
            img = images[i:i+1].numpy()
            
            # Quantize input (float32 -> int8)
            img_int8 = (img / input_scale + input_zero_point).astype(np.int8)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_int8)
            interpreter.invoke()
            
            # Get output and dequantize (int8 -> float32)
            output_int8 = interpreter.get_tensor(output_details[0]['index'])
            output = (output_int8.astype(np.float32) - output_zero_point) * output_scale
            
            # Apply softmax to convert logits to probabilities
            output_probs = softmax(output[0])
            
            batch_probs.append(output_probs)
        
        batch_probs = np.array(batch_probs)
        y_prob.append(batch_probs)
        y_pred.append(np.argmax(batch_probs, axis=1))
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    return compute_metrics(
        y_true, y_pred, y_prob, class_names,
        edibility_csv, species_to_edible, top_k, prefix
    )


def compute_metrics(
    y_true,
    y_pred,
    y_prob,
    class_names,
    edibility_csv=None,
    species_to_edible=None,
    top_k=(3, 5),
    prefix=""
):
    """
    Compute all metrics from predictions.
    Shared between Keras and TFLite evaluation.
    """
    metrics = {}

    # Species-level metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Only compute AUC if we have samples from multiple classes
    # In smoke test mode, we might not have all classes represented
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1 and y_prob.shape[1] == len(class_names):
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError as e:
            print(f"  Warning: Could not compute AUC: {e}")
            auc = 0.0
    else:
        print(f"  Warning: Skipping AUC calculation (only {len(unique_classes)} classes in eval set)")
        auc = 0.0
    
    cm_species = confusion_matrix(y_true, y_pred)

    metrics.update({
        f"{prefix}/accuracy_species": acc,
        f"{prefix}/f1_macro_species": f1_macro,
        f"{prefix}/auc_macro_species": auc
    })

    print(f"\n{prefix.upper()} Model Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  AUC Macro: {auc:.4f}")

    # Top-k accuracy
    for k in top_k:
        topk = np.mean([y_true[i] in np.argsort(y_prob[i])[-k:] for i in range(len(y_true))])
        metrics[f"{prefix}/top_{k}_accuracy"] = topk
        print(f"  Top-{k} Accuracy: {topk:.4f}")

    # Edibility metrics
    if edibility_csv is not None and species_to_edible is None:
        ed_df = pd.read_csv(edibility_csv)
        # Convert spaces to underscores to match folder names
        ed_df["Species_Label"] = ed_df["Species_Label"].str.replace(" ", "_")
        # Map to bool
        species_to_edible = dict(zip(
            ed_df["Species_Label"],
            ed_df["Edible"].map({"Yes": True, "No": False})
        ))

    if species_to_edible is not None:
        true_species = [class_names[i] for i in y_true]
        pred_species = [class_names[i] for i in y_pred]

        true_edible = np.array([species_to_edible.get(s, False) for s in true_species])
        pred_edible = np.array([species_to_edible.get(s, False) for s in pred_species])

        cm_edibility = confusion_matrix(true_edible, pred_edible, labels=[True, False])
        prec_edible = precision_score(true_edible, pred_edible, zero_division=0)
        
        # Only compute AUC if we have both edible and inedible samples
        if len(np.unique(true_edible)) > 1:
            try:
                auc_edible = roc_auc_score(true_edible, pred_edible)
            except ValueError as e:
                print(f"  Warning: Could not compute edibility AUC: {e}")
                auc_edible = 0.0
        else:
            print(f"  Warning: Skipping edibility AUC (only one class present)")
            auc_edible = 0.0
        
        metrics[f"{prefix}/precision_edible"] = prec_edible
        metrics[f"{prefix}/auc_edible"] = auc_edible
        
        print(f"  Precision (Edible): {prec_edible:.4f}")
        print(f"  AUC (Edible): {auc_edible:.4f}")

        # Store confusion matrices for plotting
        metrics[f"{prefix}/cm_edibility"] = cm_edibility

    # Store species confusion matrix
    metrics[f"{prefix}/cm_species"] = cm_species
    
    return metrics, species_to_edible


def plot_confusion_matrices(metrics_dict, class_names, output_dir="output"):
    """Plot and save confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_type in ["float", "tflite"]:
        # Species confusion matrix
        if f"{model_type}/cm_species" in metrics_dict:
            cm_species = metrics_dict[f"{model_type}/cm_species"]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                cm_species,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            plt.xlabel("Predicted Species")
            plt.ylabel("True Species")
            plt.title(f"Species Confusion Matrix ({model_type.upper()})")
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"{EVAL_OUT_DIR}/confusion_matrix_species_{model_type}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved species confusion matrix to {save_path}")
            
            if wandb.run is not None:
                wandb.log({f"confusion_matrices/{model_type}_species": wandb.Image(fig)})
            plt.close(fig)

        # Edibility confusion matrix
        if f"{model_type}/cm_edibility" in metrics_dict:
            cm_edibility = metrics_dict[f"{model_type}/cm_edibility"]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_edibility,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=["Edible", "Inedible"],
                yticklabels=["Edible", "Inedible"],
                ax=ax,
                cbar_kws={'label': 'Count'}
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Edibility Confusion Matrix ({model_type.upper()})")
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f"{EVAL_OUT_DIR}/confusion_matrix_edibility_{model_type}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved edibility confusion matrix to {save_path}")
            
            if wandb.run is not None:
                wandb.log({f"confusion_matrices/{model_type}_edibility": wandb.Image(fig)})
            plt.close(fig)


def compare_models(float_metrics, tflite_metrics):
    """Print comparison between float and quantized models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison = []
    
    for key in float_metrics:
        # Skip confusion matrices and other non-scalar values
        if key.startswith("float/") and not key.endswith(("_cm_species", "_cm_edibility")):
            metric_name = key.replace("float/", "")
            tflite_key = f"tflite/{metric_name}"
            
            if tflite_key in tflite_metrics:
                float_val = float_metrics[key]
                tflite_val = tflite_metrics[tflite_key]
                
                # Skip if these are arrays (shouldn't happen, but being safe)
                if isinstance(float_val, (int, float, np.integer, np.floating)):
                    diff = tflite_val - float_val
                    diff_pct = (diff / float_val * 100) if float_val != 0 else 0
                    
                    comparison.append({
                        "Metric": metric_name,
                        "Float": f"{float_val:.4f}",
                        "TFLite INT8": f"{tflite_val:.4f}",
                        "Difference": f"{diff:+.4f} ({diff_pct:+.2f}%)"
                    })
    
    if not comparison:
        print("No metrics to compare (likely due to smoke test mode with limited data)")
        return
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    if wandb.run is not None:
        wandb.log({"comparison_table": wandb.Table(dataframe=df)})
    
    # Save to CSV
    df.to_csv(f"output/{EVAL_OUT_DIR}/model_comparison.csv", index=False)
    print(f"\nComparison saved to output/{EVAL_OUT_DIR}/model_comparison.csv")


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="mushroom-classification",
        name=f"eval_float_vs_quantized_{EVAL_SPLIT}",
        config={
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "eval_split": EVAL_SPLIT,
            "smoke_test": SMOKE_TEST,
        }
    )

    # =====================
    # LOAD CLASS NAMES
    # =====================
    eval_dir = os.path.join(DATASET_DIR, EVAL_SPLIT)
    class_names = sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d))
    )
    NUM_CLASSES = len(class_names)
    
    print(f"Found {NUM_CLASSES} classes")
    print(f"Evaluating on: {EVAL_SPLIT} set")

    # =====================
    # BUILD DATASET
    # =====================
    eval_ds, _ = build_dataset(
        dataset_dir=DATASET_DIR,
        fold=EVAL_SPLIT,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        shuffle=False,
        seed=SEED,
    )

    # =====================
    # EVALUATE FLOAT MODEL
    # =====================
    print("\n" + "="*60)
    print("LOADING FLOAT MODEL")
    print("="*60)
    
    float_model = tf.keras.models.load_model(FLOAT_MODEL_PATH)
    float_model.summary()
    
    float_metrics, species_to_edible = evaluate_keras_model(
        model=float_model,
        dataset=eval_ds,
        class_names=class_names,
        edibility_csv="data/inaturalist_mushroom_taxon_id.csv",
        top_k=(3, 5),
        prefix="float"
    )

    # =====================
    # EVALUATE TFLITE MODEL
    # =====================
    print("\n" + "="*60)
    print("LOADING TFLITE MODEL")
    print("="*60)
    
    tflite_metrics, _ = evaluate_tflite_model(
        tflite_path=TFLITE_MODEL_PATH,
        dataset=eval_ds,
        class_names=class_names,
        species_to_edible=species_to_edible,  # Reuse from float model
        top_k=(3, 5),
        prefix="tflite"
    )

    # =====================
    # COMPARE MODELS
    # =====================
    compare_models(float_metrics, tflite_metrics)

    # =====================
    # PLOT CONFUSION MATRICES
    # =====================
    all_metrics = {**float_metrics, **tflite_metrics}
    plot_confusion_matrices(all_metrics, class_names)

    # =====================
    # LOG TO WANDB
    # =====================
    # Remove confusion matrices before logging (they're already plotted)
    metrics_to_log = {
        k: v for k, v in all_metrics.items()
        if not k.endswith("_cm_species") and not k.endswith("_cm_edibility")
    }
    wandb.log(metrics_to_log)

    # Model size comparison
    # Calculate total size of SavedModel directory (includes all files)
    if os.path.exists(FLOAT_MODEL_PATH):
        float_size = 0
        for dirpath, dirnames, filenames in os.walk(FLOAT_MODEL_PATH):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                float_size += os.path.getsize(filepath)
        float_size = float_size / (1024 * 1024)  # Convert to MB
    else:
        # Fallback: approximate from weights
        float_size = sum([w.numpy().nbytes for w in float_model.weights]) / (1024 * 1024)
    
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    
    print("\n" + "="*60)
    print("MODEL SIZE COMPARISON")
    print("="*60)
    print(f"Float model (SavedModel dir): {float_size:.2f} MB")
    print(f"TFLite INT8 model: {tflite_size:.2f} MB")
    print(f"Compression ratio: {float_size / tflite_size:.2f}x")
    print(f"Size reduction: {((float_size - tflite_size) / float_size * 100):.1f}%")
    
    wandb.log({
        "model_size/float_mb": float_size,
        "model_size/tflite_mb": tflite_size,
        "model_size/compression_ratio": float_size / tflite_size,
    })

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    if SMOKE_TEST:
        print("\n⚠️  SMOKE TEST MODE was enabled")
        print("Results are based on only 2 batches and may not be representative.")
        print("Set SMOKE_TEST = False for full evaluation.\n")
    
    wandb.finish()