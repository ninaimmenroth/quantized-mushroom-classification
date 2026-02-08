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

TFLITE_MODEL_PATH = "output/efficientnet_v2_m_mushrooms_int8.tflite"

# SMOKE TEST MODE - Set to True for quick testing with minimal data
SMOKE_TEST = True  # Change to True for smoke test


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


def evaluate_tflite_model(
    tflite_path,
    dataset,
    class_names,
    edibility_csv=None,
    species_to_edible=None,
    top_k=(3, 5),
    plot_confusion_matrix=False
):
    """
    Evaluate TFLite INT8 quantized model.
    Adapted from your partner's evaluate_model function.
    """
    print(f"\nEvaluating TFLite model from {tflite_path}...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    print(f"Input quantization - scale: {input_scale}, zero_point: {input_zero_point}")
    print(f"Output quantization - scale: {output_scale}, zero_point: {output_zero_point}")

    # Collect predictions (similar to original evaluate_model)
    y_true, y_pred, y_prob = [], [], []
    
    # Limit batches in smoke test mode
    eval_dataset = dataset.take(2) if SMOKE_TEST else dataset
    
    print("Running inference on validation set...")
    if SMOKE_TEST:
        print("SMOKE TEST MODE: Processing only 2 batches")
    
    batch_num = 0
    for images, labels in eval_dataset:
        batch_num += 1
        if batch_num % 10 == 0:
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
        "accuracy_species": acc,
        "f1_macro_species": f1_macro,
        "auc_macro_species": auc
    })

    print(f"\nSpecies Classification Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  AUC Macro: {auc:.4f}")

    # Top-k accuracy
    for k in top_k:
        topk = np.mean([y_true[i] in np.argsort(y_prob[i])[-k:] for i in range(len(y_true))])
        metrics[f"top_{k}_accuracy"] = topk
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
        
        metrics["precision_edible"] = prec_edible
        metrics["auc_edible"] = auc_edible
        
        print(f"\nEdibility Classification Results:")
        print(f"  Precision (Edible): {prec_edible:.4f}")
        print(f"  AUC (Edible): {auc_edible:.4f}")
        print(f"\nEdibility Confusion Matrix:")
        print(f"                Predicted Edible  Predicted Inedible")
        print(f"True Edible     {cm_edibility[0,0]:16d}  {cm_edibility[0,1]:18d}")
        print(f"True Inedible   {cm_edibility[1,0]:16d}  {cm_edibility[1,1]:18d}")

    # Log to wandb if active
    if wandb.run is not None:
        wandb.log({f"Evaluation Metrics/{k}": v for k, v in metrics.items()})

    # Plot confusion matrices if requested
    if plot_confusion_matrix:
        # Species confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_species,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        plt.xlabel("Predicted Species")
        plt.ylabel("True Species")
        plt.title("Species Confusion Matrix (TFLite INT8)")
        plt.tight_layout()
        
        os.makedirs("output", exist_ok=True)
        plt.savefig("output/confusion_matrix_species_tflite.png", dpi=150, bbox_inches='tight')
        print("\nSaved species confusion matrix to output/confusion_matrix_species_tflite.png")
        
        if wandb.run is not None:
            wandb.log({"Confusion_Matrices/Species_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # Edibility confusion matrix
        if species_to_edible is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_edibility,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=["Edible", "Inedible"],
                yticklabels=["Edible", "Inedible"],
                ax=ax
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Edibility Confusion Matrix (TFLite INT8)")
            plt.tight_layout()
            
            plt.savefig("output/confusion_matrix_edibility_tflite.png", dpi=150, bbox_inches='tight')
            print("Saved edibility confusion matrix to output/confusion_matrix_edibility_tflite.png")
            
            if wandb.run is not None:
                wandb.log({"Confusion_Matrices/Edibility_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

    return metrics


if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="mushroom-classification",
        name=f"eval_tflite_int8_{EVAL_SPLIT}",
        config={
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "model": "efficientnet_v2_m_int8",
            "eval_split": EVAL_SPLIT,
            "smoke_test": SMOKE_TEST,
        }
    )

    # =====================
    # BUILD DATASET
    # =====================
    eval_dir = os.path.join(DATASET_DIR, EVAL_SPLIT)
    eval_paths, eval_labels, class_names = get_file_paths_and_labels(eval_dir)
    
    NUM_CLASSES = len(class_names)
    print(f"Found {NUM_CLASSES} classes: {class_names[:5]}... (showing first 5)")
    print(f"Evaluating on: {EVAL_SPLIT} set")
    print(f"Total samples: {len(eval_paths)}")

    eval_ds = tf.data.Dataset.from_tensor_slices((eval_paths, eval_labels))
    eval_ds = eval_ds.map(
        lambda p, l: load_and_pad_image(p, l, NUM_CLASSES),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    eval_ds = eval_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # =====================
    # EVALUATE MODEL
    # =====================
    metrics = evaluate_tflite_model(
        tflite_path=TFLITE_MODEL_PATH,
        dataset=eval_ds,
        class_names=class_names,
        edibility_csv="data/inaturalist_mushroom_taxon_id.csv",
        top_k=(3, 5),
        plot_confusion_matrix=True
    )

    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("output/tflite_evaluation_metrics.csv", index=False)
    print("\nMetrics saved to output/tflite_evaluation_metrics.csv")

    # Model size
    model_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"\nTFLite model size: {model_size:.2f} MB")
    
    wandb.log({"model_size_mb": model_size})

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    if SMOKE_TEST:
        print("\n⚠️  SMOKE TEST MODE was enabled")
        print("Results are based on only 2 batches and may not be representative.")
        print("Set SMOKE_TEST = False for full evaluation.\n")
    
    wandb.finish()