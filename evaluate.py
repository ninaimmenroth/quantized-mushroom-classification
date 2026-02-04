import tensorflow as tf
import numpy as np
import pandas as pd
import os
from dataloader import build_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
IMG_SIZE = 240          # final size for model input
BATCH_SIZE = 16
NUM_CLASSES = None      # auto-detected
SEED = 42
MODEL_PATH = "output/ResNet50_Mushrooms.keras"


AUTOTUNE = tf.data.AUTOTUNE

# =====================
# COLLECT PREDICTIONS
# =====================
def collect_predictions(model, dataset):
    y_true = []
    y_pred = []
    y_prob = []

    # append result for each batch
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)

        y_prob.append(probs)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(labels.numpy(), axis=1))

    # flatten batch results into single list
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    return y_true, y_pred, y_prob


# =====================
# STANDARD METRICS
# =====================
def compute_species_metrics(y_true, y_pred, y_prob, class_names):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("\n=== Species Classification ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Multi-class AUC
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    print(f"Macro AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    return acc, f1_macro, auc, cm


# =====================
# TOP-K ACCURACY
# =====================
def top_k_accuracy(y_true, y_prob, k):
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in top_k[i] for i in range(len(y_true))])


# =====================
# EDIBILITY METRICS
# =====================
def compute_edibility_metrics(
    y_true,
    y_pred,
    class_names,
    edibility_df,
):
    """
    edibility_df columns:
      - Species_Label
      - Edible (Yes / No)
    """

    # Map species → edible
    species_to_edible = dict(
        zip(edibility_df["species"], edibility_df["edible"])
    )

    true_species = [class_names[i] for i in y_true]
    pred_species = [class_names[i] for i in y_pred]

    true_edible = np.array([species_to_edible[s] for s in true_species])
    pred_edible = np.array([species_to_edible[s] for s in pred_species])

    precision = precision_score(true_edible,pred_edible)
    cm = confusion_matrix(true_edible, pred_edible)
    auc = roc_auc_score(true_edible, pred_edible)  # binary AUC

    cm = confusion_matrix(true_edible, pred_edible, labels=[True, False])

    tn, fp, fn, tp = cm.ravel()

    print("\n=== Edibility Safety Metrics ===")
    print("Confusion matrix (rows=true, cols=pred):")
    print("          Pred Edible  Pred Inedible")
    print(f"True Edible     {tn:4d}         {fp:4d}")
    print(f"True Inedible   {fn:4d}         {tp:4d}")

    print(f"\n⚠️ Inedible → Edible (dangerous): {fn}")
    print(f"Edible → Inedible (safe):        {fp}")

    return cm


# =====================
# MAIN ENTRY POINT
# =====================
def evaluate_model(
    model_path,
    dataset,
    class_names,
    edibility_csv=None,
    top_k=(3, 5),
):
    # load the model
    model = tf.keras.models.load_model(model_path)

    # Headline metrics
    loss, acc = model.evaluate(dataset, verbose=1)
    print(f"\nKeras accuracy: {acc:.4f}")

    # Full predictions
    y_true, y_pred, y_prob = collect_predictions(model, dataset)

    # Species metrics
    compute_species_metrics(y_true, y_pred, class_names)

    # Top-k
    for k in top_k:
        tk = top_k_accuracy(y_true, y_prob, k)
        print(f"Top-{k} accuracy: {tk:.4f}")

    # Edibility metrics
    if edibility_csv is not None:
        edibility_df = pd.read_csv(edibility_csv)
        compute_edibility_metrics(
            y_true, y_pred, class_names, edibility_df
        )


# =====================
# BUILD TF.DATA DATASETS
# =====================

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
# EVALUATE MODEL
# =====================
evaluate_model(MODEL_PATH, val_ds, class_names)