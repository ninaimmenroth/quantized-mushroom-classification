import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from dataloader import build_dataset
from ResNet_Classifier import ResNet50Classifier
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


def evaluate_model(model, dataset, class_names, edibility_csv=None, top_k=(3, 5), log_wandb=True):
    """
    Evaluate model and optionally log metrics to W&B.
    Returns a dictionary of metrics for convenience.
    """
    # Collect predictions
    y_true, y_pred, y_prob = [], [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_prob.append(probs)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(labels.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = {}

    # Species-level metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    cm_species = confusion_matrix(y_true, y_pred)

    metrics.update({
        "accuracy_species": acc,
        "f1_macro_species": f1_macro,
        "auc_macro_species": auc
    })

    # Top-k accuracy
    for k in top_k:
        topk = np.mean([y_true[i] in np.argsort(y_prob[i])[-k:] for i in range(len(y_true))])
        metrics[f"top_{k}_accuracy"] = topk

    # Edibility metrics
    if edibility_csv is not None:
        ed_df = pd.read_csv(edibility_csv)
        # Convert spaces to underscores to match folder names
        ed_df["Species_Label"] = ed_df["Species_Label"].str.replace(" ", "_")
        # Map to bool
        species_to_edible = dict(zip(ed_df["Species_Label"], ed_df["Edible"].map({"Yes": True, "No": False})))

        true_species = [class_names[i] for i in y_true]
        pred_species = [class_names[i] for i in y_pred]

        true_edible = np.array([species_to_edible.get(s, False) for s in true_species])
        pred_edible = np.array([species_to_edible.get(s, False) for s in pred_species])

        cm_edibility = confusion_matrix(true_edible, pred_edible, labels=[True, False])
        metrics["precision_edible"] = precision_score(true_edible, pred_edible)
        metrics["auc_edible"] = roc_auc_score(true_edible, pred_edible)

    if log_wandb:
        # Log scalar metrics
        wandb.log(metrics, step=None)


    # Log species confusion matrix as well
    if log_wandb:
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(cm_species, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Species Confusion Matrix")

        wandb.log({"Species_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        # Plot confusion matrix and log to W&B
        if edibility_csv is not None:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm_edibility, annot=True, fmt="d", cmap="Blues", xticklabels=["Edible","Inedible"], yticklabels=["Edible","Inedible"], ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Edibility Confusion Matrix")

            wandb.log({"edibility_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

    return metrics

if __name__ == "__main__":
    # Initialize wandb run
    wandb.init(project="mushroom-classification", name="eval_run")

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
    # LOAD MODEL
    # =====================

    # recreate architecture
    classifier = ResNet50Classifier(img_size=IMG_SIZE, num_classes=NUM_CLASSES, use_augmentation=False)
    model = classifier.model    # get the model
    model.load_weights("output/ResNet50_Mushrooms.h5")  # load the weights
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    # =====================
    # EVALUATE MODEL
    # =====================
    evaluate_model(model, val_ds, class_names, "data/inaturalist_mushroom_taxon_id.csv")