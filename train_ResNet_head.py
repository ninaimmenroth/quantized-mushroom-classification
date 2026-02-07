import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os
import wandb
import pandas as pd
from dataloader import build_dataset
from ResNet_Classifier import ResNet50Classifier
from evaluate import evaluate_model


# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
IMG_SIZE = 240          # final size for model input
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = None      # auto-detected
SEED = 42
OUTPUT_DIR = "output"
MODEL_NAME = "ResNet50_Mushrooms.h5"

print(tf.config.list_physical_devices())

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

wandb.init(
    entity="luke-d-richard-berliner-hochschule-f-r-technik",
    project="mushroom-classification",
    name="resnet50_head",
    config={
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_head": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
    }
)

AUTOTUNE = tf.data.AUTOTUNE

# =====================
# WANDB LOGGER
# =====================
class WandbLogger(tf.keras.callbacks.Callback):
    """Logs training & validation metrics to W&B per epoch, including evaluation metrics."""
    def __init__(self, val_ds, class_names, edibility_csv=None):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.edibility_csv = edibility_csv
        # Preprocess edibility CSV once
        self.species_to_edible = None
        if edibility_csv is not None:
            ed_df = pd.read_csv(edibility_csv)
            ed_df["Species_Label"] = ed_df["Species_Label"].str.replace(" ", "_")
            self.species_to_edible = dict(
                zip(ed_df["Species_Label"], ed_df["Edible"].map({"Yes": True, "No": False}))
            )

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log({f"Training Logs/{k}": float(v) for k, v in logs.items()})

        # Evaluate without confusion matrices
        # eval_metrics = evaluate_model(
            # self.model,
            # self.val_ds,
            # self.class_names,
            # species_to_edible=self.species_to_edible,
            # single_log=False,           # skip charts
            # plot_confusion_matrix=False # skip confusion matrices during training
        # )

        # Log evaluation metrics using same global step
        # wandb.log({f"metrics/{k}": float(v) for k, v in eval_metrics.items()})

    def on_train_end(self, logs=None):
        # Plot confusion matrices only once at the end
        evaluate_model(
            self.model,
            self.val_ds,
            self.class_names,
            species_to_edible=self.species_to_edible,
            single_log=True,
            plot_confusion_matrix=True
        )



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
val_ds, _ = build_dataset(
    dataset_dir=DATASET_DIR,
    fold = "val",
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    shuffle=False,
    seed = SEED,
)

NUM_CLASSES = len(class_names)


# =====================
# CREATE RESNET BASELINE MODEL
# =====================
classifier = ResNet50Classifier(IMG_SIZE, NUM_CLASSES, use_augmentation=True)
model = classifier.model

model.summary()


# =====================
# COMPILE
# =====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),]
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

train_cb = WandbLogger(
    val_ds, 
    class_names, 
    edibility_csv="data/inaturalist_mushroom_taxon_id.csv", 
)

# =====================
# TRAIN
# =====================
print("\n🔹 Training classifier head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[train_cb,] # early_stopping,
)


# =====================
# EVALUATE MODEL
# =====================
# metrics = evaluate_model(
    # model,
    # val_ds,
    # class_names,
    # edibility_csv="data/inaturalist_mushroom_taxon_id.csv",
    # log_wandb=True
# )

# =====================
# SAVE MODEL
# =====================
model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
model.save_weights(model_path)
print("\n Model saved!")


# Finish W&B run
wandb.finish()

