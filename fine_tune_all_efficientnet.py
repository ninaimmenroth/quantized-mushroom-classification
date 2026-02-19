import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
from efficient_all_classifier import EfficientNet_Classifier
from dataloader import build_dataset
from evaluate import evaluate_model
import wandb
import os


# =====================
# CONFIG: base models to use
# =====================
all_base_models = {
    "b0": tf.keras.applications.EfficientNetB0,
    "b1": tf.keras.applications.EfficientNetB1,
    "b2": tf.keras.applications.EfficientNetB2,
    "b3": tf.keras.applications.EfficientNetB3,
    "v2B0": tf.keras.applications.EfficientNetV2B0,
    "v2B1": tf.keras.applications.EfficientNetV2B1,
    "v2B2": tf.keras.applications.EfficientNetV2B2,
    "v2B3": tf.keras.applications.EfficientNetV2B3,
}

img_size_dict = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
    "b3": 300,
    "v2B0": 224,
    "v2B1":	240,
    "v2B2":	260,
    "v2B3":	300,

}
# =====================
# CONFIG
# =====================
TRAIN_HEAD = True
TRAIN_FINE = True
DATASET_DIR = "data/images"
IMG_SIZE = None         # final size for model input, auto-detected based on model
BATCH_SIZE = 64
EPOCHS_HEAD = 0
EPOCHS_FINE = 10
NUM_CLASSES = None      # auto-detected
SEED = 42
LR_HEAD = 1e-3
LR_FINE = 1e-5
LABEL_SMOOTHING = 0.0
DROPOUT = 0.0

print(tf.config.list_physical_devices())

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
# Loop through base models
# =====================
for model_name, BaseModelClass in all_base_models.items():
    print(f"\n=== Training {model_name} ===\n")

    # set image size
    IMG_SIZE = img_size_dict[model_name]

# initialize wandb
    wandb.init(
        entity="luke-d-richard-berliner-hochschule-f-r-technik",
        project="mushroom-classification",
        name=f"efficientnet_{model_name}",
        config={
            "train_head": TRAIN_HEAD,
            "train_fine": TRAIN_FINE,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs_head": EPOCHS_HEAD,
            "epochs_fine": EPOCHS_FINE,
            "optimizer": "Adam",
            "lr_head": LR_HEAD,
            "lr_fine": LR_FINE,
            "label_smoothing": LABEL_SMOOTHING,
            "dropout": DROPOUT,
        }
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
    # CALLBACKS
    # =====================
    train_cb = WandbLogger(
        val_ds, 
        class_names, 
        edibility_csv="data/inaturalist_mushroom_taxon_id.csv", 
    )


    class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, total_steps, warmup_steps):
            self.base_lr = base_lr
            self.total_steps = total_steps
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)

            warmup_lr = self.base_lr * (step / self.warmup_steps)
            cosine_lr = self.base_lr * 0.5 * (
                1 + tf.cos(
                    tf.constant(tf.math.pi) *
                    (step - self.warmup_steps) /
                    (self.total_steps - self.warmup_steps)
                )
            )

            return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)


    # =====================
    # BUILD/TRAIN MODEL
    # ====================

    # base_model = tf.keras.applications.EfficientNetB0(
    #     include_top=False,
    #     weights="imagenet",
    #     input_shape=(IMG_SIZE, IMG_SIZE, 3),
    # )



    # Initialize base model
    base_model = BaseModelClass(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )


    # train the model
    if TRAIN_HEAD:
        efficientnet_classifier = EfficientNet_Classifier(
            base_model=base_model,
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            use_augmentation=True,
        )

        efficientnet_classifier.compile_head(learning_rate=LR_HEAD, label_smoothing=LABEL_SMOOTHING)

        efficientnet_classifier.train_head(train_ds, val_ds, epochs_head=EPOCHS_HEAD, callbacks=[train_cb])

        # save the trained head
        efficientnet_classifier.model.save_weights(
            f"output/efficientnet_head_{model_name}.weights.h5"
        )

    if TRAIN_FINE:
        if not TRAIN_HEAD:
            # reload model
            efficientnet_classifier = EfficientNet_Classifier(
                base_model=base_model,
                img_size=IMG_SIZE,
                num_classes=NUM_CLASSES,
                dropout=DROPOUT,
                use_augmentation=True,
            )

            efficientnet_classifier.model.load_weights(
                f"output/efficientnet_head_{model_name}.weights.h5"
            )

        efficientnet_classifier.fine_tune(
            train_ds, 
            val_ds, 
            epochs_fine=EPOCHS_FINE, 
            percent_unfreeze=0.2,
            learning_rate=LR_FINE, 
            weight_decay=0,
            callbacks=[train_cb]
        )


        # =====================
        # SAVE MODEL
        # =====================
        efficientnet_classifier.model.save_weights(f"output/efficientnet_fine_{model_name}.weights.h5")
        print("\n Model saved!")


    # Finish W&B run
    wandb.finish()
