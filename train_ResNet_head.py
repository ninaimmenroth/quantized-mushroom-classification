import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os

# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
IMG_SIZE = 240          # final size for model input
BATCH_SIZE = 16
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
NUM_CLASSES = None      # auto-detected
SEED = 42

print(tf.config.list_physical_devices())

AUTOTUNE = tf.data.AUTOTUNE

# =====================
# MANUAL DATA LOADING + RESIZE_WITH_PAD
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
# TRAIN HEAD
# =====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

print("\n🔹 Training classifier head...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
)


# =====================
# SAVE MODEL
# =====================
model.save("output/ResNet50_mushrooms")
print("\n Model saved!")