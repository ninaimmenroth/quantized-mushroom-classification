import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import AdamW
import os

# =====================
# MANUAL DATA LOADING + RESIZE_WITH_PAD
# =====================
def load_and_pad_image(path, label, num_classes:int, img_size:int):
    """Loads a JPEG image, converts to float32, pads to IMG_SIZE while preserving aspect ratio."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize_with_pad(image, img_size, img_size)
    label = tf.one_hot(label, num_classes)
    return image, label

def make_load_and_pad_fn(num_classes: int, img_size: int):
    def _fn(path, label):
        return load_and_pad_image(path, label, num_classes, img_size)
    return _fn

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
def build_dataset(
    dataset_dir:str, 
    fold:str, 
    batch_size:int=16,
    img_size:int=240, 
    shuffle:bool=True, 
    seed:int=None,
):
    
    AUTOTUNE = tf.data.AUTOTUNE

    paths, labels, class_names = get_file_paths_and_labels(os.path.join(dataset_dir, fold))

    num_classes = len(class_names)

    # =====================
    # BUILD TF.DATA DATASETS
    # =====================
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if fold == "train":
        dataset = dataset.shuffle(len(paths), seed=seed)
    dataset = dataset.map(make_load_and_pad_fn(num_classes, img_size), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset, class_names
