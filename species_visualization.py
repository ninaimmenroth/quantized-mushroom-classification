import matplotlib.pyplot as plt
import tensorflow as tf
import os

# =====================
# CONFIG
# =====================
DATASET_DIR = "data/images"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
IMG_SIZE = 240
ROWS, COLS = 2, 8  # Total 16 images

# =====================
# IMAGE PROCESSING
# =====================
def load_and_pad_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize_with_pad maintains aspect ratio by adding black bars
    padded = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    return padded

# =====================
# DATA GATHERING
# =====================
# Get list of species (folder names)
species_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])

# We only need 16 for a 2x8 grid
selected_species = species_names[:ROWS * COLS]

# =====================
# VISUALIZE GRID
# =====================
plt.figure(figsize=(20, 4)) # Wider figure for 8 columns

for i, species in enumerate(selected_species):
    # Path to the first image in this species folder
    species_path = os.path.join(TRAIN_DIR, species)
    img_name = sorted(os.listdir(species_path))[0]
    img_path = os.path.join(species_path, img_name)
    
    # Process image
    padded_img = load_and_pad_image(img_path)
    
    # Create subplot
    plt.subplot(ROWS, COLS, i + 1)
    plt.imshow(padded_img)
    plt.title(species, fontsize=10) # Folder name as species title
    plt.axis("off")

plt.tight_layout()
plt.show()