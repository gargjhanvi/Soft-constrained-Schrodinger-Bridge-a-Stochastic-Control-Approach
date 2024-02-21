
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import linalg
from tqdm import tqdm

# Check for GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Constants
BATCH_SIZE = 32
TARGET_IMAGE_SIZE = (299, 299)  # InceptionV3 expected image size
NUM_IMAGES = 5000  # Adjust this based on your dataset size

# Initialize the InceptionV3 model
inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                                                    weights='path/to/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', pooling='avg')
# Function to preprocess the image (resize and convert grayscale to RGB)
def preprocess_image(image):
    # Resize image to target size
    image = tf.image.resize(image, TARGET_IMAGE_SIZE)
    # Convert grayscale image to RGB by repeating the channels
    if image.shape[-1] == 1:  # Check if the image is grayscale
        image = tf.image.grayscale_to_rgb(image)
    return image

# Function to create a data generator
def create_datagenerator(folder_path):
    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: tf.keras.applications.inception_v3.preprocess_input(preprocess_image(x))
    )
    generator = datagen.flow_from_directory(
        folder_path,
        target_size=(299, 299),  # Original image size
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
        color_mode='rgb'  # Load images as grayscale
    )
    return generator

# Function to compute embeddings
def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)
        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)

# Function to calculate FID
def calculate_fid(real_embeddings, generated_embeddings):
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Paths to your image folders
real_images_folder = 'path/to/real/images'
generated_images_folder = 'path/to/generated/images'

# Create dataloaders
real_dataloader = create_datagenerator(real_images_folder)
generated_dataloader = create_datagenerator(generated_images_folder)

# Calculate the number of batches needed
count = math.ceil(NUM_IMAGES / BATCH_SIZE)

# Compute embeddings for real and generated images
real_image_embeddings = compute_embeddings(real_dataloader, count)
generated_image_embeddings = compute_embeddings(generated_dataloader, count)

# Calculate FID
fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
print("FID Score:", fid)
