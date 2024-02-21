
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy import linalg
from tqdm import tqdm
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
BATCH_SIZE = 32
TARGET_IMAGE_SIZE = (299, 299)  # InceptionV3 expected image size
NUM_IMAGES = 5000  # Adjust this based on your dataset size
inception_model = tf.keras.applications.InceptionV3(include_top=False, 
                                                    weights='path/to/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', pooling='avg')# Paths to inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 
def preprocess_image(image):
    # Resize image to target size
    image = tf.image.resize(image, TARGET_IMAGE_SIZE)
    # Convert grayscale image to RGB by repeating the channels
    if image.shape[-1] == 1:  # Check if the image is grayscale
        image = tf.image.grayscale_to_rgb(image)
    return image
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
def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)
        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)
def calculate_fid(real_embeddings, generated_embeddings):
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


real_images_folder = 'path/to/real/images' # Paths to your image folders
generated_images_folder = 'path/to/generated/images' # Paths to your image folders
real_dataloader = create_datagenerator(real_images_folder)
generated_dataloader = create_datagenerator(generated_images_folder)
count = math.ceil(NUM_IMAGES / BATCH_SIZE)
real_image_embeddings = compute_embeddings(real_dataloader, count)
generated_image_embeddings = compute_embeddings(generated_dataloader, count)

fid = calculate_fid(real_image_embeddings, generated_image_embeddings)
print("FID Score:", fid)
