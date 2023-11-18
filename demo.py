import pinecone
import cv2
import numpy as np
from zipfile import ZipFile
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Authenticate with Pinecone using your API key
pinecone.init(api_key='878f92d2-ce36-4be5-bbc0-05a56ff092fc', environment='gcp-starter')

# Name for the Pinecone index
index_name = 'adm4'

# Create a Pinecone index for image embeddings
pinecone.create_index(index_name, dimension=4096)  # VGG16 produces 4096-dimensional embeddings

# Function to extract VGG16 embeddings from images
def extract_embeddings(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(img_array)
    flattened_features = features.flatten()

    return flattened_features / np.linalg.norm(flattened_features)  # Normalize the embeddings

# Path to the zip file containing images
zip_file_path = '"D:\zip\img.zip"'

# Extract images from the zip file and compute embeddings
with ZipFile(zip_file_path, 'r') as zip_ref:
    image_files = zip_ref.namelist()
    for img_file in image_files:
        with zip_ref.open(img_file) as file:
            img = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), -1)
            embeddings = extract_embeddings(img)
            # Assuming 'img_file' is a unique identifier for each image
            pinecone.embed(index_name, ids=[img_file], embeddings=[embeddings])

print("Embeddings stored in Pinecone index.")
