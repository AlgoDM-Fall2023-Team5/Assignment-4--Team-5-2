import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
from PIL import Image
import os

# Initialize CLIP-based SentenceTransformer model
image_encoder = SentenceTransformer('clip-ViT-B-32')

# Connect to Pinecone
pinecone_api_key = "878f92d2-ce36-4be5-bbc0-05a56ff092fc"  # Replace with your actual Pinecone API key
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

# Specify the path to your dataset folder
dataset_folder = "C:/Users/shiri/Documents/Assg4ADM/dataset"

# Get a list of all image files in the dataset folder
image_files = [f for f in os.listdir(dataset_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Initialize the Pinecone index outside the loop
pinecone_index_name = "adm4"  # Replace with your actual Pinecone index name
pinecone_index = pinecone.Index(pinecone_index_name)

count = 0
# Loop through each image in the dataset
for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(dataset_folder, image_file)

        # Encode the image
        image = Image.open(image_path)
        image_input = image_encoder.encode(image, convert_to_tensor=True)
        image_features = image_input.flatten()

        # Convert the NumPy array to a list
        image_features_list = image_features.tolist()

        # Upsert vector into Pinecone index with image file name as tag using tuple-based upsert
        upsert_response = pinecone_index.upsert(
            vectors=[(image_file, image_features_list)]
        )
        print(count)
        count = count +1

print("Vectors ingested into Pinecone.")
