import pinecone
from PIL import Image
import torch
import pandas as pd
import math
import shutil
import numpy as np

# Connect to Pinecone
pinecone = pinecone.connect(api_key="YOUR_API_KEY")
vector_index = pinecone.Index("image_vectors")

# Define the batch size
batch_size = 16

# Compute how many batches are needed
batches = math.ceil(len(images_files) / batch_size)

#
# Process each batch
#
for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    # Select the images for the current batch
    batch_files = images_files[i*batch_size : (i+1)*batch_size]

    # Preprocess all images
    images_preprocessed = torch.stack([preprocess(image) for image in batch_files]).to(device)

    # Encode the images
    with torch.no_grad():
        images_features = model.encode_image(images_preprocessed)

    # Normalize feature vectors
    images_features /= images_features.norm(dim=-1, keepdim=True)

    # Convert to numpy and upsert into Pinecone
    encoded_vectors = images_features.cpu().numpy()
    vector_index.upsert(batch_files, encoded_vectors)
