from sentence_transformers import SentenceTransformer
import requests
import torch
from PIL import Image
from io import BytesIO

image_encoder = SentenceTransformer('clip-ViT-B-32')

def function(text):
    return f'this is a sample function with the text as : {text}'


def encode_image_query():

    print("image encoding has started")

    with open("img.jpg", "rb") as f:
        image_bytes = f.read()

    img = Image.open(BytesIO(image_bytes))
    
   # Encode the image
    image = Image.open(img)
    image_input = image_encoder.encode(image, convert_to_tensor=True)
    image_features = image_input.flatten()

    # Convert the NumPy array to a list
    image_features_list = image_features.tolist()

    print("Image encoding has ended")

    return image_features.cpu().numpy().tolist()