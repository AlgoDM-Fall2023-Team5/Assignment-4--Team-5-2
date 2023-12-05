import streamlit as st
import requests
from PIL import Image
import pinecone
from io import BytesIO
# from sentence_transformers import SentenceTransformer
import json  # Add this import
import boto3
from api import image_search 

import torch
import clip


st.set_option('browser.gatherUsageStats', False)


url = "http://127.0.0.1:8000"
path = "C:/Users/shiri/Documents/Assg4ADM/dataset"
pinecone_api_key = "7f78befa-055d-41ac-a90a-cff6a5282d66"
index_name = "adm4"

# image_encoder = SentenceTransformer('clip-ViT-B-32')




# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to encode search query
# def encode_search_query(search_query):
#     with torch.no_grad():
#         text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
#         text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
#     return text_encoded





def compute_clip_features(image_file):
    # Load the image from the file
    image = Image.open(image_file)
    
    # Preprocess the image
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode the image to compute the feature vector and normalize it
        image_features = model.encode_image(image_preprocessed)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vector back to the CPU and convert to numpy
    return image_features.cpu().numpy()








st.title("Image Similarity Search")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

def encode_image(uploaded_image, encoder):
    image_bytes = uploaded_image.read()
    image_pil = Image.open(BytesIO(image_bytes))

    # Encode the image
    image_input = encoder.encode(image_pil, convert_to_tensor=True)
    image_features = image_input.flatten()

    return image_features

if uploaded_image is not None:
    if st.button("Find Similar Images"):
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Encode the image
        image_features = compute_clip_features(uploaded_image)

        # Convert the NumPy array to a list
        image_features_list = image_features.tolist()
        st.write(image_features_list)

        # Send the image vector as JSON in the request body
        # response = requests.post(f"{url}/image_search", json={"image": image_features_list})
        # features = response.json()
        pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
        index = pinecone.Index(index_name)
        image_vector = image_features_list
    
        closest_image_ids = index.query(
            vector=image_vector,
            top_k=4, 
            include_values=False
        )

        closest_image_ids = [i['id'] for i in closest_image_ids['matches']]

        features = closest_image_ids

        s3_client = boto3.client(service_name = 's3',
                aws_access_key_id='AKIASGVOLPG2BPBLER6T',
                aws_secret_access_key='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',
                region_name = 'us-east-2')
        

        for i in features:
            img = s3_client.get_object(Bucket='assignment4admt5', Key=f'dataset/{i}')
            


            image_bytes = img['Body'].read()

            # Display the image in Streamlit
            st.image(image_bytes)

        # for i in features:
        #     image_path = path +'/' +i 
        #     image = Image.open(image_path)
        #     st.image(image, use_column_width=True)

        st.write(features)
