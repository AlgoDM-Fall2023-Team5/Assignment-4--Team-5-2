import streamlit as st
from PIL import Image
import pinecone
from io import BytesIO
from sentence_transformers import SentenceTransformer
import json
import boto3
from api import image_search
import torch
import torchvision.transforms as transforms

url = "http://127.0.0.1:8000"
path = "C:/Users/shiri/Documents/Assg4ADM/dataset"
pinecone_api_key = "7f78befa-055d-41ac-a90a-cff6a5282d66"
index_name = "adm4"
st.set_option('browser.gatherUsageStats', False)
# Load the SentenceTransformer model
image_encoder = SentenceTransformer('clip-ViT-B-32')

# Quantize the model
quantized_image_encoder = torch.quantization.quantize_dynamic(
    image_encoder, {torch.nn.Linear}, dtype=torch.qint8
)

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

        # Encode the image using the quantized model
        image_features = encode_image(uploaded_image, quantized_image_encoder)

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

        s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id='AKIASGVOLPG2BPBLER6T',
            aws_secret_access_key='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',
            region_name='us-east-2'
        )
        
        for i in features:
            img = s3_client.get_object(Bucket='assignment4admt5', Key=f'dataset/{i}')
            
            image_bytes = img['Body'].read()

            # Display the image in Streamlit
            st.image(image_bytes)

        st.write(features)
