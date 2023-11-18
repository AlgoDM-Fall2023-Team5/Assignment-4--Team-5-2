from fastapi import FastAPI
from sample import function
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests
class data(BaseModel):
    text : str 

#encoder
image_encoder = SentenceTransformer('clip-ViT-B-32')

app = FastAPI()


@app.post("/Image_Predict")
async def predict(text : data):
    x = function(text.text)
    return x



@app.post("/image_search")
async def image_search(image: UploadFile = File(...)):
    # Load the uploaded image
    image_bytes = await image.read()
    image_pil = Image.open(BytesIO(image_bytes))

    # Encode the image using the provided image encoder
    image_features = encode_image(image_pil)

    # Forward the encoded image to Pinecone for similarity search
    pinecone_api_key = "YOUR_PINECONE_API_KEY"
    pinecone_index_name = "adm4"
    pinecone_url = f"https://api.pinecone.io/v1/index/{pinecone_index_name}/query"

    headers = {
        "Authorization": f"Apikey {pinecone_api_key}",
    }

    data = {
        "top_k": 3,
        "query_vector": image_features.tolist(),
    }

    response = requests.post(pinecone_url, headers=headers, json=data)

    if response.status_code == 200:
        return JSONResponse(content=response.json())
    else:
        return JSONResponse(content={"error": "Error performing similarity search."}, status_code=response.status_code)

def encode_image(image):
    # Implement the image encoding logic using the provided image encoder
    # Replace this with your actual encoding logic
    image_input = image_encoder.encode(image, convert_to_tensor=True)
    image_features = image_input.flatten()
    return image_features
