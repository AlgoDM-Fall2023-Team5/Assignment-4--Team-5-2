from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pinecone
import numpy as np

app = FastAPI()

pinecone_api_key = "878f92d2-ce36-4be5-bbc0-05a56ff092fc"
index_name = "adm4"

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
index = pinecone.Index(index_name)

class ImageSearchRequest(BaseModel):
    image: List[float]

@app.post("/image_search")
async def image_search(request: ImageSearchRequest):
    image_vector = request.image
    
    closest_image_ids = index.query(
        vector=image_vector,
        top_k=4, 
        include_values=False
    )

    closest_image_ids = [i['id'] for i in closest_image_ids['matches']]

    return closest_image_ids
