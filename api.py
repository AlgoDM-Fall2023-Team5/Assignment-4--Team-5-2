from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pinecone
import numpy as np
import pinecone
import boto3
from botocore.exceptions import NoCredentialsError
import os
from PIL import Image

app = FastAPI()

pinecone_api_key = "7f78befa-055d-41ac-a90a-cff6a5282d66"
index_name = "adm4"
s3_bucket_name = "assignment4admt5"

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
index = pinecone.Index(index_name)

s3 = boto3.client('s3', aws_access_key_id='AKIASGVOLPG2BT6D6XXO', aws_secret_access_key='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')


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

    images_from_s3 = []

    for image_id in closest_image_ids:
        try:
            image_object = s3.get_object(Bucket=s3_bucket_name, Key=f"images/{image_id}")
            images_from_s3.append(image_object['Body'].read())
        except NoCredentialsError:
            return HTTPException(status_code=500, detail="AWS credentials not available.")
        except Exception as e:
            return HTTPException(status_code=500, detail=f"Error retrieving image {image_id}: {str(e)}")

    

    return closest_image_ids
def image_search(request: ImageSearchRequest):
    image_vector = request.image
    
    closest_image_ids = index.query(
        vector=image_vector,
        top_k=4, 
        include_values=False
    )

    closest_image_ids = [i['id'] for i in closest_image_ids['matches']]
    return closest_image_ids

