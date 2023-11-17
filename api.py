from fastapi import FastAPI
from sample import function
from pydantic import BaseModel

class data(BaseModel):
    text : str 

app = FastAPI()


@app.get("/")
async def root():
    return {"message":"Hello World"}

@app.post("/Image_Predict")
async def predict(text : data):
    x = function(text.text)
    return x
