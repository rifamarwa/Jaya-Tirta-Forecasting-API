from fastapi import FastAPI
from model import HasilRamal
from typing import List
import json

app = FastAPI()

data = open('sample.json')
data_ramal = json.load(data)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/v1/hasilramal")
async def fetch_data():
    return data_ramal;


data.close()


# db: List[HasilRamal] = [
#     HasilRamal(
#         id_peramalan=1,
#         data_ramal=2341
#     ),
#     HasilRamal(
#         id_peramalan=2,
#         data_ramal=2211
#     )
# ]



