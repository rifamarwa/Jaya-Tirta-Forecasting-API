from numpy import number
from pydantic import BaseModel

class HasilRamal(BaseModel):
    id_peramalan: int
    data_ramal: int