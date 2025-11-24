from pydantic import BaseModel

class TextOutput(BaseModel):
    label: str
    score: float
    stars: int