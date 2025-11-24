from fastapi import APIRouter, HTTPException
from app.models.bert_base_uncased import model, tokenizer, device
import torch
from app.schemas.predict import TextOutput
import torch.nn.functional as F

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("/")
async def sentiment(text: str):

    def is_text_too_long(_text: str, max_token : int = 512) -> bool:
        return len(tokenizer.encode(_text, add_special_tokens=True)) > max_token

    if is_text_too_long(text):
        raise HTTPException(
            status_code=400,
            detail="Text too long",
        )
    input_token = tokenizer(
        text.lower(),
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**input_token)

    scores = F.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(scores, dim=-1).item()
    confidence = scores[0][predicted_class].item()

    stars = predicted_class +1
    label = "positive" if stars >= 4 else "negative" if stars <= 2 else "neutral"

    return TextOutput(
        label= label,
        score= round(confidence, 4),
        stars= stars
        )

