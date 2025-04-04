from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI()

class Text(BaseModel):
    content: str

@app.post("/analyze")
async def analyze_sentiment(text: Text):
    blob = TextBlob(text.content)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    label = "POSITIVE" if polarity > 0 else "NEGATIVE"
    return {"label": label, "score": abs(polarity)}

@app.get("/")
async def root():
    return {"message": "Отправьте POST запрос на эндпойнт /analyze чтобы определить настрой текста."}
