from fastapi import FastAPI
from graph import app
from schema import AnswerResponse

api = FastAPI()

@api.post("/ask", response_model=AnswerResponse)
def ask(request: dict):
    result = app.invoke({
        "query": request["query"],
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0
    })

    return AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )