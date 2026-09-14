FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn chromadb sentence-transformers langgraph pydantic

EXPOSE 7860

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "7860"]