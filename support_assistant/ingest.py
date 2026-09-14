from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

base_path = Path(__file__).parent
docs_path = base_path / "docs"

documents = []
ids = []

for file in sorted(docs_path.glob("doc_*.txt")):
    text = file.read_text(encoding="utf-8")
    print(file.name, "characters:", len(text))
    documents.append(text)
    ids.append(file.stem)

print("Documents found:", len(documents))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents).tolist()

client = chromadb.PersistentClient(path=str(base_path / "chroma_db"))

try:
    client.delete_collection("zepto_policies")
except:
    pass

collection = client.create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents stored:", collection.count())
print("Collection metadata:", collection.metadata)