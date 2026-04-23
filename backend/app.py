from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

app = FastAPI(title="RAG Support Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG once
loader = PyPDFLoader("data/knowledge_base.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

llm = OllamaLLM(model="llama3.2:1b")

class Query(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: Query) -> Dict[str, Any]:
    docs = retriever.invoke(query.query)

    if not docs:
        return {"decision": "ESCALATE", "answer": "No relevant information found."}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context.
If answer is not in context, say: NOT_FOUND

Context:
{context}

Question:
{query.query}
"""

    response = llm.invoke(prompt)

    response_text = response.strip().lower()

    decision = "ESCALATE"
    if not any(phrase in response_text for phrase in ["not_found", "not available", "does not contain", "no information"]) and len(response_text) >= 40:
        decision = "ANSWER"

    return {"decision": decision, "answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

