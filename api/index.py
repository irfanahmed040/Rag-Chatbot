import io
import os

import docx2txt
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

app = FastAPI()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
embeddings = MistralAIEmbeddings(model="mistral-embed")  # text -> list of 1024 numbers
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# ---------- Helpers ----------
def read_text(filename: str, data: bytes) -> str:
    """Turn an uploaded file into plain text."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        return docx2txt.process(io.BytesIO(data))
    return data.decode("utf-8", errors="ignore")  # .txt, .md, .csv


def top_chunks(question: str, chunks: list[str], chunk_vectors: list[list[float]], k: int = 4):
    """Return the k chunks whose meaning is closest to the question (cosine similarity)."""
    q = np.array(embeddings.embed_query(question))
    m = np.array(chunk_vectors)
    scores = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:k]  # indexes of the highest scores
    return [chunks[i] for i in best]


#chunkings
@app.post("/api/upload")
async def upload(files: list[UploadFile]):
    chunks = []
    for f in files:
        text = read_text(f.filename, await f.read())
        chunks.extend(splitter.split_text(text))

    if not chunks:
        raise HTTPException(400, "No text found. Scanned/image PDFs are not supported.")

    # Embed every chunk. Rounded to 4 decimals to keep the response small.
    vectors = [[round(x, 4) for x in v] for v in embeddings.embed_documents(chunks)]
    return {"chunks": chunks, "embeddings": vectors}


# ---------- 2. Chat: question + chunks -> answer ----------
class ChatRequest(BaseModel):
    question: str
    chunks: list[str]
    embeddings: list[list[float]]


@app.post("/api/chat")
def chat(req: ChatRequest):
    # Find the 4 chunks closest in meaning to the question
    context = "\n\n".join(top_chunks(req.question, req.chunks, req.embeddings))

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know. Reply in plain text.

Context:
{context}

Question: {req.question}"""

    return {"answer": llm.invoke(prompt).content}


# Local development only: serve the frontend too. On Vercel, public/ is served automatically.
if os.path.isdir("public"):
    app.mount("/", StaticFiles(directory="public", html=True))
