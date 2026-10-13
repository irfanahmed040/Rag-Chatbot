import io
import os
import re

import docx2txt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

app = FastAPI()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
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


def tokenize(text: str) -> list[str]:
    """Lowercase words only, so 'Price,' and 'price' match in keyword search."""
    return re.findall(r"\w+", text.lower())


# ---------- 1. Upload: file -> text chunks ----------
# Vercel functions don't remember anything between requests,
# so we send the chunks back to the browser and it keeps them.
@app.post("/api/upload")
async def upload(files: list[UploadFile]):
    chunks = []
    for f in files:
        text = read_text(f.filename, await f.read())
        chunks.extend(splitter.split_text(text))

    if not chunks:
        raise HTTPException(400, "No text found. Scanned/image PDFs are not supported.")
    return {"chunks": chunks}


# ---------- 2. Chat: question + chunks -> answer ----------
class ChatRequest(BaseModel):
    question: str
    chunks: list[str]


@app.post("/api/chat")
def chat(req: ChatRequest):
    # Find the 4 chunks that best match the question's keywords (BM25)
    retriever = BM25Retriever.from_texts(req.chunks, preprocess_func=tokenize, k=4)
    context = "\n\n".join(doc.page_content for doc in retriever.invoke(req.question))

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know. Reply in plain text.

Context:
{context}

Question: {req.question}"""

    return {"answer": llm.invoke(prompt).content}


# Local development only: serve the frontend too. On Vercel, public/ is served automatically.
if os.path.isdir("public"):
    app.mount("/", StaticFiles(directory="public", html=True))
