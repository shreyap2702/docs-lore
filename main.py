from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader
import mimetypes
import asyncio
import requests
import logging
import io
import os
import docx
import email
from email import policy
from dotenv import load_dotenv
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

logger = logging.getLogger(__name__)
app = FastAPI(title="LLM Query-Retrieval System")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

EXPECTED_AUTH_TOKEN = "10fbd8807c6d9b5a37028c3eb4bd885959f06d006aedd2dc5ba64c5ccea913c0"
security = HTTPBearer()


class QueryRequest(BaseModel):
    documents: str
    questions: list[str]


class QueryResponse(BaseModel):
    answers: list[str]


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token"
        )
    return True


async def download_file_content(url: str) -> tuple[bytes, str]:
    logger.info(f"Downloading from {url}")
    try:
        response = await asyncio.to_thread(requests.get, url, stream=True, timeout=30)
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("Content-Type", "")
        return content, content_type
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Download timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request error: {e}")


def detect_file_type(url: str, content_type: str) -> str:
    ext_type, _ = mimetypes.guess_type(url)
    mime = content_type.lower()
    if "pdf" in mime or ext_type == "application/pdf":
        return "pdf"
    elif "word" in mime or ext_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    elif "rfc822" in mime or ext_type == "message/rfc822":
        return "eml"
    return "unknown"


async def parse_pdf(data: bytes) -> str:
    try:
        reader = await asyncio.to_thread(PdfReader, io.BytesIO(data))
        return "\n".join([await asyncio.to_thread(page.extract_text) or "" for page in reader.pages])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")


async def parse_docx(data: bytes) -> str:
    try:
        doc = await asyncio.to_thread(docx.Document, io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX parsing error: {e}")


async def parse_email(data: bytes) -> str:
    try:
        msg = await asyncio.to_thread(email.message_from_bytes, data, policy=policy.default)
        main_body = msg.get_body()
        if main_body:
            for part in main_body.iter_attachments():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
            return main_body.get_content()
        return ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email parsing error: {e}")


def chunk_text(text: str, source_name: str) -> list[Document]:
    chunks = text_splitter.create_documents([text])
    for chunk in chunks:
        chunk.metadata["source"] = source_name
    return chunks


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def store_chunks_in_pinecone(chunks: list[Document]):
    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=os.environ["PINECONE_INDEX_NAME"],
        namespace="hackrx-2025"
    )
    return vectorstore


def get_retrieval_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the information from the documents:\n\n{context}\n\nQuestion: {input}"
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)

    return (
        {"context": retriever | RunnablePassthrough(), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, auth: bool = Depends(verify_token)):
    file_bytes, content_type = await download_file_content(request.documents)
    file_type = detect_file_type(request.documents, content_type)

    if file_type == "pdf":
        text = await parse_pdf(file_bytes)
    elif file_type == "docx":
        text = await parse_docx(file_bytes)
    elif file_type == "eml":
        text = await parse_email(file_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported document type: {file_type}")

    chunks = chunk_text(text, request.documents)
    vectorstore = store_chunks_in_pinecone(chunks)
    rag_chain = get_retrieval_chain(vectorstore)

    answers = [rag_chain.invoke(q) for q in request.questions]
    return QueryResponse(answers=answers)
