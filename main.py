from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import asyncio
import requests
import logging
logger = logging.getLogger(__name__)
from pypdf import PdfReader
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import docx
import email
from email import policy
from email import iterators

class QueryRequest(BaseModel):
    documents : str
    questions : list[str]
    
class QueryResponse(BaseModel):
    answers : list[str]

app = FastAPI(title="LLM Query-Retrieval System")

EXPECTED_AUTH_TOKEN = "10fbd8807c6d9b5a37028c3eb4bd885959f06d006aedd2dc5ba64c5ccea913c0" # From problem statement
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token"
        )
    return True

async def download_file_content(url: str) -> bytes:
    logger.info(f"Attempting to download document from: {url}")
    try:
        response = await asyncio.to_thread(requests.get, url, stream=True, timeout=30)
        response.raise_for_status()
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
        logger.info(f"Successfully downloaded {len(content)} bytes.")
        return content
    except requests.exceptions.Timeout:
        logger.error(f"Download timed out for {url}")
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Download timed out for document: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading document from {url}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to download document: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during document download from {url}.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during document download: {e}")

async def parse_pdf(pdf_bytes: bytes) -> str:
    logger.info("Attempting to parse PDF bytes into text.")
    text_content = ""
    try:
        reader = await asyncio.to_thread(PdfReader, io.BytesIO(pdf_bytes))
        for page_num, page in enumerate(reader.pages):
            page_text = await asyncio.to_thread(page.extract_text) or ""
            text_content += page_text
        logger.info(f"Successfully parsed PDF. Extracted {len(text_content)} characters.")
        return text_content
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse PDF: {e}"
        )

async def parse_docx(docx_bytes: bytes) -> str:
    logger.info("Attempting to parse DOCX bytes into text.")
    text_content = ""
    try:
        doc = await asyncio.to_thread(docx.Document, io.BytesIO(docx_bytes))
        for paragraph in doc.paragraphs:
            text_content += paragraph.text + "\n"
        logger.info(f"Successfully parsed DOCX. Extracted {len(text_content)} characters.")
        return text_content
    except Exception as e:
        logger.error(f"Error parsing DOCX: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse DOCX: {e}"
        )

async def parse_email(email_bytes: bytes) -> str:
    logger.info("Attempting to parse email bytes into text.")
    text_content = ""
    try:
        msg = await asyncio.to_thread(email.message_from_bytes, email_bytes, policy=policy.default)
        main_body = msg.get_body()

        if main_body:
            for part in main_body.iter_attachments():
                if part.get_content_type() == 'text/plain':
                    text_content += part.get_content()
                    break
            else:
                if main_body.get_content_type() == 'text/plain':
                    text_content = main_body.get_content()
                elif main_body.get_content_type() == 'text/html':
                    text_content = main_body.get_content()
                    logger.warning("Extracted HTML content from email. Consider using HTML parser for cleaner text.")
        
        logger.info(f"Successfully parsed email. Extracted {len(text_content)} characters.")
        return text_content
    except Exception as e:
        logger.error(f"Error parsing email: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse email: {e}"
        )
        

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(
    request: QueryRequest, 
    auth_verified: bool = Depends(verify_token) 
):

    print(f"Document URL: {request.documents}")
    print(f"Questions: {request.questions}")

    mock_answers = [f"Placeholder answer for: {q}" for q in request.questions]
    return QueryResponse(answers=mock_answers)

