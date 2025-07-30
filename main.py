from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import requests
import logging
logger = logging.getLogger(__name__)
from pypdf import PdfReader
import io

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

async def download_file(url: str) -> bytes:
    """
    Downloads a file from a given URL and returns its content as bytes.
    """
    logger.info(f"Downloading from: {url}")

    try:
        response = await asyncio.to_thread(
            requests.get,
            url,
            timeout=30
        )
        response.raise_for_status()
        content = response.content
        logger.info(f"Downloaded {len(content)} bytes successfully")

        return content

    except requests.exceptions.Timeout:
        logger.error(f"Download timed out for {url}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT, 
            detail="Download timed out"
        )

    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Download Failed"
        )
        
async def pdf_parser(pdf_bytes: bytes)->str:
    logger.info("Attempting to parse PDF bytes into text.")
    text_content = ""
    
    try:
        reader = await asyncio.to_thread(PdfReader, io.BytesIO(pdf_bytes))
        
        for page_no, page in enumerate(reader.pages):
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
        

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(
    request: QueryRequest, 
    auth_verified: bool = Depends(verify_token) 
):

    print(f"Document URL: {request.documents}")
    print(f"Questions: {request.questions}")

    mock_answers = [f"Placeholder answer for: {q}" for q in request.questions]
    return QueryResponse(answers=mock_answers)

