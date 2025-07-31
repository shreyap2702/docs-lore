from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
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

# --- 1. Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackathon-policy-docs") # Get from .env, with default

# --- 2. Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 3. Global Initializations (These run once at app startup) ---
# Simple In-Memory Cache for Processed Documents
processed_documents_cache = set()

# Text Splitter (Tunable parameters: chunk_size, chunk_overlap)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# LLM and Embeddings with API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768, # Dimension for "models/embedding-001" is typically 768
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-west-2') # Adjust as per your Pinecone plan
    )
    logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
else:
    logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

# Pinecone Vector Store (using the global Pinecone client and embeddings)
vectorstore = PineconeVectorStore(index=pc.Index(PINECONE_INDEX_NAME), embedding=embeddings)

# RAG Prompt Template (Tunable - experiment with phrasing)
rag_prompt_template = ChatPromptTemplate.from_template("""
    You are an intelligent query-retrieval system. Answer the user's question ONLY based on the following context.
    If the answer is not found in the context, clearly state "The provided document does not contain information to answer this question." Do not make up answers.

    For each answer, explicitly state which part(s) of the provided context helped you formulate the answer (e.g., "Based on Clause X...", or "As per the section on Y...").

    Context:
    {context}

    Question: {question}

    Answer:
    """)

# Document combining chain
document_chain = create_stuff_documents_chain(llm, rag_prompt_template)

# Full RAG chain (Retrieval 'k' value adjusted here for better context)
rag_chain = (
    {"context": vectorstore.as_retriever(search_kwargs={"k": 5}), "input": RunnablePassthrough()} # K changed from 3 to 5
    | document_chain
    | StrOutputParser()
)
# --- End Global Initializations ---

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
        logger.warning("Authentication failed: Invalid token.")
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
        logger.error(f"Download timed out for {url}")
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail="Download timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {url}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Request error: {e}")


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
    logger.info("Attempting to parse PDF bytes into text.")
    try:
        reader = await asyncio.to_thread(PdfReader, io.BytesIO(data))
        text = "\n".join([await asyncio.to_thread(page.extract_text) or "" for page in reader.pages])
        logger.info(f"Successfully parsed PDF. Extracted {len(text)} characters.")
        return text
    except Exception as e:
        logger.error(f"PDF parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"PDF parsing error: {e}")


async def parse_docx(data: bytes) -> str:
    logger.info("Attempting to parse DOCX bytes into text.")
    try:
        doc = await asyncio.to_thread(docx.Document, io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        logger.info(f"Successfully parsed DOCX. Extracted {len(text)} characters.")
        return text
    except Exception as e:
        logger.error(f"DOCX parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DOCX parsing error: {e}")


async def parse_email(data: bytes) -> str:
    logger.info("Attempting to parse email bytes into text.")
    try:
        msg = await asyncio.to_thread(email.message_from_bytes, data, policy=policy.default)
        main_body = msg.get_body()
        if main_body:
            for part in main_body.iter_attachments():
                if part.get_content_type() == 'text/plain':
                    return part.get_content()
            if main_body.get_content_type() == 'text/plain':
                text = main_body.get_content()
            elif main_body.get_content_type() == 'text/html':
                text = main_body.get_content()
                logger.warning("Extracted HTML content from email. Consider using HTML parser for cleaner text.")
            else:
                text = ""
            logger.info(f"Successfully parsed email. Extracted {len(text)} characters.")
            return text
        return ""
    except Exception as e:
        logger.error(f"Email parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Email parsing error: {e}")


def chunk_text(text: str, source_name: str) -> list[Document]:
    # text_splitter is already defined globally
    chunks = text_splitter.create_documents([text])
    for chunk in chunks:
        chunk.metadata["source"] = source_name
    return chunks


# --- Removed get_embeddings, store_chunks_in_pinecone, get_retrieval_chain functions ---
# Their logic is now handled by global initializations and direct calls in run_submission

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, auth: bool = Depends(verify_token)):
    logger.info(f"Received request for document: {request.documents}")
    logger.info(f"Questions: {request.questions}")

    try:
        # --- NEW: Check Cache Before Processing Document ---
        if request.documents not in processed_documents_cache:
            logger.info(f"Document {request.documents} not in cache. Proceeding with ingestion.")

            # 1. Download Document Content
            file_bytes, content_type = await download_file_content(request.documents)
            
            # 2. Detect File Type
            file_type = detect_file_type(request.documents, content_type)
            
            # 3. Parse Document Content
            text = ""
            if file_type == "pdf":
                text = await parse_pdf(file_bytes)
            elif file_type == "docx":
                text = await parse_docx(file_bytes)
            elif file_type == "eml":
                text = await parse_email(file_bytes)
            else:
                logger.warning(f"Unsupported document type detected: {file_type} for URL {request.documents}")
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported document type: {file_type}")

            # 4. Chunk Text
            chunks_for_pinecone = chunk_text(text, request.documents)

            # 5. Store Chunks in Pinecone
            await asyncio.to_thread(vectorstore.add_documents, chunks_for_pinecone)
            logger.info(f"Upserted {len(chunks_for_pinecone)} chunks to Pinecone for {request.documents}.")
            
            # Add document URL to cache so it's skipped next time
            processed_documents_cache.add(request.documents)
        else:
            logger.info(f"Document {request.documents} found in cache. Skipping ingestion steps.")


        # 6. Retrieve and Generate Answers using the global RAG chain
        answers = []
        for question in request.questions:
            logger.info(f"Answering question: {question}")
            # Use await .ainvoke() for asynchronous chain execution
            result = await rag_chain.ainvoke({"question": question}) 
            answers.append(result)
            logger.info(f"Generated answer for '{question}': {result[:100]}...")

        return QueryResponse(answers=answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("An unhandled error occurred during run_submission processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {str(e)}"
        )