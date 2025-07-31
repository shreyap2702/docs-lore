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
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_retrieval_chain
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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackathon-policy-docs")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Simple In-Memory Cache for Processed Documents ---
processed_documents_cache = set()

# --- Global Initializations ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.2, google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768, # Dimension for "models/embedding-001" is typically 768
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )
    logger.info(f"Created new Pinecone index: {PINECONE_INDEX_NAME}")
else:
    logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")

vectorstore = PineconeVectorStore(index=pc.Index(PINECONE_INDEX_NAME), embedding=embeddings, namespace="hackrx-2025")

# Fixed prompt template to use 'input' instead of 'question'
rag_prompt_template = ChatPromptTemplate.from_template("""
    You are a comprehensive and detailed question-answering system for any type of document content.
    Your goal is to provide thorough, accurate answers that include all relevant details from the provided context.
    Answer the user's question based on the following context, providing complete information including:
    - Specific details, numbers, dates, and time periods
    - Conditions, requirements, and criteria
    - Limitations, restrictions, and caps
    - Procedures, processes, and steps
    - Definitions, classifications, and categories
    - Any other relevant factual information present in the context
    If the context is empty or does not contain relevant information to answer the question, clearly state "The provided document does not contain information to answer this question."
    If the context contains relevant information, provide a detailed answer that captures all important details from the source material.
    Do not make up answers or include information not present in the context.
    Do NOT include any introductory phrases, citations, or references to specific sections/clauses from the context in your answer.
    For yes/no questions, provide a detailed explanation with all relevant conditions and requirements.
 

    Context:
    {context}

    Question: {input} 

    Answer:
    """)

document_chain = create_stuff_documents_chain(llm, rag_prompt_template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8, "score_threshold": 0.4})
rag_chain = create_retrieval_chain(retriever, document_chain)

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


def validate_text_input(text_input):
    """Validate and clean text input for embeddings"""
    if text_input is None:
        return ""
    if not isinstance(text_input, str):
        text_input = str(text_input)
    return text_input.strip()


async def download_file_content(url: str) -> tuple[bytes, str]:
    logger.info(f"Downloading from {url}")
    try:
        response = await asyncio.to_thread(requests.get, url, stream=True, timeout=30)
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get("Content-Type", "")

        # --- CORRECTION ADDED HERE ---
        if not isinstance(content, bytes) or len(content) == 0:
            logger.error(f"Downloaded content for {url} is not bytes or is empty. Type: {type(content)}, Length: {len(content) if content else 0}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, # More specific error code
                detail=f"Downloaded file from {url} is empty or malformed."
            )
        # --- END CORRECTION ---

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
        text_parts = []
        for page in reader.pages:
            page_text = await asyncio.to_thread(page.extract_text)
            if page_text:
                text_parts.append(page_text.strip())
        
        text = "\n".join(text_parts)
        # Validate extracted text
        text = validate_text_input(text)
        logger.info(f"Successfully parsed PDF. Extracted {len(text)} characters.")
        return text
    except Exception as e:
        logger.error(f"PDF parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"PDF parsing error: {e}")


async def parse_docx(data: bytes) -> str:
    logger.info("Attempting to parse DOCX bytes into text.")
    try:
        doc = await asyncio.to_thread(docx.Document, io.BytesIO(data))
        text_parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(text_parts)
        # Validate extracted text
        text = validate_text_input(text)
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
        text = ""
        if main_body:
            for part in main_body.iter_attachments():
                if part.get_content_type() == 'text/plain':
                    text = part.get_content()
                    break
            if not text and main_body.get_content_type() == 'text/plain':
                text = main_body.get_content()
            elif not text and main_body.get_content_type() == 'text/html':
                text = main_body.get_content()
                logger.warning("Extracted HTML content from email. Consider using HTML parser for cleaner text.")
        
        # Validate extracted text
        text = validate_text_input(text)
        logger.info(f"Successfully parsed email. Extracted {len(text)} characters.")
        return text
    except Exception as e:
        logger.error(f"Email parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Email parsing error: {e}")


def chunk_text(text: str, source_name: str) -> list[Document]:
    # Validate input text
    text = validate_text_input(text)
    if not text:
        logger.warning(f"Empty text provided for chunking from source: {source_name}")
        return []
    
    chunks = text_splitter.create_documents([text])
    valid_chunks = []
    
    for chunk in chunks:
        # Ensure chunk content is valid
        if hasattr(chunk, 'page_content'):
            chunk.page_content = validate_text_input(chunk.page_content)
            if chunk.page_content:  # Only add non-empty chunks
                chunk.metadata["source"] = source_name
                valid_chunks.append(chunk)
        else:
            logger.warning(f"Invalid chunk format: {type(chunk)}")
    
    logger.info(f"Created {len(valid_chunks)} valid chunks from {len(chunks)} total chunks")
    return valid_chunks


@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(request: QueryRequest, auth: bool = Depends(verify_token)):
    logger.info(f"Received request for document: {request.documents}")
    logger.info(f"Questions: {request.questions}")

    try:
        if request.documents not in processed_documents_cache:
            logger.info(f"Document {request.documents} not in cache. Proceeding with ingestion.")

            file_bytes, content_type = await download_file_content(request.documents)
            
            file_type = detect_file_type(request.documents, content_type)
            logger.info(f"DEBUG: Before parsing - file_bytes type: {type(file_bytes)}")
            logger.info(f"DEBUG: Before parsing - file_bytes length: {len(file_bytes)} bytes")
            logger.info(f"DEBUG: Before parsing - Detected file_type: {file_type}")
            
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

            if not text.strip():
                logger.error(f"No text extracted from document: {request.documents}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No text content could be extracted from the document"
                )

            chunks_for_pinecone = chunk_text(text, request.documents)
            
            if not chunks_for_pinecone:
                logger.error(f"No valid chunks created from document: {request.documents}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid text chunks could be created from the document"
                )

            await asyncio.to_thread(vectorstore.add_documents, chunks_for_pinecone)
            logger.info(f"Upserted {len(chunks_for_pinecone)} chunks to Pinecone for {request.documents}.")
            
            processed_documents_cache.add(request.documents)
        else:
            logger.info(f"Document {request.documents} found in cache. Skipping ingestion steps.")

        answers = []
        for question in request.questions:
            logger.info(f"Answering question: {question}")

    # Validate question input
            clean_question = validate_text_input(question)
            if not clean_question:
                logger.warning(f"Empty or invalid question: {question}")
                answers.append("Invalid or empty question provided.")
                continue

            try:
                # Debug: Check what documents are being retrieved
                docs = await asyncio.to_thread(retriever.get_relevant_documents, clean_question)
                logger.info(f"Retrieved {len(docs)} documents for question: {clean_question}")
                if docs:
                    logger.info(f"First doc preview: {docs[0].page_content[:200]}...")
                
                result = await rag_chain.ainvoke({"input": clean_question})  # Use 'input' key
                answer = result.get("answer", "No answer found")  # Extract the 'answer' key from the result dict
                answers.append(answer)
                logger.info(f"Generated answer for '{question}': {answer[:100]}...")
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                answers.append(f"Error processing question: {str(e)}")


        return QueryResponse(answers=answers)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("An unhandled error occurred during run_submission processing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {str(e)}"
        )