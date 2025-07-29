from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_submission(
    request: QueryRequest, 
    auth_verified: bool = Depends(verify_token) 
):

    print(f"Document URL: {request.documents}")
    print(f"Questions: {request.questions}")

    mock_answers = [f"Placeholder answer for: {q}" for q in request.questions]
    return QueryResponse(answers=mock_answers)

