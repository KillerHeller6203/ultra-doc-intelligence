import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingestion.parser import DocumentParser
from backend.ingestion.processor import DocumentProcessor
from backend.ingestion.vector_store import VectorStore
from backend.rag.engine import RAGEngine
from backend.rag.extractor import DataExtractor

# Load environment variables
load_dotenv()

app = FastAPI(title="Ultra Doc-Intelligence API")

# Initialize components
parser = DocumentParser()
processor = DocumentProcessor()
vector_store = VectorStore()
rag_engine = RAGEngine(vector_store)
extractor = DataExtractor(vector_store)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload", summary="Upload and process a logistics document", description="Accepts PDF, DOCX, or TXT files, chunks them semantically, and stores them in the vector index.")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = parser.parse(file.filename, content)
        chunks = processor.process(text)
        
        # For this POC, we clear the index on each upload to focus on the current doc
        vector_store.clear()
        vector_store.add_documents(chunks)
        
        return {"message": f"File {file.filename} processed and indexed successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", summary="Ask a question about the uploaded document", description="Retrieves relevant chunks and generates a grounded answer. Enforces similarity and confidence guardrails.")
async def ask_question(request: QuestionRequest):
    try:
        if not vector_store.chunks:
            raise HTTPException(status_code=400, detail="No document uploaded yet.")
        
        result = rag_engine.ask(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract", summary="Extract structured logistics data", description="Extracts key fields like shipment_id, shipper, consignee, etc., into a validated JSON format.")
async def extract_data():
    try:
        if not vector_store.chunks:
            raise HTTPException(status_code=400, detail="No document uploaded yet.")
        
        extracted_data = extractor.extract_fields()
        return extracted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
