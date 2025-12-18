from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from models.schemas import (
    QuestionRequest, 
    AnswerResponse, 
    SummarizeRequest, 
    SummaryResponse,
    PDFExtractResponse
)

from services.document_service import document_service
from services.gemini_service import gemini_service


router = APIRouter(
    prefix="/api/courses",
    tags=["courses"]  
)


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "formats": document_service.SUPPORTED_EXTENSIONS,
        "extensions": document_service.get_supported_extensions()
    }


@router.post("/upload", response_model=PDFExtractResponse)
async def upload_course(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    """
    Upload a document and extract its text content.
    Supports: PDF, TXT, MD, DOCX, PNG, JPG, JPEG, GIF, WEBP
    """
    # Validate file type
    file_type = document_service.get_file_type(file.filename)
    
    if file_type == "unsupported":
        supported = document_service.get_supported_extensions()
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: {', '.join(supported)}"
        )
    
    # Images require API key for Gemini Vision
    if file_type == "image" and not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key is required for image text extraction"
        )
    
    try:
        content = await file.read()
        
        # 10MB limit
        if len(content) > 10 * 1024 * 1024:
             raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 10MB."
            )
            
        # Use the new extract_text method
        text, page_count, chunks = await document_service.extract_text(
            content, 
            file.filename,
            api_key
        )
        
        return PDFExtractResponse(
            text=text,
            page_count=page_count,
            success=True
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        return PDFExtractResponse(
            text="",
            page_count=0,
            success=False,
            error=str(e)
        )


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_course(request: SummarizeRequest):
    """Generate AI summary and key points from course content."""
    try:
        result = await gemini_service.generate_summary(
            request.course_content,
            request.api_key,
            request.language
        )
        
        return SummaryResponse(
            summary=result["summary"],
            key_points=result["key_points"],
            flashcards=result["flashcards"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate summary: {str(e)}"
        )


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the course content."""
    try:
        answer = await gemini_service.answer_question(
            question=request.question,
            course_content=request.course_content,
            api_key=request.api_key
        )
        
        return AnswerResponse(answer=answer)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(e)}"
        )