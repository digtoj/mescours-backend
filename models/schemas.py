from pydantic import BaseModel
from typing import List, Optional

#Request: Ask question
class QuestionRequest(BaseModel):
    course_content:str
    question: str
    api_Key: str

#Response: Answer question
class AnswerResponse(BaseModel):
    answer: str
    referenced_content: Optional[str] = None

#Request: Generate summary
class SummarizeRequest(BaseModel):
    course_content: str
    api_key: str

#Response: Summary 
class SummaryResponse(BaseModel):
    summary: str
    key_points: List[str]
    audio_script: str

class PDFExtractResponse(BaseModel):
    text: str
    page_count: int
    success: bool
    error: Optional[str] = None