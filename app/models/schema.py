from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum

class StatusEnum(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class BaseResponse(BaseModel):
    status: StatusEnum
    message: str

class FaceBase(BaseModel):
    name: str
    face_id: str
    created_at: datetime

class FaceCreate(BaseModel):
    name: str

class FaceResponse(BaseResponse):
    data: Optional[FaceBase] = None

class CheckInResponse(BaseResponse):
    data: Optional[dict] = Field(
        None,
        example={
            "matched": True,
            "face_id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "John Doe",
            "similarity": 0.98
        }
    )

class ErrorResponse(BaseResponse):
    detail: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Face already exists in database",
                "detail": "Similarity score: 0.95"
            }
        }

class SearchResult(BaseModel):
    face_id: str
    name: str
    similarity: float
    created_at: datetime

class SearchResponse(BaseResponse):
    results: List[SearchResult]
    total: int