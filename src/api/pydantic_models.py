from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from enum import Enum

import bleach
from pydantic import BaseModel, Field, field_validator


class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    DEEPSEEK_R1 = "deepseek-r1"
    LLAMA_3_2 = "llama3.2:3b"
    LLAMA_3 = "llama3"


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

    @field_validator("question")
    def sanitize_question(cls, v):
        # Strip all HTML tags
        return bleach.clean(v, strip=True)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int
