from pydantic import BaseModel, EmailStr
from enum import Enum
from typing import Optional, List, Dict, Union

# Enums for Role & License
class RoleEnum(str, Enum):
    Admin = "Admin"
    Editor = "Editor"
    Viewer = "Viewer"

class LicenseEnum(str, Enum):
    Basic = "Basic"
    Teams = "Teams"
    Enterprise = "Enterprise"

# User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: RoleEnum
    license: LicenseEnum

class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: RoleEnum
    license: LicenseEnum

    class Config:
        from_attributes = True

# Auth Tokens
class Token(BaseModel):
    access_token: str
    token_type: str

# Forgot & Reset Password
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    token: str
    new_password: str


# Language Translation
class TranslationOut(BaseModel):
    input_lang: str
    output_lang: str
    input_text: str
    output_text: str


# Sentiment Analysis Output
class SentimentOut(BaseModel):
    summary: str                  # e.g., "Positive"
    percentage: Dict[str, int]    # e.g., {"positive": 80, "negative": 20}

# Image Label
class ImageLabelOut(BaseModel):
    product_name: str
    category: str

    class Config:
        orm_mode = True
# Agentic Product Search (Agentic Flow)

class AgenticProductSearchIn(BaseModel):
    query: str
    action: str = "search"  # "search" or "purchase"
    product_id: Optional[Union[int, str]] = None

class AgenticProductSearchOut(BaseModel):
    message: str
    products: list[dict] = []
    purchased: bool = False