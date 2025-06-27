from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from db.database import get_db
from db.crud import get_user_by_email, create_user, get_sentiment_labels, get_language_translation
from api.schemas import UserCreate, SentimentOut, Token, ForgotPasswordRequest, ResetPasswordRequest
from db.models import RoleEnum, LicenseEnum, User
from core.utils import hash_password, create_reset_token
from fastapi.security import OAuth2PasswordRequestForm
from auth.auth_manager import AuthManager

router = APIRouter()

@router.post("/signup")
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    existing_user = get_user_by_email(db, user_in.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db, user_in)


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    print(f"[DEBUG] Login attempt: username={form_data.username}")
    user = AuthManager.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        print(f"[DEBUG] Invalid credentials for: {form_data.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    print(f"[DEBUG] Login successful for: {form_data.username}")
    access_token = AuthManager.create_access_token(data={"sub": user.email, "role": user.role.value, "license": user.license.value})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/forgot-password")
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    token = create_reset_token()
    user.reset_token = token
    db.commit()
    return {"message": "Reset token generated", "reset_token": token}

@router.post("/reset-password")
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or user.reset_token != request.token:
        raise HTTPException(status_code=400, detail="Invalid token or user")
    user.hashed_password = hash_password(request.new_password)
    user.reset_token = None
    db.commit()
    return {"message": "Password reset successful"}
