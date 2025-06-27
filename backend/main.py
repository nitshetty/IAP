from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Security, Form, Body
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import functools
from db.models import User, RoleEnum, LicenseEnum, SentimentLabel, LanguageTranslation
from sqlalchemy.orm import Session
from sqlalchemy import or_
from db.database import SessionLocal, engine, Base, get_db
from core.utils import hash_password, verify_password, create_reset_token
from core.logger import LoggingMiddleware
from typing import List
from datetime import datetime, timedelta
import os, io
import jwt
from fastapi.responses import FileResponse, PlainTextResponse
from docx import Document
import PyPDF2
from PIL import Image
import tempfile
import fitz
import docx
import easyocr
import numpy as np
import cv2
from service.sentiment.analyze_utils import analyze_sentiment_with_percentage
from service.sentiment.sentiment_routes import router as sentiment_router
from service.langauge.translation_routes import router as translation_router
from api.routes import router as routes_router
from service.image.image_classification_routes import router as image_classification_router
from service.agentic.agentic_product_search_routes import router as agentic_product_search_router
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from auth.auth_manager import AuthManager


app = FastAPI()
app.add_middleware(LoggingMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Base.metadata.create_all(bind=engine)

# JWT Auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Register routers
app.include_router(routes_router)
app.include_router(sentiment_router)
app.include_router(translation_router)
app.include_router(image_classification_router)
app.include_router(agentic_product_search_router)