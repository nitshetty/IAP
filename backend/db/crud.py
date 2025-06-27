from sqlalchemy.orm import Session
from db.models import User, SentimentLabel, LanguageTranslation
from api.schemas import UserCreate
from core.utils import hash_password

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    hashed_password = hash_password(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,
        license=user.license
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_sentiment_labels(db: Session):
    return db.query(SentimentLabel).all()

def get_language_translation(db: Session, input_lang: str, output_lang: str, input_text: str):
    return db.query(LanguageTranslation).filter(
        LanguageTranslation.input_lang == input_lang,
        LanguageTranslation.output_lang == output_lang,
        LanguageTranslation.input_text.ilike(f"%{input_text}%")
    ).all()