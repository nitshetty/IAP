from sqlalchemy import Column, Integer, String
from backend.db.db.database import Base

from sqlalchemy import Column, String, Enum as SqlEnum, Integer, Text

import enum

# Role and License Enums
class RoleEnum(str, enum.Enum):
    Admin = "Admin"
    Editor = "Editor"
    Viewer = "Viewer"

class LicenseEnum(str, enum.Enum):
    Basic = "Basic"
    Teams = "Teams"
    Enterprise = "Enterprise"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SqlEnum(RoleEnum), nullable=False)
    license = Column(SqlEnum(LicenseEnum), nullable=False)
    reset_token = Column(String, nullable=True)

# Sentiment labels for sentiment analysis use case
class SentimentLabel(Base):
    __tablename__ = "sentiment_labels"
    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, nullable=False)              
    keywords = Column(String, nullable=False)          
        
# Image labels for image classification use case
class ImageLabel(Base):
    __tablename__ = "image_labels"
    id = Column(Integer, primary_key=True, index=True)
    ocr_text = Column(String, nullable=False)           
    product_name = Column(String, nullable=False)       
    category = Column(String, nullable=False)           

# Translations for language translation use case
class LanguageTranslation(Base):
    __tablename__ = "language_translations"
    id = Column(Integer, primary_key=True, index=True)
    input_lang = Column(String, nullable=False)
    output_lang = Column(String, nullable=False)
    input_text = Column(String, nullable=False)
    output_text = Column(String, nullable=False)

# Products for agentic product search use case
class ProductRecord(Base):
    __tablename__ = "products_agentic"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    category = Column(String, nullable=False)
    price = Column(String, nullable=False)
    in_stock = Column(Integer, nullable=False)