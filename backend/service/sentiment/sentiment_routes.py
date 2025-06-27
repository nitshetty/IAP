from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from sqlalchemy.orm import Session
from db.models import RoleEnum, LicenseEnum, SentimentLabel, User
from db.database import get_db
from api.schemas import SentimentOut
from auth.auth_manager import AuthManager
from fastapi.responses import PlainTextResponse, FileResponse
import io
from docx import Document
import PyPDF2
from transformers import pipeline
from groq import Groq
import os

router = APIRouter()

@router.post("/usecase/sentiment-analysis", response_model=SentimentOut)
@AuthManager.check_access([RoleEnum.Editor], [LicenseEnum.Teams, LicenseEnum.Basic])
async def sentiment_analysis(
    text_input: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):

    # Normalize both fields safely
    text_input_clean = (text_input or "").strip()
    file_valid = file is not None and hasattr(file, 'filename') and file.filename.strip() != ""

    if text_input_clean and file_valid:
        raise HTTPException(
            status_code=400,
            detail="Please provide either text input or a file, but not both."
        )

    if not text_input_clean and not file_valid:
        raise HTTPException(
            status_code=400,
            detail="Please provide either text input or a file."
        )

    content = text_input if text_input_clean else ""
    if file_valid:
        ext = file.filename.split(".")[-1].lower()
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if ext == "pdf":
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            content = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif ext == "docx":
            doc = Document(io.BytesIO(file_content))
            content = "\n".join(p.text for p in doc.paragraphs)
        elif ext == "txt":
            content = file_content.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    content = content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="No text found in the document or input.")

    # Try DB-based sentiment analysis first
    labels = db.query(SentimentLabel).all()
    words = content.lower().split()
    label_match_counts = {label.label.lower(): 0 for label in labels}
    total_keywords = 0
    for label_entry in labels:
        keywords = [kw.strip().lower() for kw in label_entry.keywords.split(",")]
        for word in words:
            if word in keywords:
                label_match_counts[label_entry.label.lower()] += 1
                total_keywords += 1
    if total_keywords > 0:
        percentages = {}
        for label, count in label_match_counts.items():
            if count > 0:
                percentages[label] = int((count / total_keywords) * 100)
        summary = max(percentages, key=percentages.get)
        result_obj = {"summary": summary.capitalize(), "percentage": percentages}
        return result_obj

    # Fallback: Use Groq generative AI for nuanced sentiment analysis ONLY
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=groq_api_key)
        system_prompt = (
            "You are a sentiment analysis assistant. "
            "Given a text, analyze its sentiment and respond ONLY in this JSON format: "
            "{\"summary\": <one of: Positive, Negative, Neutral>, \"percentage\": {\"Positive\": <int>, \"Negative\": <int>, \"Neutral\": <int>}}. "
            "Percentages must sum to 100. Do not add any explanation."
        )
        user_prompt = f"Analyze the sentiment of the following text: {content}"
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        import json as pyjson
        import re
        response = chat_completion.choices[0].message.content.strip()
        # Extract JSON from response (in case model adds extra text)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            response_json = pyjson.loads(match.group(0))
            # Only keep allowed sentiment keys
            allowed_keys = ['Positive', 'Negative', 'Neutral']
            perc = response_json.get('percentage', {})
            # Use floats for accuracy
            perc_floats = {k: float(perc.get(k, 0)) for k in allowed_keys}
            # Threshold: set any value below 20% to 0
            threshold = 20.0
            perc_thresholded = {k: (v if v >= threshold else 0.0) for k, v in perc_floats.items()}
            if sum(perc_thresholded.values()) == 0:
                # If all are below threshold, keep the original highest as 100, rest 0
                main_key = max(perc_floats, key=perc_floats.get)
                perc_thresholded = {k: (100.0 if k == main_key else 0.0) for k in allowed_keys}
            # Normalize to 100
            total = sum(perc_thresholded.values())
            if total > 0:
                perc_norm = {k: (v * 100.0 / total) for k, v in perc_thresholded.items()}
            else:
                perc_norm = {k: 0.0 for k in allowed_keys}
            perc_rounded = {k: int(round(v)) for k, v in perc_norm.items()}
            diff = 100 - sum(perc_rounded.values())
            if diff != 0:
                main_key = max(perc_norm, key=perc_norm.get)
                perc_rounded[main_key] += diff
            cleaned_response = {
                'summary': max(perc_rounded, key=perc_rounded.get),
                'percentage': perc_rounded
            }
            return cleaned_response
        else:
            raise Exception(f"Groq did not return valid JSON: {response}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
