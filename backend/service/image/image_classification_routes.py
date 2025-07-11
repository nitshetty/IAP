from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from db.models import RoleEnum, LicenseEnum, ImageLabel, User
from db.database import get_db
from auth.auth_manager import AuthManager
from typing import List, Optional
import easyocr
import numpy as np
import cv2
from api.schemas import ImageLabelOut
from sqlalchemy import or_
import os
import re
import json as pyjson
from groq import Groq
import base64
import time

router = APIRouter()

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

@router.post("/usecase/image-classification", response_model=List[ImageLabelOut])
@AuthManager.check_access([RoleEnum.Admin, RoleEnum.Editor, RoleEnum.Viewer], [LicenseEnum.Teams, LicenseEnum.Basic])
async def image_classification(
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user),
):
    # Null or missing file
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded. Please upload a valid .jpg or .png image.")

    # Handle common browser MIME issues
    content_type = (file.content_type or "").lower().strip()
    valid_file_types = {"image/jpeg", "image/jpg", "image/png"}

    if content_type not in valid_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{content_type}'. Please upload a valid .jpg or .png image."
        )

    image_bytes = await file.read()

    # Sanity check: empty file or fake content
    if not image_bytes or image_bytes.strip() == b'string' or len(image_bytes) <= 7:
        raise HTTPException(status_code=400, detail="Uploaded image is invalid or empty.")

    # Try to decode image to ensure it's really valid
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        # OCR Step
        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(img_np, detail=0)
        ocr_text = " ".join(result).strip()
        print(f"OCR extracted text: '{ocr_text}'")

        if ocr_text:
            words = [w for w in ocr_text.split() if len(w) > 2]
            if words:
                filters = [ImageLabel.ocr_text.ilike(f"%{word}%") for word in words]
                results = db.query(ImageLabel).filter(or_(*filters)).all()
                if results:
                    return [ImageLabelOut(product_name=r.product_name, category=r.category) for r in results]

        # Vision-Language Fallback
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_path = f"data:image/jpeg;base64,{base64_image}"

        prompt = (
            "Analyze the image and classify the products accurately. "
            "Provide detailed classifications in JSON format: "
            "[{product_name: ..., category: food/beverage/other}]. "
            "Use external knowledge to ensure logical consistency."
        )

        for attempt in range(MAX_RETRIES):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_path}},
                            ],
                        }
                    ],
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                )

                if not chat_completion or not chat_completion.choices:
                    print("Groq API did not return any choices.")
                    raise HTTPException(status_code=500, detail="No response from vision model.")

                content = chat_completion.choices[0].message.content
                if not content:
                    print("Groq API returned empty content.")
                    raise HTTPException(status_code=500, detail="Empty content from vision model.")

                match = re.search(r"\[.*?\]", content, re.DOTALL)
                if not match:
                    print(f"Groq API response does not contain valid JSON: {content}")
                    raise HTTPException(status_code=500, detail="No valid JSON list found in model output.")

                try:
                    parsed = pyjson.loads(match.group(0))
                except pyjson.JSONDecodeError as e:
                    print(f"JSON parsing failed: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"JSON parse failed: {str(e)}")

                if not isinstance(parsed, list):
                    print(f"Parsed result is not a list: {parsed}")
                    raise HTTPException(status_code=500, detail="Parsed result is not a list.")

                results = []
                for item in parsed:
                    product = item.get("product_name", "unknown").lower()
                    category = item.get("category", "unknown").lower()
                    if product == "string" or category == "string":
                        print(f"Skipping invalid product or category: {item}")
                        continue

                    # Validate against database dynamically
                    db_result = db.query(ImageLabel).filter(ImageLabel.product_name.ilike(f"%{product}%")).first()
                    if db_result:
                        category = db_result.category
                    else:
                        print(f"No database match found for product: {product}")

                    results.append(ImageLabelOut(product_name=product, category=category))

                if results:
                    return results

                print("No valid classification results found.")
                raise HTTPException(status_code=500, detail="No valid classification results found.")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print("Max retries reached. Falling back to default response.")
                    raise HTTPException(status_code=500, detail="Vision model failed after multiple attempts.")

    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))