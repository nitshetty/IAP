import re
import os, json
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from db.models import RoleEnum, LicenseEnum, ProductRecord, User
from db.database import get_db
from auth.auth_manager import AuthManager
from api.schemas import AgenticProductSearchIn, AgenticProductSearchOut
from textblob import TextBlob
from groq import Groq

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

router = APIRouter()

# In-memory mapping for synthetic product IDs to names
synthetic_product_map = {}

def extract_keywords(text):
    blob = TextBlob(text)
    keywords = blob.noun_phrases
    return keywords if keywords else [text.strip()]

@router.post("/usecase/agentic-product-search", response_model=AgenticProductSearchOut)
@AuthManager.check_access([RoleEnum.Admin, RoleEnum.Editor, RoleEnum.Viewer], [LicenseEnum.Teams, LicenseEnum.Basic])
async def agentic_product_search(
    data: AgenticProductSearchIn = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(AuthManager.get_current_user)
):
    # Validate request body fields
    if not data.query or not isinstance(data.query, str) or data.query.strip() == "":
        raise HTTPException(status_code=422, detail="Invalid or missing 'query' field in request body.")

    if not data.action or data.action not in ["search", "purchase"]:
        raise HTTPException(status_code=422, detail="Invalid or missing 'action' field in request body.")

    if data.action == "purchase" and (not data.product_id or not isinstance(data.product_id, int)):
        raise HTTPException(status_code=422, detail="Invalid or missing 'product_id' field for purchase action.")

    query_str = data.query.strip() if data.query else ""

    if data.action == "search":
        if not query_str:
            return AgenticProductSearchOut(message="Please provide a product description to search.", products=[])

        # Extract keywords
        keywords = extract_keywords(query_str)
        all_matches = []

        # Search local DB
        for kw in keywords:
            matched = db.query(ProductRecord).filter(
                ProductRecord.name.ilike(f"%{kw}%") | ProductRecord.category.ilike(f"%{kw}%")
            ).all()
            all_matches.extend(matched)

        # Deduplicate
        seen = set()
        products = []
        for p in all_matches:
            if p.id not in seen:
                seen.add(p.id)
                products.append({
                    "id": p.id,
                    "name": p.name,
                    "category": p.category,
                    "price": p.price,
                    "in_stock": p.in_stock
                })
            if len(products) >= 3:
                break

        if products:
            return AgenticProductSearchOut(
                message=f"Based on your query: '{query_str}', here are some product suggestions:",
                products=products
            )

        # Fallback: Groq API if not found in DB
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"Suggest 3 unique products for: {query_str}. For each, provide name, category, price, and in_stock (random 1-10). Return as JSON list."
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            text = chat_completion.choices[0].message.content
            json_str = re.search(r'\[.*\]', text, re.DOTALL).group(0)
            groq_products = json.loads(json_str)
            # Deduplicate by name and assign synthetic IDs
            seen_names = set()
            unique_products = []
            synthetic_id = 10001
            for p in groq_products:
                if p["name"] not in seen_names:
                    seen_names.add(p["name"])
                    p["id"] = synthetic_id
                    synthetic_product_map[synthetic_id] = p["name"]  # Store mapping
                    synthetic_id += 1
                    unique_products.append(p)
                if len(unique_products) >= 3:
                    break
            return AgenticProductSearchOut(
                message=f"Groq suggestions for: '{query_str}'",
                products=unique_products
            )
        except Exception as e:
            return AgenticProductSearchOut(message=f"Error reaching Groq service: {e}", products=[])

    elif data.action == "purchase":
        # Simulate purchase (DB or synthetic product)
        product_id = getattr(data, 'product_id', None)
        if not product_id:
            return AgenticProductSearchOut(message="No product ID provided for purchase.", products=[], purchased=False)

        product_name = None
        # Search DB products
        db_product = None
        if product_id:
            db_product = db.query(ProductRecord).filter(ProductRecord.id == product_id).first()
        if db_product:
            product_name = db_product.name
            # Reduce stock size by 1
            if db_product.in_stock > 0:
                db_product.in_stock -= 1
                db.commit()
            else:
                return AgenticProductSearchOut(message="Product out of stock.", products=[], purchased=False)
        else:
            # If not in DB, assume synthetic product (IDs start from 10001)
            if product_id and product_id >= 10001:
                product_name = synthetic_product_map.get(product_id)  # Get name from map
        if product_name:
            return AgenticProductSearchOut(message=f"Successfully purchased: {product_name}", products=[], purchased=True)
        else:
            return AgenticProductSearchOut(message="Product not found.", products=[], purchased=False)