import functools
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
from db.models import User, RoleEnum, LicenseEnum
from core.utils import verify_password
from db.database import get_db
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class AuthManager:
    @staticmethod
    def create_access_token(data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str):
        user = db.query(User).filter(User.email == email).first()
        print(f"[DEBUG] DB user lookup for email={email}: {user}")
        if not user:
            print(f"[DEBUG] No user found for email={email}")
            return None
        password_ok = verify_password(password, user.hashed_password)
        print(f"[DEBUG] Password verification for {email}: {password_ok}")
        if not password_ok:
            print(f"[DEBUG] Password mismatch for {email}")
            return None
        print(f"[DEBUG] User authenticated: {email}")
        return user

    @staticmethod
    def get_current_user(token: str = Security(oauth2_scheme), db: Session = Depends(get_db)) -> User:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if email is None:
                raise HTTPException(status_code=401, detail="Invalid token")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    @staticmethod
    def check_access(allowed_roles: list[RoleEnum], allowed_licenses: list[LicenseEnum]):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                current_user = kwargs.get('current_user')
                if current_user is None:
                    raise HTTPException(status_code=401, detail="User not found")
                if current_user.role not in allowed_roles:
                    raise HTTPException(status_code=403, detail="Role not permitted")
                if current_user.license not in allowed_licenses:
                    raise HTTPException(status_code=403, detail="License not permitted")
                return await func(*args, **kwargs)
            return wrapper
        return decorator
