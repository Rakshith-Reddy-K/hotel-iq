"""
Authentication Routes for HotelIQ
=================================
Add this file to your backend folder and import the router in main.py

Usage in main.py:
    from auth_routes import auth_router
    app.include_router(auth_router, prefix="/api/v1")
"""

import os
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
import bcrypt
from jose import jwt

# Import your database connection
from sql.db_pool import get_connection

# ======================================================
# CONFIGURATION
# ======================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# ======================================================
# SCHEMAS
# ======================================================

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8, max_length=100)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenRefresh(BaseModel):
    refresh_token: str

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str]
    last_name: Optional[str]
    is_active: bool
    created_at: str
    last_login: Optional[str]

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class LoginResponse(BaseModel):
    user: UserResponse
    tokens: TokenResponse

class MessageResponse(BaseModel):
    message: str
    success: bool = True

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)

def get_password_hash(password: str) -> str:
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')

def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": user_id, "exp": expire, "type": "access"}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token() -> str:
    return secrets.token_urlsafe(64)

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def user_row_to_response(row) -> UserResponse:
    return UserResponse(
        id=str(row[0]),
        email=row[1],
        username=row[2],
        first_name=row[4],
        last_name=row[5],
        is_active=row[6],
        created_at=row[7].isoformat() if row[7] else None,
        last_login=row[9].isoformat() if row[9] else None,
    )

# ======================================================
# ROUTES
# ======================================================

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            
            # Check if email exists
            cur.execute("SELECT id FROM users WHERE email = %s", (user_data.email,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Check if username exists
            cur.execute("SELECT id FROM users WHERE username = %s", (user_data.username,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Username already taken")
            
            # Create user
            user_id = str(uuid.uuid4())
            password_hash = get_password_hash(user_data.password)
            
            cur.execute("""
                INSERT INTO users (id, email, username, password_hash, first_name, last_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, email, username, password_hash, first_name, last_name, is_active, created_at, updated_at, last_login
            """, (user_id, user_data.email, user_data.username, password_hash, 
                  user_data.first_name, user_data.last_name))
            
            user_row = cur.fetchone()
            
            # Create tokens
            access_token = create_access_token(user_id)
            refresh_token = create_refresh_token()
            
            # Store refresh token
            cur.execute("""
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
            """, (user_id, hash_token(refresh_token), 
                  datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)))
            
            cur.close()
            
            return LoginResponse(
                user=user_row_to_response(user_row),
                tokens=TokenResponse(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                )
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=LoginResponse)
async def login(credentials: UserLogin):
    """Login with email and password."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            
            # Get user
            cur.execute("""
                SELECT id, email, username, password_hash, first_name, last_name, is_active, created_at, updated_at, last_login
                FROM users WHERE email = %s
            """, (credentials.email,))
            
            user_row = cur.fetchone()
            
            if not user_row or not verify_password(credentials.password, user_row[3]):
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            if not user_row[6]:  # is_active
                raise HTTPException(status_code=403, detail="Account is deactivated")
            
            user_id = str(user_row[0])
            
            # Update last login
            cur.execute("UPDATE users SET last_login = %s WHERE id = %s", 
                        (datetime.utcnow(), user_id))
            
            # Create tokens
            access_token = create_access_token(user_id)
            refresh_token = create_refresh_token()
            
            # Store refresh token
            cur.execute("""
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
            """, (user_id, hash_token(refresh_token), 
                  datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)))
            
            cur.close()
            
            return LoginResponse(
                user=user_row_to_response(user_row),
                tokens=TokenResponse(
                    access_token=access_token,
                    refresh_token=refresh_token,
                    expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                )
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(token_data: TokenRefresh):
    """Refresh access token."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            
            token_hash = hash_token(token_data.refresh_token)
            
            # Validate refresh token
            cur.execute("""
                SELECT user_id FROM refresh_tokens 
                WHERE token_hash = %s AND revoked = FALSE AND expires_at > %s
            """, (token_hash, datetime.utcnow()))
            
            result = cur.fetchone()
            if not result:
                raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
            
            user_id = str(result[0])
            
            # Revoke old token
            cur.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = %s", (token_hash,))
            
            # Create new tokens
            access_token = create_access_token(user_id)
            new_refresh_token = create_refresh_token()
            
            # Store new refresh token
            cur.execute("""
                INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
                VALUES (%s, %s, %s)
            """, (user_id, hash_token(new_refresh_token), 
                  datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)))
            
            cur.close()
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout", response_model=MessageResponse)
async def logout(token_data: TokenRefresh):
    """Logout by revoking refresh token."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            token_hash = hash_token(token_data.refresh_token)
            cur.execute("UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = %s", (token_hash,))
            cur.close()
            
        return MessageResponse(message="Successfully logged out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export router
auth_router = router