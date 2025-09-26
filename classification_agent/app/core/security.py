"""
Security utilities for authentication and authorization.
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """Handles authentication and authorization."""

    def __init__(self):
        self.algorithm = "HS256"
        self.secret_key = settings.secret_key

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning("Token verification failed", error=str(e))
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key."""
        if not settings.api_key:
            return True  # No API key required
        return secrets.compare_digest(api_key, settings.api_key)


security_manager = SecurityManager()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """Dependency to get current authenticated user."""
    # Check for X-API-Key header first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        if security_manager.verify_api_key(api_key):
            return {"user": "api_key_user", "auth_type": "api_key"}
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Check for Bearer token
    if not credentials:
        if settings.api_key:  # API key required but not provided
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"user": "anonymous"}  # No auth required

    # Check if it's an API key or JWT token
    token = credentials.credentials

    # Try API key first
    if security_manager.verify_api_key(token):
        return {"user": "api_key_user", "auth_type": "api_key"}

    # Try JWT token
    try:
        payload = security_manager.verify_token(token)
        return {"user": payload.get("sub"), "auth_type": "jwt", "payload": payload}
    except HTTPException:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_auth(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Dependency that requires authentication."""
    if user.get("user") == "anonymous":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user