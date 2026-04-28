"""
FinSight AI — Unified Security Module
======================================
Firebase ID token verification for all protected routes.
Eliminates the dual-auth system (Firebase frontend + local JWT backend).

Now ALL tokens come from Firebase Auth and are verified using Google's
public keys. No local JWT minting, no local password hashing, no users table.
"""

import os
import logging
import time
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger("finsight.security")

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# We use google.oauth2 to verify the token without requiring a service account JSON file.
# firebase_admin requires credentials on Render, but google.oauth2 only requires the public project ID.
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "finsight-ai-app")
_request = google_requests.Request()

def verify_firebase_token(id_token_str: str) -> dict:
    """Verify a Firebase ID token using google.oauth2."""
    try:
        # verify_firebase_token automatically fetches Google's public certificates,
        # checks the signature, audience, issuer, expiration, etc.
        payload = id_token.verify_firebase_token(id_token_str, _request, audience=FIREBASE_PROJECT_ID)
        return dict(payload)
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------- FastAPI dependency ----------

# Use HTTPBearer instead of OAuth2PasswordBearer — cleaner for Firebase tokens
security_scheme = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> dict:
    """
    FastAPI dependency that verifies the Firebase ID token from the
    Authorization header and returns the decoded user info.

    Returns a dict with at least: uid, email, name
    """
    token = credentials.credentials
    payload = verify_firebase_token(token)

    return {
        "uid": payload.get("sub", ""),
        "email": payload.get("email", ""),
        "name": payload.get("name", payload.get("email", "User")),
        "email_verified": payload.get("email_verified", False),
    }


# ---------- Session expiry helpers ----------

# 24-hour session window (in seconds)
SESSION_MAX_AGE_SECONDS = 24 * 60 * 60  # 86400


def check_session_age(payload: dict) -> bool:
    """
    Check if the Firebase auth_time is within 24 hours.
    Returns True if session is still valid, False if session is too old.
    """
    auth_time = payload.get("auth_time", 0)
    if not auth_time:
        return True  # can't check, allow

    now = time.time()
    session_age = now - auth_time
    return session_age < SESSION_MAX_AGE_SECONDS


async def get_current_user_strict(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> dict:
    """
    Same as get_current_user but also enforces 24h session expiry.
    If the user authenticated more than 24h ago, they must re-login.
    """
    token = credentials.credentials
    payload = verify_firebase_token(token)

    if not check_session_age(payload):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired (24h limit). Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "uid": payload.get("sub", ""),
        "email": payload.get("email", ""),
        "name": payload.get("name", payload.get("email", "User")),
        "email_verified": payload.get("email_verified", False),
        "auth_time": payload.get("auth_time", 0),
    }


# Legacy exports for backward compatibility (no longer used by main.py)
class Token(BaseModel):
    access_token: str
    token_type: str
