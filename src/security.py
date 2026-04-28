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
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger("finsight.security")

# ---------- Token verification ----------
# We use Google's public token info endpoint for Firebase ID token validation.
# This avoids needing firebase-admin SDK (heavy) and works on any deployment.

import json
import time
from urllib.request import urlopen, Request
from urllib.error import URLError
import jwt as pyjwt  # PyJWT

# Firebase project ID — must match what the frontend uses
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "finsight-ai-app")

# Google's public JWK endpoint for Firebase
GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"

# Cache for Google's public certificates
_cert_cache = {"certs": None, "expires_at": 0}


def _fetch_google_certs() -> dict:
    """Fetch and cache Google's public certificates for Firebase token validation."""
    now = time.time()
    if _cert_cache["certs"] and now < _cert_cache["expires_at"]:
        return _cert_cache["certs"]

    try:
        req = Request(GOOGLE_CERTS_URL, headers={"User-Agent": "FinSight-AI/3.0"})
        with urlopen(req, timeout=10) as resp:
            certs = json.loads(resp.read().decode("utf-8"))
            # Cache for 1 hour (Google certs rotate ~every 6 hours)
            _cert_cache["certs"] = certs
            _cert_cache["expires_at"] = now + 3600
            return certs
    except (URLError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to fetch Google certs: {e}")
        if _cert_cache["certs"]:
            return _cert_cache["certs"]  # use stale cache
        raise HTTPException(503, "Unable to verify tokens: certificate fetch failed")


def verify_firebase_token(id_token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded payload.

    Validates:
    - Signature against Google's public keys
    - Expiration (exp)
    - Issued-at (iat) 
    - Audience (aud) matches our project
    - Issuer (iss) matches Firebase
    - Subject (sub) is non-empty

    Returns the decoded token payload dict.
    """
    try:
        # Decode header to find the key ID
        unverified_header = pyjwt.get_unverified_header(id_token)
        kid = unverified_header.get("kid")
        if not kid:
            raise ValueError("Token has no key ID (kid)")

        # Fetch Google public certs
        certs = _fetch_google_certs()
        cert_pem = certs.get(kid)
        if not cert_pem:
            # Force refresh in case keys rotated
            _cert_cache["expires_at"] = 0
            certs = _fetch_google_certs()
            cert_pem = certs.get(kid)
            if not cert_pem:
                raise ValueError(f"No matching public key for kid={kid}")

        # Verify and decode
        payload = pyjwt.decode(
            id_token,
            cert_pem,
            algorithms=["RS256"],
            audience=FIREBASE_PROJECT_ID,
            issuer=f"https://securetoken.google.com/{FIREBASE_PROJECT_ID}",
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            }
        )

        # Firebase requires sub to be non-empty
        if not payload.get("sub"):
            raise ValueError("Token has no subject (sub)")

        return payload

    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except pyjwt.InvalidTokenError as e:
        logger.warning(f"Invalid Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
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
