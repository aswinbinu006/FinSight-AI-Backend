"""
FinSight AI — Production API Server v4.1
=========================================
Strict .pkl-only backend — NO runtime training:
- Loads pre-trained models ONLY from saved_models/*.pkl
- Server REFUSES to start if any .pkl file is missing
- Firebase Auth only (no local JWT)
- Pydantic validation on ALL endpoints
- 24h session enforcement
- Race-condition-safe model loading

To generate .pkl files, run:
    python -m finsight_models_production.train_and_save_models
"""

import os
import sys
import logging
import contextlib
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()

# --- Security (Firebase Auth) ---
try:
    from security import get_current_user, get_current_user_strict
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from security import get_current_user, get_current_user_strict

# --- ML Models ---
from finsight_models_production import (
    HealthModel, WasteModel, GoalModel, ClusterModel,
    score_behavioral_answers, explain_health, explain_waste, explain_goal,
    chat_with_copilot
)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION & VALIDATION
# ═══════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("finsight.api")

# Validate critical environment variables at startup
_missing_env = []
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    _missing_env.append("GEMINI_API_KEY")
    logger.warning("GEMINI_API_KEY is missing — behavioral scoring will use defaults")

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", os.getenv("VITE_FIREBASE_PROJECT_ID", "finsight-ai-app"))
if not FIREBASE_PROJECT_ID:
    _missing_env.append("FIREBASE_PROJECT_ID")
    logger.warning("FIREBASE_PROJECT_ID is missing — token verification may fail")

# Set it so security.py picks it up
os.environ["FIREBASE_PROJECT_ID"] = FIREBASE_PROJECT_ID

if _missing_env:
    logger.warning(f"Missing environment variables: {', '.join(_missing_env)}")

# ═══════════════════════════════════════════════════════════════════════
# ML MODEL MANAGEMENT — Thread-safe with pickle persistence
# ═══════════════════════════════════════════════════════════════════════

MODEL_DIR = Path(os.path.dirname(__file__)) / "finsight_models_production" / "saved_models"


class AIModelSuite:
    """Thread-safe ML model loader — .pkl files ONLY, no runtime training.
    
    The backend strictly loads pre-trained .pkl models from the saved_models/
    directory. If any model file is missing, initialization fails with a clear
    error. To generate the .pkl files, run:
    
        python -m finsight_models_production.train_and_save_models
    """

    # The 4 required model files
    REQUIRED_MODELS = ["health", "waste", "goal", "cluster"]

    def __init__(self):
        self.health = HealthModel()
        self.waste = WasteModel()
        self.goal = GoalModel()
        self.cluster = ClusterModel()
        self._ready = False
        self._lock = threading.Lock()

    @property
    def ready(self):
        return self._ready

    def _load_model(self, model, name: str):
        """Load a pre-trained .pkl model from disk. Raises if missing."""
        filepath = MODEL_DIR / f"{name}.pkl"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Required model file not found: {filepath}\n"
                f"  → Run this command first to generate it:\n"
                f"    python -m finsight_models_production.train_and_save_models"
            )

        if not hasattr(model, 'load'):
            raise AttributeError(f"Model '{name}' does not have a load() method.")

        model.load(str(filepath))
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Loaded {name}.pkl ({size_mb:.1f} MB)")

    def initialize(self):
        """Load ALL models from .pkl files. Thread-safe. No training."""
        with self._lock:
            if self._ready:
                return

            logger.info("═" * 60)
            logger.info("Loading pre-trained models from .pkl files...")
            logger.info(f"Model directory: {MODEL_DIR}")

            # Verify all .pkl files exist BEFORE loading any
            missing = []
            for name in self.REQUIRED_MODELS:
                pkl_path = MODEL_DIR / f"{name}.pkl"
                if not pkl_path.exists():
                    missing.append(str(pkl_path))

            if missing:
                msg = (
                    f"\n{'=' * 60}\n"
                    f"  FATAL: {len(missing)} required .pkl model file(s) missing!\n"
                    f"{'=' * 60}\n"
                    f"  Missing files:\n"
                )
                for m in missing:
                    msg += f"    ✗ {m}\n"
                msg += (
                    f"\n  To fix this, run the training script:\n"
                    f"    python -m finsight_models_production.train_and_save_models\n"
                    f"\n  This will generate all .pkl files in saved_models/.\n"
                    f"  The backend CANNOT train models at runtime.\n"
                    f"{'=' * 60}"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)

            # All files confirmed — now load them
            import time
            start = time.time()

            self._load_model(self.health, "health")
            self._load_model(self.waste, "waste")
            self._load_model(self.goal, "goal")
            self._load_model(self.cluster, "cluster")

            elapsed = time.time() - start

            self._ready = True
            logger.info(f"═" * 60)
            logger.info(f"All 4 models loaded in {elapsed:.2f}s — Engine READY")
            logger.info(f"═" * 60)


ai = AIModelSuite()


# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS — validated input for every endpoint
# ═══════════════════════════════════════════════════════════════════════

class HealthPredictionRequest(BaseModel):
    behavioral_scores: dict
    context: Optional[dict] = Field(default_factory=dict)

    @validator("behavioral_scores")
    def validate_scores(cls, v):
        if not v:
            raise ValueError("behavioral_scores cannot be empty")
        return v


class SubscriptionItem(BaseModel):
    name: str = Field(default="Other")
    price: float = Field(ge=0, default=100)
    billing_cycle: str = Field(default="monthly")
    usage_frequency: str = Field(default="sometimes")
    awareness: str = Field(default="sometimes")
    necessity: str = Field(default="sometimes")
    has_perks: bool = Field(default=False)


class WastePredictionRequest(BaseModel):
    subscriptions: List[SubscriptionItem] = Field(default_factory=list)

    @validator("subscriptions")
    def validate_subs(cls, v):
        if not v:
            raise ValueError("At least one subscription is required")
        return v


class GoalPredictionRequest(BaseModel):
    behavioral_scores: dict = Field(default_factory=dict)
    goal_description: str = Field(default="Wealth Building")
    target_amount: float = Field(ge=0, default=50000)
    saved_so_far: float = Field(ge=0, default=0)
    monthly_savings: float = Field(ge=0, default=1000)
    timeline_months: int = Field(ge=1, le=360, default=24)


class BehavioralScoringRequest(BaseModel):
    answers: dict

    @validator("answers")
    def validate_answers(cls, v):
        required_keys = ["payday", "weekend", "subs", "impulse", "goal",
                         "stress", "social", "emergency", "future", "learning"]
        missing = [k for k in required_keys if k not in v]
        if missing:
            raise ValueError(f"Missing answers: {', '.join(missing)}")
        return v


class CopilotChatRequest(BaseModel):
    message: str
    context: dict = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pre-trained .pkl models at startup. Crashes if any are missing."""
    ai.initialize()  # Will raise FileNotFoundError if .pkl files are missing
    yield


app = FastAPI(
    title="FinSight AI",
    description="Institutional-grade financial intelligence prediction engine.",
    version="4.2.0",
    lifespan=lifespan,
)

# CORS — environment-aware origins
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development") == "production"

# Production origins (always allowed)
_CORS_ORIGINS = [
    "https://finsight-ai-app.web.app",
    "https://finsight-ai-app.firebaseapp.com",
]

# Development origins (only allowed when not in production)
if not IS_PRODUCTION:
    _CORS_ORIGINS.extend([
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# UTILITY: ensure models are ready before prediction
# ═══════════════════════════════════════════════════════════════════════

def require_models():
    """Raise 503 if models aren't loaded. Should never happen since we load at startup."""
    if not ai.ready:
        raise HTTPException(
            status_code=503,
            detail="AI models are not loaded. Server may have started without .pkl files.",
        )


# ═══════════════════════════════════════════════════════════════════════
# RATE LIMITING — Simple in-memory rate limiter for auth endpoints
# ═══════════════════════════════════════════════════════════════════════

from collections import defaultdict

class RateLimiter:
    """In-memory rate limiter. Tracks request counts per IP per window."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits = defaultdict(list)  # ip -> [timestamp, ...]
        self._lock = threading.Lock()

    def check(self, ip: str) -> bool:
        """Returns True if request is allowed, False if rate-limited."""
        import time
        now = time.time()
        with self._lock:
            # Prune old entries
            self._hits[ip] = [t for t in self._hits[ip] if now - t < self.window]
            if len(self._hits[ip]) >= self.max_requests:
                return False
            self._hits[ip].append(now)
            return True


# 10 requests per minute on auth endpoints
auth_limiter = RateLimiter(max_requests=10, window_seconds=60)


def rate_limit_auth(request: Request):
    """FastAPI dependency that rate-limits auth endpoints by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    if not auth_limiter.check(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down and try again in a minute.",
        )


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

# --- Health Check (no auth required) ---

@app.get("/")
@app.get("/status")
@app.get("/health")
async def get_status():
    """Health check endpoint for monitoring and frontend verification."""
    return {
        "status": "operational",
        "service": "finsight-ai",
        "timestamp": datetime.now().isoformat(),
        "models_ready": ai.ready,
        "version": "4.0.0",
    }


# --- Session validation endpoint ---

@app.get("/auth/validate")
async def validate_session(
    request: Request,
    user: dict = Depends(get_current_user_strict),
    _: None = Depends(rate_limit_auth),
):
    """Validate that the user's session is still active (< 24h)."""
    return {
        "valid": True,
        "uid": user["uid"],
        "email": user["email"],
    }


# --- ML Prediction Routes (all require Firebase auth + 24h session) ---

@app.post("/predict/health")
async def predict_health(
    req: HealthPredictionRequest,
    user: dict = Depends(get_current_user_strict),
):
    """Predict financial health score from behavioral scores."""
    require_models()
    try:
        scores = req.behavioral_scores
        ctx = req.context or {}
        emi = ctx.get("emi_burden", "light")
        income = ctx.get("income_bracket", "50k-100k")
        res = ai.health.predict(scores, emi, income)
        res["explanation"] = explain_health(res)
        return res
    except AssertionError as e:
        raise HTTPException(503, "Models not ready. Please retry.")
    except Exception as e:
        logger.error(f"Health prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/predict/waste")
async def predict_waste(
    req: WastePredictionRequest,
    user: dict = Depends(get_current_user_strict),
):
    """Predict waste scores for subscriptions."""
    require_models()
    try:
        subs = [s.model_dump() for s in req.subscriptions]
        res = ai.waste.predict_bulk(subs)
        res["explanation"] = explain_waste(res)
        return res
    except AssertionError as e:
        raise HTTPException(503, "Models not ready. Please retry.")
    except Exception as e:
        logger.error(f"Waste prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/predict/goal")
async def predict_goal(
    req: GoalPredictionRequest,
    user: dict = Depends(get_current_user_strict),
):
    """Predict goal achievement probability."""
    require_models()
    try:
        res = ai.goal.predict(
            behavioral_scores=req.behavioral_scores,
            goal_description=req.goal_description,
            goal_target=req.target_amount,
            saved_so_far=req.saved_so_far,
            monthly_savings=req.monthly_savings,
            months_remaining=req.timeline_months,
        )
        res["explanation"] = explain_goal(res)
        return res
    except AssertionError as e:
        raise HTTPException(503, "Models not ready. Please retry.")
    except Exception as e:
        logger.error(f"Goal prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/predict/behavioral-scores")
async def get_behavioral_scores(
    req: BehavioralScoringRequest,
    user: dict = Depends(get_current_user_strict),
):
    """Score behavioral answers using Gemini AI."""
    try:
        scores = score_behavioral_answers(req.answers)
        return scores
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Behavioral scoring error: {e}")
        raise HTTPException(500, f"Scoring failed: {str(e)}")


@app.post("/chat/copilot")
async def copilot_chat(
    req: CopilotChatRequest,
    user: dict = Depends(get_current_user_strict),
):
    """Deep financial intelligence chat using Gemini AI."""
    try:
        response = chat_with_copilot(req.message, req.context)
        return {"response": response}
    except Exception as e:
        logger.error(f"Copilot chat error: {e}")
        raise HTTPException(500, f"AI Co-Pilot is temporarily unavailable: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
