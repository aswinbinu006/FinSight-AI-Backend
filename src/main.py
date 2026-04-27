import os
import sys
import logging
import sqlite3
import json
import contextlib
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv

# Auth utilities
try:
    from security import Token, get_password_hash, verify_password, create_access_token, get_current_user
except ImportError:
    # Fallback for local development if security.py is missing or in wrong place
    sys.path.append(os.path.dirname(__file__))
    from security import Token, get_password_hash, verify_password, create_access_token, get_current_user

# ML Models
from finsight_models_production import (
    HealthModel, WasteModel, GoalModel, ClusterModel, 
    score_behavioral_answers, explain_health, explain_waste, explain_goal
)

load_dotenv()

# --- CONFIGURATION ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "database/finsight.db")

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("finsight.api")

# --- DATABASE ENGINE ---
class DatabaseManager:
    def __init__(self):
        self.use_postgres = False
        try:
            import psycopg2 # type: ignore
            if DATABASE_URL.startswith(("postgresql://", "postgres://")):
                self.use_postgres = True
        except ImportError:
            pass
        
        logger.info(f"Initialized with {'PostgreSQL' if self.use_postgres else 'SQLite'}")

    def get_connection(self):
        if self.use_postgres:
            parsed = urlparse(DATABASE_URL)
            import psycopg2 # type: ignore
            return psycopg2.connect(
                dbname=parsed.path.lstrip("/"),
                user=parsed.username,
                password=parsed.password,
                host=parsed.hostname,
                port=parsed.port or 5432,
            )
        return sqlite3.connect(SQLITE_DB_PATH)

    def init_db(self):
        conn = self.get_connection()
        c = conn.cursor()
        try:
            # Users Table
            c.execute('''CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY, 
                password TEXT NOT NULL, 
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # User Data Table
            col_type = "DOUBLE PRECISION" if self.use_postgres else "REAL"
            c.execute(f'''CREATE TABLE IF NOT EXISTS user_metrics (
                username TEXT PRIMARY KEY, 
                health_score {col_type}, 
                waste_data TEXT, 
                goal_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            conn.commit()
        finally:
            conn.close()

db = DatabaseManager()
db.init_db()

# --- ML MODELS ---
class AIModelSuite:
    def __init__(self):
        self.health = HealthModel()
        self.waste = WasteModel()
        self.goal = GoalModel()
        self.cluster = ClusterModel()
    
    def train_all(self):
        logger.info("Retraining FinSight AI models...")
        self.health.train()
        self.waste.train()
        self.goal.train()
        self.cluster.train()
        logger.info("Model training sequence complete.")

ai = AIModelSuite()

# --- LIFESPAN ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("TRAIN_MODELS_ON_STARTUP", "true").lower() == "true":
        ai.train_all()
    logger.info("FinSight AI Engine is 100% Operational.")
    yield

app = FastAPI(
    title="FinSight AI",
    description="Institutional-grade financial intelligence prediction engine.",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS ---
class SignupRequest(BaseModel):
    username: str
    password: str
    full_name: str

class IntelligenceRequest(BaseModel):
    behavioral_scores: dict
    context: Optional[dict] = {}

# --- UTILS ---
def send_welcome_alert(email: str, name: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"WELCOME_EMAIL_SIMULATOR | To: {email} | Time: {timestamp}")
    print(f"\n[SECURITY ALERT] Welcome to FinSight AI, {name}. Your encrypted workspace is now ready.\n", flush=True)

# --- ROUTES ---

@app.get("/status")
async def get_status():
    return {
        "status": "viva-ready",
        "timestamp": datetime.now().isoformat(),
        "database": "postgresql" if db.use_postgres else "sqlite",
        "models": ["health", "waste", "goal", "cluster"],
        "version": "3.0.0"
    }

@app.post("/auth/signup")
async def signup(user: SignupRequest, background_tasks: BackgroundTasks):
    conn = db.get_connection()
    c = conn.cursor()
    try:
        hashed = get_password_hash(user.password)
        query = "INSERT INTO users (username, password, full_name) VALUES (%s, %s, %s)" if db.use_postgres else \
                "INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)"
        c.execute(query, (user.username, hashed, user.full_name))
        conn.commit()
        background_tasks.add_task(send_welcome_alert, user.username, user.full_name)
        return {"msg": "Identity Provisioned"}
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(400, "Identity already exists")
        raise HTTPException(500, "Provisioning failed")
    finally:
        conn.close()

@app.post("/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = db.get_connection()
    c = conn.cursor()
    try:
        query = "SELECT password FROM users WHERE username = %s" if db.use_postgres else \
                "SELECT password FROM users WHERE username = ?"
        c.execute(query, (form_data.username,))
        row = c.fetchone()
        if not row or not verify_password(form_data.password, row[0]):
            raise HTTPException(401, "Invalid credentials")
        
        token = create_access_token({"sub": form_data.username})
        return {"access_token": token, "token_type": "bearer"}
    finally:
        conn.close()

@app.get("/user/me")
async def get_me(username: str = Depends(get_current_user)):
    return {"username": username}

@app.post("/predict/health")
async def predict_health(req: IntelligenceRequest, user=Depends(get_current_user)):
    try:
        scores = req.behavioral_scores
        ctx = req.context or {}
        emi = ctx.get("emi_burden", "light")
        income = ctx.get("income_bracket", "50k-100k")
        res = ai.health.predict(scores, emi, income)
        res["explanation"] = explain_health(res)
        return res
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/waste")
async def predict_waste(req: dict, user=Depends(get_current_user)):
    # req expects {"subscriptions": [...]}
    try:
        subs = req.get("subscriptions", [])
        res = ai.waste.predict_bulk(subs)
        res["explanation"] = explain_waste(res)
        return res
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/goal")
async def predict_goal(req: dict, user=Depends(get_current_user)):
    try:
        res = ai.goal.predict(
            behavioral_scores=req.get("behavioral_scores", {}),
            goal_description=req.get("goal_description", "Wealth Building"),
            goal_target=req.get("target_amount", 50000),
            saved_so_far=req.get("saved_so_far", 0),
            monthly_savings=req.get("monthly_savings", 1000),
            months_remaining=req.get("timeline_months", 24)
        )
        res["explanation"] = explain_goal(res)
        return res
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/behavioral-scores")
async def get_scores(req: dict, user=Depends(get_current_user)):
    try:
        return score_behavioral_answers(req.get("answers", {}))
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
