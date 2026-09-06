"""WoundCare+ clinical decision-support API.

The API deliberately treats the model as a decision-support component.  A
licensed nurse must review and save every assessment; the model never writes
an EMR by itself.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import bcrypt
import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from ultralytics import YOLO

from woundcare_inference import CascadeConfig, classify_image, infer_segmentation_cascade


ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("WOUNDCARE_DB_PATH", str(ROOT_DIR / "healthcare.db")))
KEY_PATH = Path(os.getenv("WOUNDCARE_KEY_PATH", str(ROOT_DIR / "secret.key")))
APP_ENV = os.getenv("WOUNDCARE_ENV", "development").lower()
SESSION_TTL_HOURS = int(os.getenv("WOUNDCARE_SESSION_TTL_HOURS", "8"))
MAX_UPLOAD_BYTES = int(os.getenv("WOUNDCARE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_ROLES = {"admin", "head_nurse", "nurse"}
PROFESSOR_PREVIEW = os.getenv("WOUNDCARE_PROFESSOR_PREVIEW", "false").strip().lower() in {"1", "true", "yes"}
LOGIN_MAX_ATTEMPTS = max(1, int(os.getenv("WOUNDCARE_LOGIN_MAX_ATTEMPTS", "8")))
LOGIN_WINDOW_SECONDS = max(60, int(os.getenv("WOUNDCARE_LOGIN_WINDOW_SECONDS", "900")))
_login_attempts: dict[str, list[float]] = {}


def _load_fernet() -> Fernet:
    configured = os.getenv("WOUNDCARE_FERNET_KEY")
    if configured:
        return Fernet(configured.encode())
    if not KEY_PATH.exists():
        if APP_ENV == "production":
            raise RuntimeError("WOUNDCARE_FERNET_KEY must be configured in production")
        KEY_PATH.write_bytes(Fernet.generate_key())
    return Fernet(KEY_PATH.read_bytes().strip())


cipher_suite = _load_fernet()


def encrypt_data(value: Optional[str]) -> str:
    if not value:
        return ""
    return cipher_suite.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_data(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return cipher_suite.decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeError):
        # Existing development databases contain a few legacy plaintext fields.
        # Keep them readable during migration; all new writes are encrypted.
        return value


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 15000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:
    with db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'head_nurse', 'nurse')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room TEXT NOT NULL,
                name TEXT NOT NULL,
                gender TEXT,
                age TEXT,
                admission_date TEXT,
                medical_history TEXT,
                allergies TEXT,
                assigned_nurse TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'green',
                last_assessment TEXT
            );
            CREATE TABLE IF NOT EXISTS emr_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                nurse_name TEXT NOT NULL,
                shift TEXT NOT NULL,
                wound_image TEXT,
                ai_analysis TEXT,
                treatment TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'green',
                review_status TEXT NOT NULL DEFAULT 'reviewed',
                human_review_confirmed INTEGER NOT NULL DEFAULT 0,
                human_reviewed_at TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nurse_name TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_hash TEXT UNIQUE NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,
                last_seen_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS rag_guidance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                wound_classes TEXT NOT NULL DEFAULT '',
                recommendation TEXT NOT NULL,
                rationale TEXT NOT NULL DEFAULT '',
                precautions TEXT NOT NULL DEFAULT '',
                evidence_level TEXT NOT NULL DEFAULT 'clinical_consensus',
                source TEXT NOT NULL DEFAULT 'head_nurse_review',
                created_by INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE RESTRICT
            );
            CREATE TABLE IF NOT EXISTS professor_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                task TEXT NOT NULL,
                usability_score INTEGER NOT NULL CHECK(usability_score BETWEEN 1 AND 5),
                clinical_clarity_score INTEGER NOT NULL CHECK(clinical_clarity_score BETWEEN 1 AND 5),
                comments TEXT NOT NULL,
                page TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_emr_patient_time ON emr_records(patient_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_rag_active ON rag_guidance(is_active, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON professor_feedback(created_at DESC);
            """
        )
        # Non-destructive migrations for databases created by the earlier demo.
        _ensure_column(connection, "users", "is_active", "INTEGER NOT NULL DEFAULT 1")
        # SQLite only permits constant defaults when adding a column to an
        # existing table. Add the legacy column without a default, then backfill
        # it inside the same transaction.
        _ensure_column(connection, "users", "created_at", "TEXT")
        connection.execute("UPDATE users SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)")
        _ensure_column(connection, "emr_records", "review_status", "TEXT NOT NULL DEFAULT 'reviewed'")
        _ensure_column(connection, "patients", "assigned_nurse", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "emr_records", "human_review_confirmed", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "emr_records", "human_reviewed_at", "TEXT")

        # Never create a publicly-known password.  A fresh deployment must set
        # WOUNDCARE_BOOTSTRAP_PASSWORD explicitly before the first login.
        user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        bootstrap_password = os.getenv("WOUNDCARE_BOOTSTRAP_PASSWORD")
        if user_count == 0 and bootstrap_password:
            password_hash = bcrypt.hashpw(bootstrap_password.encode(), bcrypt.gensalt()).decode()
            connection.execute(
                "INSERT INTO users(username, password, role) VALUES (?, ?, 'admin')",
                ("admin", password_hash),
            )

        if PROFESSOR_PREVIEW:
            preview_patient_count = connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            if preview_patient_count == 0:
                preview_patients = (
                    ("DEMO-01", "示範個案 A", "未提供", "45", "2026-08-18", "教授預覽資料；非真實病人。", "無", "green"),
                    ("DEMO-02", "示範個案 B", "未提供", "62", "2026-08-18", "教授預覽資料；非真實病人。", "無", "yellow"),
                )
                connection.executemany(
                    """
                    INSERT INTO patients(room, name, gender, age, admission_date, medical_history, allergies, assigned_nurse, status, last_assessment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, '尚無評估')
                    """,
                    [tuple(encrypt_data(value) for value in patient[:7]) + (patient[7],) for patient in preview_patients],
                )


init_db()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def audit_event(user: Optional[dict[str, Any]], action: str, resource: str = "", metadata: str = "") -> None:
    with db() as connection:
        connection.execute(
            "INSERT INTO audit_log(user_id, username, action, resource, metadata) VALUES (?, ?, ?, ?, ?)",
            (user.get("id") if user else None, user.get("username") if user else None, action, resource, metadata[:2000]),
        )


def _session_user(token: str) -> Optional[dict[str, Any]]:
    now = iso_now()
    with db() as connection:
        row = connection.execute(
            """
            SELECT u.id, u.username, u.role
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.is_active = 1
            """,
            (hash_token(token), now),
        ).fetchone()
        if not row:
            return None
        connection.execute("UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?", (now, hash_token(token)))
        return dict(row)


def current_user(authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登入工作階段已失效")
    token = authorization.split(" ", 1)[1].strip()
    user = _session_user(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登入工作階段已失效")
    return user


def require_roles(*roles: str):
    def dependency(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="目前帳號沒有此操作權限")
        return user

    return dependency


def enforce_login_rate_limit(request: Request) -> str:
    """Apply a small in-process backstop; the reverse proxy remains the primary limit."""
    client_id = request.client.host if request.client else "unknown"
    now = time.monotonic()
    attempts = [attempt for attempt in _login_attempts.get(client_id, []) if now - attempt < LOGIN_WINDOW_SECONDS]
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="登入嘗試過於頻繁，請稍後再試")
    _login_attempts[client_id] = attempts
    return client_id


def register_failed_login(client_id: str) -> None:
    _login_attempts.setdefault(client_id, []).append(time.monotonic())


def clear_login_attempts(client_id: str) -> None:
    _login_attempts.pop(client_id, None)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; font-src 'self' data:"
        )
        if APP_ENV == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


allowed_origins = [origin.strip() for origin in os.getenv("WOUNDCARE_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",") if origin.strip()]
allowed_hosts = [host.strip() for host in os.getenv("WOUNDCARE_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if host.strip()]

app = FastAPI(
    title="WoundCare+ Clinical API",
    version="2.0.0",
    docs_url="/docs" if APP_ENV != "production" else None,
    redoc_url=None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=128)
    role: str = Field(pattern=r"^(admin|head_nurse|nurse)$")


class UserLogin(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class PatientCreate(BaseModel):
    room: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=80)
    gender: str = Field(max_length=20)
    age: str = Field(max_length=8)
    admission_date: str = Field(max_length=20)
    medical_history: str = Field(default="", max_length=2000)
    allergies: str = Field(default="", max_length=1000)
    assigned_nurse: str = Field(default="", max_length=32, pattern=r"^[A-Za-z0-9_.-]*$")


class EMRRecordCreate(BaseModel):
    shift: str = Field(min_length=1, max_length=20)
    wound_image: Optional[str] = Field(default=None, max_length=14_000_000)
    ai_analysis: Optional[str] = Field(default=None, max_length=20_000)
    treatment: str = Field(min_length=1, max_length=5000)
    status: str = Field(pattern=r"^(red|yellow|green)$")
    review_status: str = Field(default="needs_review", pattern=r"^(reviewed|needs_review)$")
    human_review_confirmed: bool = False


class ModelPathUpdate(BaseModel):
    model_path: str = Field(min_length=1, max_length=300)


class RAGGuidanceCreate(BaseModel):
    title: str = Field(min_length=3, max_length=160)
    wound_classes: str = Field(default="", max_length=500)
    recommendation: str = Field(min_length=10, max_length=5000)
    rationale: str = Field(default="", max_length=3000)
    precautions: str = Field(default="", max_length=3000)
    evidence_level: str = Field(default="clinical_consensus", max_length=80)
    source: str = Field(default="head_nurse_review", max_length=160)
    source_emr_id: Optional[int] = Field(default=None, ge=1)


class RAGGuidanceStatusUpdate(BaseModel):
    is_active: bool


class ProfessorFeedbackCreate(BaseModel):
    task: str = Field(default="整體教授測試", min_length=3, max_length=160)
    usability_score: int = Field(ge=1, le=5)
    clinical_clarity_score: int = Field(ge=1, le=5)
    comments: str = Field(min_length=3, max_length=3000)
    page: str = Field(default="", max_length=80)


def _load_model(path: Path) -> Optional[YOLO]:
    try:
        return YOLO(str(path)) if path.exists() else None
    except Exception:
        return None


DET_MODEL_PATH = Path(os.getenv("WOUNDCARE_DET_MODEL", str(ROOT_DIR / "Wound_AI_Detection_v2/R1_YOLO11m_82/weights/best.pt")))
CLS_MODEL_PATH = Path(os.getenv("WOUNDCARE_CLS_MODEL", str(ROOT_DIR / "runs/classify/wound_classifier_v32/weights/best.pt")))
SEG_MODEL_PATH = Path(os.getenv("WOUNDCARE_SEG_MODEL", str(ROOT_DIR / "experiments/results/segmentation/YOLO11m_WSNet_seg_aug_v5_20260818/weights/best.pt")))
try:
    SEG_CONFIDENCE = float(os.getenv("WOUNDCARE_SEG_CONFIDENCE", "0.10"))
except ValueError:
    SEG_CONFIDENCE = 0.10
try:
    CLS_CONFIDENCE = float(os.getenv("WOUNDCARE_CLS_CONFIDENCE", "0.60"))
except ValueError:
    CLS_CONFIDENCE = 0.60
try:
    SEG_IMGSZ = max(32, int(os.getenv("WOUNDCARE_SEG_IMGSZ", "384")))
except ValueError:
    SEG_IMGSZ = 384
SEG_DEVICE = os.getenv("WOUNDCARE_SEG_DEVICE", "cpu")
CLASSIFICATION_SOURCE = os.getenv("WOUNDCARE_CLASSIFICATION_SOURCE", "full_image").strip().lower()
if CLASSIFICATION_SOURCE not in {"full_image", "segmentation_crop"}:
    CLASSIFICATION_SOURCE = "full_image"
det_model = _load_model(DET_MODEL_PATH)
cls_model = _load_model(CLS_MODEL_PATH)
seg_model = _load_model(SEG_MODEL_PATH)
CASCADE_CONFIG = CascadeConfig(
    segmenter_confidence=SEG_CONFIDENCE,
    classifier_confidence=CLS_CONFIDENCE,
    segmenter_imgsz=SEG_IMGSZ,
    classifier_imgsz=224,
    device=SEG_DEVICE,
    classification_source=CLASSIFICATION_SOURCE,
)


def _safe_model_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    candidate = candidate.resolve()
    allowed_roots = [ROOT_DIR / "Wound_AI_Detection_v2", ROOT_DIR / "runs", ROOT_DIR / "experiments"]
    if candidate.suffix.lower() != ".pt" or not any(candidate == root.resolve() or root.resolve() in candidate.parents for root in allowed_roots):
        raise HTTPException(status_code=400, detail="模型路徑不在允許的模型目錄")
    return candidate


def model_identity(path: Path) -> dict[str, Optional[str]]:
    if not path.exists():
        return {"file": path.name, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "sha256": digest.hexdigest()}


@app.get("/healthz")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/health")
def health(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return {
        "status": "ok",
        "professor_preview": PROFESSOR_PREVIEW,
        "detection_model_loaded": det_model is not None,
        "segmentation_model_loaded": seg_model is not None,
        "classification_model_loaded": cls_model is not None,
        "inference_mode": "full_image_primary_with_roi" if seg_model is not None and CLASSIFICATION_SOURCE == "full_image" else ("segmentation_crop_with_full_image_fallback" if seg_model is not None else "legacy_detection"),
        "classification_model": model_identity(CLS_MODEL_PATH),
        "classification_input_size": CASCADE_CONFIG.classifier_imgsz,
    }


@app.post("/api/login")
def login_user(credentials: UserLogin, request: Request) -> dict[str, Any]:
    client_id = enforce_login_rate_limit(request)
    with db() as connection:
        row = connection.execute("SELECT id, username, password, role FROM users WHERE username = ? AND is_active = 1", (credentials.username,)).fetchone()
    valid = False
    if row:
        try:
            valid = bcrypt.checkpw(credentials.password.encode(), row["password"].encode())
        except ValueError:
            valid = hmac.compare_digest(hashlib.sha256(credentials.password.encode()).hexdigest(), row["password"])
    if not valid:
        register_failed_login(client_id)
        audit_event(None, "login_failed", "auth", credentials.username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="帳號或密碼不正確")

    raw_token = secrets.token_urlsafe(36)
    now = utc_now()
    expires = now + timedelta(hours=SESSION_TTL_HOURS)
    with db() as connection:
        connection.execute(
            "INSERT INTO sessions(token_hash, user_id, created_at, expires_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
            (hash_token(raw_token), row["id"], now.isoformat(), expires.isoformat(), now.isoformat()),
        )
    user = {"id": row["id"], "username": row["username"], "role": row["role"]}
    clear_login_attempts(client_id)
    audit_event(user, "login_success", "auth")
    return {"message": "登入成功", "token": raw_token, "expires_at": expires.isoformat(), **user}


@app.post("/api/logout")
def logout_user(authorization: Optional[str] = Header(default=None), user: dict[str, Any] = Depends(current_user)) -> dict[str, str]:
    if authorization:
        with db() as connection:
            connection.execute("UPDATE sessions SET revoked_at = ? WHERE token_hash = ?", (iso_now(), hash_token(authorization.split(" ", 1)[1])))
    audit_event(user, "logout", "auth")
    return {"message": "已安全登出"}


@app.post("/api/register")
def register_user(user: UserRegister, actor: dict[str, Any] = Depends(require_roles("admin"))) -> dict[str, str]:
    password_hash = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    try:
        with db() as connection:
            connection.execute("INSERT INTO users(username, password, role) VALUES (?, ?, ?)", (user.username, password_hash, user.role))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="帳號已存在")
    audit_event(actor, "user_created", user.username, user.role)
    return {"message": "帳號已建立"}


@app.post("/api/engineer/model/load")
def load_model(config: ModelPathUpdate, actor: dict[str, Any] = Depends(require_roles("admin"))) -> dict[str, str]:
    global det_model, DET_MODEL_PATH
    path = _safe_model_path(config.model_path)
    loaded = _load_model(path)
    if loaded is None:
        raise HTTPException(status_code=400, detail="模型檔不存在或無法載入")
    det_model = loaded
    DET_MODEL_PATH = path
    audit_event(actor, "model_loaded", str(path))
    return {"message": "偵測模型已載入", "path": str(path)}


@app.get("/api/engineer/model/status")
def model_status(actor: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> dict[str, Any]:
    return {
        "detection": {"status": "loaded" if det_model else "unavailable", **model_identity(DET_MODEL_PATH)},
        "segmentation": {"status": "loaded" if seg_model else "unavailable", **model_identity(SEG_MODEL_PATH), "device": SEG_DEVICE},
        "classification": {"status": "loaded" if cls_model else "unavailable", **model_identity(CLS_MODEL_PATH)},
        "cascade": {"segmenter_confidence": SEG_CONFIDENCE, "classifier_confidence": CLS_CONFIDENCE, "segmenter_imgsz": SEG_IMGSZ, "classification_source": CLASSIFICATION_SOURCE},
    }


def _clinical_advice(classes: list[str], count: int) -> str:
    if not classes:
        return "未偵測到可信的傷口區域。請重新拍攝：保持光線均勻、鏡頭垂直，並讓傷口完整位於畫面內。"
    urgent = "Stab_wound" in classes or "Burns" in classes
    prefix = "請立即依院內流程通知護理長覆核。" if urgent else "請依院內傷口照護流程完成清潔、覆蓋與追蹤。"
    return f"模型標記 {count} 個候選區域（{', '.join(classes)}）。{prefix} AI 結果僅供決策支援，不能取代臨床評估。"


def _rag_tokens(value: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    # Keep CJK runs as a token and split Latin/number words. This is a small,
    # dependency-free lexical retriever suitable for the local knowledge base.
    return {token for token in normalized.split() if len(token) > 1}


def retrieve_rag_guidance(query: str, limit: int = 5) -> list[dict[str, Any]]:
    query_tokens = _rag_tokens(query)
    with db() as connection:
        rows = connection.execute(
            """
            SELECT id, title, wound_classes, recommendation, rationale, precautions,
                   evidence_level, source, is_active, created_at, updated_at
            FROM rag_guidance
            WHERE is_active = 1
            ORDER BY updated_at DESC, id DESC
            LIMIT 200
            """
        ).fetchall()
    scored: list[tuple[int, sqlite3.Row]] = []
    for row in rows:
        searchable = " ".join((row["title"], row["wound_classes"], row["recommendation"], row["precautions"]))
        item_tokens = _rag_tokens(searchable)
        score = len(query_tokens & item_tokens) if query_tokens else 0
        if score > 0 or not query_tokens:
            scored.append((score, row))
    scored.sort(key=lambda pair: (pair[0], pair[1]["updated_at"], pair[1]["id"]), reverse=True)
    return [
        {
            "id": row["id"],
            "title": row["title"],
            "wound_classes": row["wound_classes"],
            "recommendation": row["recommendation"],
            "rationale": row["rationale"],
            "precautions": row["precautions"],
            "evidence_level": row["evidence_level"],
            "source": row["source"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "match_score": score,
        }
        for score, row in scored[: max(1, min(limit, 20))]
    ]


@app.post("/api/predict")
async def predict_wound(file: UploadFile = File(...), user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="僅接受 JPG、PNG 或 WebP 圖片")
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="圖片大小不可超過 10 MB")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="圖片無法解析")
    if cls_model is None or (seg_model is None and det_model is None):
        raise HTTPException(status_code=503, detail="AI 模型尚未就緒，請稍後再試")

    try:
        annotated = image.copy()
        detections: list[dict[str, Any]] = []
        height, width = image.shape[:2]
        inference_mode = "legacy_detection"
        fallback_used = False
        segment_confidence = None
        class_confidence = None
        crop_box = None
        low_confidence = False

        if seg_model is not None:
            try:
                cascade = infer_segmentation_cascade(image, seg_model, cls_model, CASCADE_CONFIG)
            except Exception:
                # A segmentation failure must not make the clinical API
                # unavailable.  Classify the original image and require review.
                fallback_label, fallback_confidence = classify_image(cls_model, image, CASCADE_CONFIG)
                cascade = {
                    "mode": "full_image_fallback",
                    "fallback_used": True,
                    "detected_boxes": 0,
                    "segment_instances": 0,
                    "segment_confidence": None,
                    "crop_box": None,
                    "label": fallback_label,
                    "class_confidence": fallback_confidence,
                    "low_confidence": fallback_label is None or fallback_confidence is None or fallback_confidence < CLS_CONFIDENCE,
                }
            inference_mode = cascade["mode"]
            fallback_used = bool(cascade["fallback_used"])
            segment_confidence = cascade.get("segment_confidence")
            class_confidence = cascade.get("class_confidence")
            crop_box = cascade.get("crop_box")
            low_confidence = bool(cascade.get("low_confidence"))
            label = cascade.get("label")
            if inference_mode in {"segmentation_crop", "full_image_primary"} and crop_box and label:
                detections.append({
                    "box": crop_box,
                    "confidence": segment_confidence,
                    "label": label,
                    "class_confidence": class_confidence,
                    "source": "segmentation_crop" if inference_mode == "segmentation_crop" else "segmentation_roi_only",
                })
                x1, y1, x2, y2 = [crop_box[key] for key in ("x1", "y1", "x2", "y2")]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (24, 160, 147), 3)
                cv2.putText(annotated, f"{label} {float(class_confidence or 0):.0%}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 160, 147), 2)
            elif label:
                cv2.putText(annotated, f"原圖分類：{label} {float(class_confidence or 0):.0%}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 160, 147), 2)
        else:
            result = det_model(image, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                crop = image[y1:y2, x1:x2]
                label = "Wound"
                box_class_confidence = None
                if crop.size:
                    label, box_class_confidence = classify_image(cls_model, crop, CASCADE_CONFIG)
                detections.append({"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "confidence": round(confidence, 4), "label": label or "Wound", "class_confidence": round(box_class_confidence, 4) if box_class_confidence is not None else None, "source": "legacy_detection"})
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (24, 160, 147), 3)
                cv2.putText(annotated, f"{label or 'Wound'} {confidence:.0%}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 160, 147), 2)

        classes = sorted({item["label"] for item in detections if item.get("label")})
        if not classes and seg_model is not None:
            fallback_label = cascade.get("label")
            if fallback_label:
                classes = [fallback_label]
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            raise RuntimeError("annotated image encoding failed")
        audit_event(user, "ai_prediction", "wound_image", f"mode={inference_mode};boxes={len(detections)};fallback={fallback_used}")
        rag_guidance = retrieve_rag_guidance(" ".join(classes), limit=3)
        advice = _clinical_advice(classes, len(detections))
        if inference_mode == "full_image_fallback":
            advice = "偵測 ROI 信心不足，已改用原圖分類；請由護理師確認傷口位置與分類後再保存。" + (f" 模型分類：{classes[0]}。" if classes else "")
        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii"),
            "detected_boxes": len(detections),
            "detections": detections,
            "classes": classes,
            "inference_mode": inference_mode,
            "fallback_used": fallback_used,
            "segment_confidence": segment_confidence,
            "class_confidence": class_confidence,
            "crop_box": crop_box,
            "low_confidence": low_confidence,
            "llm_advice": advice,
            "rag_guidance": rag_guidance,
            "requires_human_review": True,
            "model": {
                **model_identity(CLS_MODEL_PATH),
                "input_size": CASCADE_CONFIG.classifier_imgsz,
                "inference_mode": inference_mode,
            },
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="影像分析失敗，請確認圖片後重試")


def _patient_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {"id": row["id"], "room": decrypt_data(row["room"]), "name": decrypt_data(row["name"]), "gender": decrypt_data(row["gender"]), "age": decrypt_data(row["age"]), "admission_date": decrypt_data(row["admission_date"]), "medical_history": decrypt_data(row["medical_history"]), "allergies": decrypt_data(row["allergies"]), "assigned_nurse": row["assigned_nurse"], "status": row["status"], "last_assessment": row["last_assessment"]}


def require_patient_access(connection: sqlite3.Connection, patient_id: int, user: dict[str, Any]) -> sqlite3.Row:
    patient = connection.execute("SELECT id, name, assigned_nurse FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        raise HTTPException(status_code=404, detail="找不到病人")
    if user["role"] == "nurse" and patient["assigned_nurse"] != user["username"]:
        raise HTTPException(status_code=403, detail="此病人未指派給目前護理師")
    return patient


@app.get("/api/patients")
def get_patients(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    with db() as connection:
        if user["role"] == "nurse":
            rows = connection.execute("SELECT id, room, name, gender, age, admission_date, medical_history, allergies, assigned_nurse, status, last_assessment FROM patients WHERE assigned_nurse = ? ORDER BY id DESC", (user["username"],)).fetchall()
        else:
            rows = connection.execute("SELECT id, room, name, gender, age, admission_date, medical_history, allergies, assigned_nurse, status, last_assessment FROM patients ORDER BY id DESC").fetchall()
    return [_patient_payload(row) for row in rows]


@app.post("/api/patients")
def add_patient(patient: PatientCreate, actor: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> dict[str, Any]:
    with db() as connection:
        if patient.assigned_nurse:
            assignee = connection.execute("SELECT username, role, is_active FROM users WHERE username = ?", (patient.assigned_nurse,)).fetchone()
            if not assignee or assignee["role"] != "nurse" or not assignee["is_active"]:
                raise HTTPException(status_code=422, detail="指定的護理師帳號不存在或未啟用")
        cursor = connection.execute("INSERT INTO patients(room, name, gender, age, admission_date, medical_history, allergies, assigned_nurse, status, last_assessment) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'green', '尚無評估')", tuple(encrypt_data(value) for value in (patient.room, patient.name, patient.gender, patient.age, patient.admission_date, patient.medical_history, patient.allergies)) + (patient.assigned_nurse,))
        patient_id = cursor.lastrowid
    audit_event(actor, "patient_created", str(patient_id))
    return {"message": "病人資料已加密存檔", "id": patient_id}


def _emr_payload(row: sqlite3.Row, include_patient: bool = False) -> dict[str, Any]:
    payload = {"id": row["id"], "patient_id": row["patient_id"], "nurse_name": row["nurse_name"], "shift": row["shift"], "wound_image": decrypt_data(row["wound_image"]), "ai_analysis": decrypt_data(row["ai_analysis"]), "treatment": decrypt_data(row["treatment"]), "status": row["status"], "review_status": row["review_status"], "human_review_confirmed": bool(row["human_review_confirmed"]), "human_reviewed_at": row["human_reviewed_at"], "timestamp": row["timestamp"]}
    if include_patient:
        payload["patient_name"] = decrypt_data(row["patient_name"])
    return payload


@app.get("/api/patients/{patient_id}/emr")
def get_patient_emr(patient_id: int, user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    with db() as connection:
        require_patient_access(connection, patient_id, user)
        rows = connection.execute("SELECT id, patient_id, nurse_name, shift, wound_image, ai_analysis, treatment, status, review_status, human_review_confirmed, human_reviewed_at, timestamp FROM emr_records WHERE patient_id = ? ORDER BY timestamp DESC", (patient_id,)).fetchall()
    return [_emr_payload(row) for row in rows]


@app.post("/api/patients/{patient_id}/emr")
def add_patient_emr(patient_id: int, record: EMRRecordCreate, actor: dict[str, Any] = Depends(require_roles("admin", "head_nurse", "nurse"))) -> dict[str, Any]:
    if record.review_status == "reviewed" and not record.human_review_confirmed:
        raise HTTPException(status_code=422, detail="必須明確確認已完成人工覆核，才能標記為已覆核")
    with db() as connection:
        patient = require_patient_access(connection, patient_id, actor)
        reviewed_at = iso_now() if record.human_review_confirmed else None
        connection.execute("INSERT INTO emr_records(patient_id, nurse_name, shift, wound_image, ai_analysis, treatment, status, review_status, human_review_confirmed, human_reviewed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (patient_id, actor["username"], record.shift, encrypt_data(record.wound_image), encrypt_data(record.ai_analysis), encrypt_data(record.treatment), record.status, record.review_status, int(record.human_review_confirmed), reviewed_at))
        connection.execute("UPDATE patients SET status = ?, last_assessment = ? WHERE id = ?", (record.status, f"{record.shift}｜{datetime.now().strftime('%m/%d %H:%M')}", patient_id))
        connection.execute("INSERT INTO reports(nurse_name, patient_name, action) VALUES (?, ?, ?)", (encrypt_data(actor["username"]), patient["name"], f"完成傷口評估（{record.status}）"))
        emr_id = connection.execute("SELECT last_insert_rowid()").fetchone()[0]
    audit_event(actor, "emr_created", str(emr_id), f"patient_id={patient_id};status={record.status}")
    return {"message": "評估已儲存，並等待護理團隊追蹤", "id": emr_id}


@app.get("/api/emr/all")
def get_all_emrs(user: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute("SELECT e.id, e.patient_id, e.nurse_name, e.shift, e.wound_image, e.ai_analysis, e.treatment, e.status, e.review_status, e.human_review_confirmed, e.human_reviewed_at, e.timestamp, p.name AS patient_name FROM emr_records e JOIN patients p ON e.patient_id = p.id ORDER BY e.timestamp DESC").fetchall()
    return [_emr_payload(row, include_patient=True) for row in rows]


@app.get("/api/admin/users")
def get_users(user: dict[str, Any] = Depends(require_roles("admin"))) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute("SELECT id, username, role, is_active, created_at FROM users ORDER BY id DESC").fetchall()
    return [dict(row) for row in rows]


@app.post("/api/admin/users")
def admin_create_user(user: UserRegister, actor: dict[str, Any] = Depends(require_roles("admin"))) -> dict[str, str]:
    return register_user(user, actor)


@app.delete("/api/admin/users/{username}")
def delete_user(username: str, actor: dict[str, Any] = Depends(require_roles("admin"))) -> dict[str, str]:
    if username == actor["username"] or username == "admin":
        raise HTTPException(status_code=400, detail="不可刪除目前登入帳號或系統保護帳號")
    with db() as connection:
        connection.execute("UPDATE users SET is_active = 0 WHERE username = ?", (username,))
    audit_event(actor, "user_deactivated", username)
    return {"message": "帳號已停用"}


@app.get("/api/audit")
def get_audit_log(user: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute("SELECT id, username, action, resource, metadata, created_at FROM audit_log ORDER BY id DESC LIMIT 200").fetchall()
    return [dict(row) for row in rows]


@app.post("/api/professor-feedback")
def create_professor_feedback(feedback: ProfessorFeedbackCreate, actor: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    with db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO professor_feedback(user_id, username, task, usability_score, clinical_clarity_score, comments, page)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (actor["id"], actor["username"], feedback.task.strip(), feedback.usability_score, feedback.clinical_clarity_score, feedback.comments.strip(), feedback.page.strip()),
        )
        feedback_id = cursor.lastrowid
    audit_event(actor, "professor_feedback_submitted", str(feedback_id), f"page={feedback.page.strip()[:80]}")
    return {"message": "回饋已送出，謝謝您的協助", "id": feedback_id}


@app.get("/api/professor-feedback")
def get_professor_feedback(user: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> list[dict[str, Any]]:
    with db() as connection:
        rows = connection.execute(
            "SELECT id, username, task, usability_score, clinical_clarity_score, comments, page, created_at FROM professor_feedback ORDER BY id DESC LIMIT 200"
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/rag/guidance")
def get_rag_guidance(query: str = "", limit: int = 5, user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    return retrieve_rag_guidance(query[:500], limit)


@app.post("/api/rag/guidance")
def create_rag_guidance(guidance: RAGGuidanceCreate, actor: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> dict[str, Any]:
    with db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO rag_guidance(title, wound_classes, recommendation, rationale,
                                     precautions, evidence_level, source, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (guidance.title.strip(), guidance.wound_classes.strip(), guidance.recommendation.strip(), guidance.rationale.strip(), guidance.precautions.strip(), guidance.evidence_level.strip(), guidance.source.strip(), actor["id"]),
        )
        guidance_id = cursor.lastrowid
    metadata = f"guidance_id={guidance_id}"
    if guidance.source_emr_id:
        metadata += f";source_emr_id={guidance.source_emr_id}"
    audit_event(actor, "rag_guidance_created", "rag_guidance", metadata)
    return {"message": "護理長建議已寫入 RAG 知識庫", "id": guidance_id}


@app.patch("/api/rag/guidance/{guidance_id}")
def update_rag_guidance_status(guidance_id: int, update: RAGGuidanceStatusUpdate, actor: dict[str, Any] = Depends(require_roles("admin", "head_nurse"))) -> dict[str, str]:
    with db() as connection:
        changed = connection.execute("UPDATE rag_guidance SET is_active = ?, updated_at = ? WHERE id = ?", (int(update.is_active), iso_now(), guidance_id)).rowcount
    if not changed:
        raise HTTPException(status_code=404, detail="找不到 RAG 建議")
    audit_event(actor, "rag_guidance_status_changed", str(guidance_id), f"is_active={update.is_active}")
    return {"message": "RAG 建議狀態已更新"}


dist_dir = ROOT_DIR / "wound_nurse_app" / "dist"
assets_dir = dist_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/{catchall:path}")
def serve_react_app(catchall: str):
    index = dist_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"message": "Frontend 尚未建置，請先執行 wound_nurse_app 的 npm run build"}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("WOUNDCARE_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
