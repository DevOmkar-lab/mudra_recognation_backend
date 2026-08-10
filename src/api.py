"""FastAPI interface for the trained Mudra recognition model."""
from __future__ import annotations

import hashlib
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pymongo.errors import DuplicateKeyError

from src.db import tokens_collection, users_collection
from src.services.recognition import RecognitionError, RecognitionService
from src.utils.config import load_config


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class RecognitionResponse(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float = Field(ge=0, le=1)
    matched: bool
    probabilities: dict[str, float]
    landmarks_detected: int | None = None

    model_config = {"populate_by_name": True}


class FeatureRequest(BaseModel):
    features: list[float]


class SignUpRequest(BaseModel):
    username: str
    password: str
    email: Optional[str]


class SignInRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    username: str
    email: Optional[str] = None


def _service(request: Request) -> RecognitionService:
    return request.app.state.recognition_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recognition_service = RecognitionService(load_config())
    yield
    app.state.recognition_service.close()


config = load_config()
app = FastAPI(title="Mudra Recognition API", version="1.0.0", lifespan=lifespan)
origins = config.get("api", {}).get("allowed_origins", ["http://localhost:3000", "http://127.0.0.1:3000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()


def get_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, hashed_password: str) -> bool:
    return get_password_hash(password) == hashed_password


def create_access_token() -> str:
    return secrets.token_urlsafe(32)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Optional[str]]:
    token = credentials.credentials
    token_doc = tokens_collection.find_one({"token": token})
    if not token_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_doc = users_collection.find_one({"email": token_doc["email"]})
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"username": user_doc["username"], "email": user_doc.get("email")}


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
@app.get("/api/healt", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(status="ok" if service.ready else "degraded", model_ready=service.ready)


@app.post("/api/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["auth"])
def signup(payload: SignUpRequest) -> UserResponse:
    if payload.email is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")
    password_hash = get_password_hash(payload.password)
    try:
        users_collection.insert_one(
            {
                "username": payload.username,
                "password_hash": password_hash,
                "email": payload.email,
            }
        )
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists",
        )
    return UserResponse(username=payload.username, email=payload.email)


@app.post("/api/auth/signin", response_model=TokenResponse, tags=["auth"])
def signin(payload: SignInRequest) -> TokenResponse:
    user = users_collection.find_one({"email": payload.email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token()
    tokens_collection.insert_one({"token": token, "email": payload.email})
    return TokenResponse(access_token=token)


@app.get("/api/auth/me", response_model=UserResponse, tags=["auth"])
def me(current_user: Dict[str, Optional[str]] = Depends(get_current_user)) -> UserResponse:
    return UserResponse(username=current_user["username"], email=current_user["email"])


@app.post("/api/v1/recognize", response_model=RecognitionResponse, response_model_exclude_none=True, tags=["recognition"])
async def recognize_image(request: Request, image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP hand image")]) -> dict[str, Any]:
    service = _service(request)
    if not service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=service.load_error or "Recognition model is unavailable")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload an image file")
    try:
        result = await run_in_threadpool(service.recognize_image, await image.read())
        return result
    except RecognitionError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    finally:
        await image.close()


@app.post("/api/v1/recognize/features", response_model=RecognitionResponse, response_model_exclude_none=True, tags=["recognition"])
async def recognize_features(request: Request, payload: FeatureRequest) -> dict[str, Any]:
    service = _service(request)
    if not service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=service.load_error or "Recognition model is unavailable")
    try:
        return await run_in_threadpool(service.recognize_features, payload.features)
    except RecognitionError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
