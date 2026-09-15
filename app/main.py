from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.routes import router

# Configure Logger
logger.add("logs/backend.log", rotation="10 MB", retention="10 days", level="INFO")

app = FastAPI(
    title="Ask My Documents API",
    description="A privacy-first local RAG platform using Ollama and ChromaDB.",
    version="1.0.0"
)

logger.info("Starting Ask My Documents API...")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catcher for all unhandled exceptions."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "An unexpected error occurred on the server.",
        },
    )

# Enable CORS for the local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to Ask My Documents API. Visit /docs for documentation."}

@app.get("/health")
async def health_check():
    """Verifies that the API is up and running."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "details": {
            "api": "online",
            "backend": "FastAPI"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
