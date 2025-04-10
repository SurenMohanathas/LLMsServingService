"""
Main FastAPI application.
"""
from typing import Dict, List, Any

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time

from ..utils.config import load_server_config, load_models_config, Config
from ..utils.logging import setup_logging, log_request, log_response
from ..models.model_manager import ModelManager
from ..models.inference import Inference
from .models import create_models_router
from .generation import create_generation_router


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    # Load configuration
    server_config = load_server_config()
    models_config = load_models_config()
    
    # Set up logging
    setup_logging(
        level=server_config.logging.level,
        log_file=server_config.logging.file,
        max_size_mb=server_config.logging.max_size,
        backup_count=server_config.logging.backup_count,
    )
    
    # Create model manager and inference
    model_manager = ModelManager(
        models_config=models_config,
        settings=server_config.models,
    )
    
    inference = Inference(
        model_manager=model_manager,
        settings=server_config.inference,
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="LLMsServingService API",
        description="API for serving local LLMs",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=f"{server_config.server.api_prefix}/openapi.json",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.security.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create routers
    models_router = create_models_router(model_manager, models_config)
    generation_router = create_generation_router(inference)
    
    # Add routers to app with prefix
    app.include_router(
        models_router,
        prefix=server_config.server.api_prefix,
    )
    app.include_router(
        generation_router,
        prefix=server_config.server.api_prefix,
    )
    
    # Add custom documentation routes
    @app.get(f"{server_config.server.api_prefix}/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=f"{server_config.server.api_prefix}/openapi.json",
            title=app.title + " - API Documentation",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )
    
    # Add middleware for request logging and timing
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # Start timer
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        if server_config.logging.log_payloads:
            log_response(route=request.url.path, status_code=response.status_code, duration_ms=duration_ms)
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        
        return response
    
    # Add error handlers
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(exc)}"},
        )
    
    # Add health check endpoint
    @app.get(f"{server_config.server.api_prefix}/health", tags=["Health"])
    async def health_check():
        """
        Check API health.
        
        ## Description
        Returns health status of the API server.
        
        ## Returns
        Status information.
        
        ## Example
        ```python
        import requests
        
        response = requests.get("http://localhost:8000/api/v1/health")
        status = response.json()
        print(f"API is {'up' if status['status'] == 'ok' else 'down'}")
        ```
        """
        return {
            "status": "ok",
            "version": "0.1.0",
            "models_loaded": len(model_manager.loaded_models),
        }
    
    return app