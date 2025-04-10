"""
API routes for model management.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from ..utils.config import ModelsConfig, load_models_config
from ..models.model_manager import ModelManager
from .schemas import (
    ModelInfo,
    LoadedModelInfo,
    LoadModelRequest,
    LoadModelResponse,
    UnloadModelRequest,
    UnloadModelResponse,
    SystemInfoResponse,
)


def create_models_router(model_manager: ModelManager, models_config: ModelsConfig) -> APIRouter:
    """Create FastAPI router for model management endpoints."""
    router = APIRouter(prefix="/models", tags=["Models"])
    
    @router.get("/available", response_model=List[ModelInfo], summary="List available models")
    async def list_available_models(
        type: Optional[str] = Query(None, description="Filter by model type"),
        family: Optional[str] = Query(None, description="Filter by model family"),
        tag: Optional[str] = Query(None, description="Filter by model tag"),
    ):
        """
        List all available models with optional filtering.
        
        ## Description
        Returns a list of models that can be loaded, with information about each model.
        
        ## Parameters
        - **type**: (optional) Filter by model type (e.g., causal_lm, embedding)
        - **family**: (optional) Filter by model family (e.g., llama, mistral)
        - **tag**: (optional) Filter by model tag (e.g., base, instruct, code)
        
        ## Returns
        A list of model information objects.
        
        ## Example
        ```python
        import requests
        
        # List all available models
        response = requests.get("http://localhost:8000/api/v1/models/available")
        models = response.json()
        
        # Filter for embedding models
        response = requests.get("http://localhost:8000/api/v1/models/available?type=embedding")
        embedding_models = response.json()
        ```
        """
        models = models_config.models
        
        # Apply filters
        if type:
            models = [m for m in models if m.type == type]
        if family:
            models = [m for m in models if m.family == family]
        if tag:
            models = [m for m in models if tag in m.tags]
        
        return models
    
    @router.get("/loaded", response_model=List[LoadedModelInfo], summary="List loaded models")
    async def list_loaded_models():
        """
        List all currently loaded models.
        
        ## Description
        Returns information about all models that are currently loaded in memory,
        including usage statistics.
        
        ## Returns
        A list of loaded model information objects.
        
        ## Example
        ```python
        import requests
        
        # Get all loaded models
        response = requests.get("http://localhost:8000/api/v1/models/loaded")
        loaded_models = response.json()
        
        # Check if a specific model is loaded
        is_mistral_loaded = any(m["model_id"] == "mistral-7b" for m in loaded_models)
        ```
        """
        return model_manager.get_loaded_models()
    
    @router.post("/load", response_model=LoadModelResponse, summary="Load a model")
    async def load_model(request: LoadModelRequest):
        """
        Load a model into memory.
        
        ## Description
        Loads a model by ID with optional quantization and GPU settings.
        If the model is already loaded, returns information about the loaded instance.
        
        ## Request Body
        - **model_id**: ID of the model to load
        - **quantization**: (optional) Quantization method (none, 4-bit, 8-bit)
        - **use_gpu**: (optional) Whether to use GPU if available
        
        ## Returns
        Information about the loaded model.
        
        ## Example
        ```python
        import requests
        
        # Load a model with default settings
        response = requests.post(
            "http://localhost:8000/api/v1/models/load",
            json={"model_id": "mistral-7b"}
        )
        result = response.json()
        
        # Load with 4-bit quantization
        response = requests.post(
            "http://localhost:8000/api/v1/models/load",
            json={"model_id": "llama3-8b", "quantization": "4-bit"}
        )
        result = response.json()
        ```
        """
        try:
            model_instance = model_manager.load_model(
                model_id=request.model_id,
                quantization=request.quantization or "none",
                use_gpu=request.use_gpu,
            )
            
            return LoadModelResponse(
                success=True,
                model_id=model_instance.model_id,
                quantization=model_instance.quantization,
                device=model_instance.device,
                message=f"Model {request.model_id} loaded successfully on {model_instance.device}",
            )
        except Exception as e:
            logger.exception(f"Error loading model {request.model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}",
            )
    
    @router.post("/unload", response_model=UnloadModelResponse, summary="Unload a model")
    async def unload_model(request: UnloadModelRequest):
        """
        Unload a model from memory.
        
        ## Description
        Unloads a previously loaded model, freeing up memory resources.
        
        ## Request Body
        - **model_id**: ID of the model to unload
        - **quantization**: (optional) Quantization method of the loaded model
        
        ## Returns
        Status of the unload operation.
        
        ## Example
        ```python
        import requests
        
        # Unload a model
        response = requests.post(
            "http://localhost:8000/api/v1/models/unload",
            json={"model_id": "mistral-7b"}
        )
        result = response.json()
        ```
        """
        try:
            success = model_manager.unload_model(
                model_id=request.model_id,
                quantization=request.quantization or "none",
            )
            
            if success:
                return UnloadModelResponse(
                    success=True,
                    model_id=request.model_id,
                    message=f"Model {request.model_id} unloaded successfully",
                )
            else:
                return UnloadModelResponse(
                    success=False,
                    model_id=request.model_id,
                    message=f"Model {request.model_id} was not loaded",
                )
        except Exception as e:
            logger.exception(f"Error unloading model {request.model_id}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to unload model: {str(e)}",
            )
    
    @router.get("/system-info", response_model=SystemInfoResponse, summary="Get system information")
    async def get_system_info():
        """
        Get system information.
        
        ## Description
        Returns information about the system, including CPU, memory, and GPU usage,
        as well as information about loaded models.
        
        ## Returns
        System information object.
        
        ## Example
        ```python
        import requests
        
        # Get system information
        response = requests.get("http://localhost:8000/api/v1/models/system-info")
        system_info = response.json()
        
        # Check GPU availability
        if system_info["gpu"]["available"]:
            print(f"GPU available: {system_info['gpu']['devices'][0]['name']}")
        else:
            print("No GPU available")
        ```
        """
        return model_manager.get_system_info()
    
    return router