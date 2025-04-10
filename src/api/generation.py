"""
API routes for text generation and embedding.
"""
from typing import List, Optional
import hashlib
import json

from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from ..models.inference import Inference
from .schemas import (
    GenerateTextRequest,
    GenerateTextResponse,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
)


def create_generation_router(inference: Inference) -> APIRouter:
    """Create FastAPI router for generation endpoints."""
    router = APIRouter(prefix="/generation", tags=["Generation"])
    
    @router.post("/text", response_model=GenerateTextResponse, summary="Generate text")
    async def generate_text(request: GenerateTextRequest):
        """
        Generate text using a language model.
        
        ## Description
        Takes a prompt and generates text using the specified model.
        
        ## Request Body
        - **model_id**: ID of the model to use
        - **prompt**: Text prompt for generation
        - **quantization**: (optional) Quantization method (none, 4-bit, 8-bit)
        - **max_length**: (optional) Maximum length of generated text
        - **temperature**: (optional) Sampling temperature
        - **top_p**: (optional) Nucleus sampling parameter
        - **top_k**: (optional) Top-k sampling parameter
        - **repetition_penalty**: (optional) Penalty for repeating tokens
        - **num_return_sequences**: (optional) Number of sequences to return
        
        ## Returns
        The generated text and metadata.
        
        ## Example
        ```python
        import requests
        
        # Generate text with default parameters
        response = requests.post(
            "http://localhost:8000/api/v1/generation/text",
            json={
                "model_id": "mistral-7b",
                "prompt": "Write a short poem about artificial intelligence:"
            }
        )
        result = response.json()
        print(result["generated_text"])
        
        # Generate text with custom parameters
        response = requests.post(
            "http://localhost:8000/api/v1/generation/text",
            json={
                "model_id": "llama3-8b",
                "prompt": "List 5 ways to improve productivity:",
                "quantization": "4-bit",
                "temperature": 0.9,
                "max_length": 500
            }
        )
        result = response.json()
        ```
        """
        try:
            # Generate cache key if caching is enabled
            cache_key = None
            if inference.settings.cache_responses:
                cache_key = hashlib.md5(
                    json.dumps(request.model_dump(), sort_keys=True).encode()
                ).hexdigest()
            
            # Run inference
            result = inference.generate_text(
                model_id=request.model_id,
                prompt=request.prompt,
                quantization=request.quantization or "none",
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                num_return_sequences=request.num_return_sequences or 1,
                cache_key=cache_key,
            )
            
            return result
        except Exception as e:
            logger.exception(f"Error generating text: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Text generation failed: {str(e)}",
            )
    
    @router.post("/embeddings", response_model=GenerateEmbeddingsResponse, summary="Generate embeddings")
    async def generate_embeddings(request: GenerateEmbeddingsRequest):
        """
        Generate embeddings for texts.
        
        ## Description
        Generates vector embeddings for the provided texts using the specified model.
        
        ## Request Body
        - **model_id**: ID of the embedding model to use
        - **texts**: List of texts to generate embeddings for
        - **quantization**: (optional) Quantization method (none, 4-bit, 8-bit)
        - **pooling_method**: (optional) Pooling method (mean, cls)
        - **normalize**: (optional) Whether to normalize embeddings
        
        ## Returns
        The generated embeddings and metadata.
        
        ## Example
        ```python
        import requests
        
        # Generate embeddings for texts
        response = requests.post(
            "http://localhost:8000/api/v1/generation/embeddings",
            json={
                "model_id": "bge-small-en",
                "texts": ["Hello, world!", "This is a test."]
            }
        )
        result = response.json()
        
        # Access embeddings
        embeddings = result["embeddings"]
        ```
        """
        try:
            # Check if the request is valid
            if not request.texts:
                raise HTTPException(
                    status_code=400,
                    detail="No texts provided for embedding",
                )
            
            # Run inference
            result = inference.generate_embeddings(
                model_id=request.model_id,
                texts=request.texts,
                quantization=request.quantization or "none",
                pooling_method=request.pooling_method or "mean",
                normalize=request.normalize if request.normalize is not None else True,
            )
            
            return result
        except Exception as e:
            logger.exception(f"Error generating embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}",
            )
    
    @router.post("/clear-cache", summary="Clear response cache")
    async def clear_cache():
        """
        Clear the response cache.
        
        ## Description
        Clears any cached responses from previous generation requests.
        
        ## Returns
        Confirmation message.
        
        ## Example
        ```python
        import requests
        
        # Clear the cache
        response = requests.post("http://localhost:8000/api/v1/generation/clear-cache")
        print(response.json()["message"])
        ```
        """
        inference.clear_cache()
        return {"message": "Cache cleared successfully"}
    
    return router