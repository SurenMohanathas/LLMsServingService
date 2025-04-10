"""
API request and response schemas.
"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, root_validator


class GenerateTextRequest(BaseModel):
    """Request schema for text generation."""
    model_id: str = Field(..., description="Model ID to use for generation")
    prompt: str = Field(..., description="Prompt for text generation")
    quantization: Optional[str] = Field("none", description="Quantization method (none, 4-bit, 8-bit)")
    max_length: Optional[int] = Field(None, description="Maximum length of generated text")
    temperature: Optional[float] = Field(None, description="Sampling temperature (higher: more creative, lower: more deterministic)")
    top_p: Optional[float] = Field(None, description="Nucleus sampling parameter (higher: more diversity)")
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter (higher: more diversity)")
    repetition_penalty: Optional[float] = Field(None, description="Penalty for repeating tokens (higher: less repetition)")
    num_return_sequences: Optional[int] = Field(1, description="Number of different sequences to return")


class GenerateTextResponse(BaseModel):
    """Response schema for text generation."""
    model_id: str = Field(..., description="Model ID used for generation")
    generated_text: Union[str, List[str]] = Field(..., description="Generated text output")
    elapsed_time: float = Field(..., description="Processing time in seconds")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")


class GenerateEmbeddingsRequest(BaseModel):
    """Request schema for embedding generation."""
    model_id: str = Field(..., description="Model ID to use for embeddings")
    texts: List[str] = Field(..., description="Texts to generate embeddings for")
    quantization: Optional[str] = Field("none", description="Quantization method (none, 4-bit, 8-bit)")
    pooling_method: Optional[str] = Field("mean", description="Pooling method (mean, cls)")
    normalize: Optional[bool] = Field(True, description="Whether to normalize embeddings")


class GenerateEmbeddingsResponse(BaseModel):
    """Response schema for embedding generation."""
    model_id: str = Field(..., description="Model ID used for embeddings")
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    dimensions: int = Field(..., description="Dimensions of embeddings")
    input_texts: int = Field(..., description="Number of input texts")
    total_tokens: int = Field(..., description="Total number of processed tokens")
    elapsed_time: float = Field(..., description="Processing time in seconds")


class ModelInfo(BaseModel):
    """Information about a model."""
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    huggingface_repo: str = Field(..., description="Hugging Face repository")
    type: str = Field(..., description="Model type (causal_lm, embedding)")
    family: str = Field(..., description="Model family")
    context_length: int = Field(..., description="Context length in tokens")
    ram_required: int = Field(..., description="RAM required in GB")
    gpu_required: int = Field(..., description="GPU memory required in GB")
    quantization: List[str] = Field(..., description="Supported quantization methods")
    tags: List[str] = Field(..., description="Model tags")
    description: str = Field(..., description="Model description")


class LoadedModelInfo(BaseModel):
    """Information about a loaded model."""
    model_id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Model name")
    quantization: str = Field(..., description="Current quantization method")
    device: str = Field(..., description="Device (cuda, cpu)")
    type: str = Field(..., description="Model type")
    family: str = Field(..., description="Model family")
    load_time: float = Field(..., description="Time when model was loaded")
    uptime_seconds: float = Field(..., description="Seconds since model was loaded")
    last_used: float = Field(..., description="Time when model was last used")
    idle_time_seconds: float = Field(..., description="Seconds since model was last used")
    requests_processed: int = Field(..., description="Number of requests processed")
    total_tokens_processed: int = Field(..., description="Total tokens processed")


class LoadModelRequest(BaseModel):
    """Request schema for loading a model."""
    model_id: str = Field(..., description="Model ID to load")
    quantization: Optional[str] = Field("none", description="Quantization method (none, 4-bit, 8-bit)")
    use_gpu: Optional[bool] = Field(None, description="Whether to use GPU (if available)")


class LoadModelResponse(BaseModel):
    """Response schema for loading a model."""
    success: bool = Field(..., description="Whether the model was loaded successfully")
    model_id: str = Field(..., description="Model ID")
    quantization: str = Field(..., description="Quantization method used")
    device: str = Field(..., description="Device where model was loaded")
    message: str = Field(..., description="Status message")


class UnloadModelRequest(BaseModel):
    """Request schema for unloading a model."""
    model_id: str = Field(..., description="Model ID to unload")
    quantization: Optional[str] = Field("none", description="Quantization method of loaded model")


class UnloadModelResponse(BaseModel):
    """Response schema for unloading a model."""
    success: bool = Field(..., description="Whether the model was unloaded successfully")
    model_id: str = Field(..., description="Model ID")
    message: str = Field(..., description="Status message")


class SystemInfoResponse(BaseModel):
    """Response schema for system information."""
    cpu: Dict[str, Any] = Field(..., description="CPU information")
    memory: Dict[str, Any] = Field(..., description="Memory information")
    gpu: Dict[str, Any] = Field(..., description="GPU information")
    models: Dict[str, Any] = Field(..., description="Model loading information")