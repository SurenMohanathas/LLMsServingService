"""
Configuration management utilities.
"""
import os
from typing import Dict, Any, List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    id: str
    name: str
    huggingface_repo: str
    type: str
    family: str
    context_length: int
    ram_required: int
    gpu_required: int
    quantization: List[str]
    tags: List[str]
    description: str


class ModelsConfig(BaseModel):
    """Configuration for all available models."""
    models: List[ModelConfig]


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    timeout: int = 180
    api_prefix: str = "/api/v1"


class ModelSettings(BaseModel):
    """Settings for model loading and management."""
    cache_dir: str = "./model_cache"
    max_loaded_models: int = 2
    default_model: Optional[str] = None
    default_quantization: str = "none"
    use_gpu: bool = True
    fallback_to_cpu: bool = True


class InferenceSettings(BaseModel):
    """Settings for inference."""
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_batch_size: int = 4
    cache_responses: bool = True
    max_inference_time: int = 60


class SecuritySettings(BaseModel):
    """Security settings."""
    enable_auth: bool = False
    token_expiration: int = 86400
    rate_limit: int = 60
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])


class LoggingSettings(BaseModel):
    """Logging settings."""
    level: str = "INFO"
    file: str = "logs/server.log"
    max_size: int = 10
    backup_count: int = 5
    log_payloads: bool = False
    log_model_events: bool = True


class Config(BaseModel):
    """Main configuration."""
    server: ServerConfig
    models: ModelSettings
    inference: InferenceSettings
    security: SecuritySettings
    logging: LoggingSettings


def load_models_config(config_path: str = "config/models_config.yaml") -> ModelsConfig:
    """Load models configuration from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Models config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    return ModelsConfig(**config_data)


def load_server_config(config_path: str = "config/server_config.yaml") -> Config:
    """Load server configuration from YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Server config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    return Config(**config_data)


def get_model_by_id(model_id: str, config: ModelsConfig) -> Optional[ModelConfig]:
    """Get model configuration by ID."""
    for model in config.models:
        if model.id == model_id:
            return model
    return None