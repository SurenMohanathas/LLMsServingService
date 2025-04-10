"""
Model management functionality.
"""
import os
import time
import gc
from typing import Dict, Optional, List, Any, Tuple, Union
import threading
import psutil
import torch
from loguru import logger

from ..utils.config import ModelConfig, get_model_by_id, ModelsConfig, ModelSettings, Config
from ..utils.logging import log_model_load, log_model_unload


class ModelInstance:
    """Represents a loaded model instance."""
    
    def __init__(
        self,
        model_id: str,
        model_config: ModelConfig,
        quantization: str,
        device: str,
        model: Any,
        tokenizer: Any,
    ):
        self.model_id = model_id
        self.model_config = model_config
        self.quantization = quantization
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.last_used = time.time()
        self.requests_processed = 0
        self.total_tokens_processed = 0
        self.load_time = time.time()
        self.lock = threading.RLock()
    
    def update_usage(self, tokens_processed: int = 0):
        """Update usage statistics for this model instance."""
        with self.lock:
            self.last_used = time.time()
            self.requests_processed += 1
            self.total_tokens_processed += tokens_processed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this model instance."""
        with self.lock:
            return {
                "model_id": self.model_id,
                "quantization": self.quantization,
                "device": self.device,
                "load_time": self.load_time,
                "uptime_seconds": time.time() - self.load_time,
                "last_used": self.last_used,
                "idle_time_seconds": time.time() - self.last_used,
                "requests_processed": self.requests_processed,
                "total_tokens_processed": self.total_tokens_processed,
            }


class ModelManager:
    """Manages loading, unloading, and tracking of models."""
    
    def __init__(self, models_config: ModelsConfig, settings: ModelSettings):
        self.models_config = models_config
        self.settings = settings
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.lock = threading.RLock()
        
        # Create model cache directory if it doesn't exist
        os.makedirs(self.settings.cache_dir, exist_ok=True)
        
        # Track GPU memory used
        self.gpu_memory_used = 0
        
        # Initialize default model if specified
        if self.settings.default_model:
            try:
                self.load_model(
                    self.settings.default_model,
                    self.settings.default_quantization,
                    self.settings.use_gpu,
                )
            except Exception as e:
                logger.error(f"Failed to load default model: {str(e)}")
    
    def get_device(self) -> str:
        """Get the appropriate device for model loading."""
        if self.settings.use_gpu and torch.cuda.is_available():
            return "cuda"
        if self.settings.fallback_to_cpu:
            return "cpu"
        raise RuntimeError("GPU requested but not available, and fallback to CPU is disabled")
    
    def get_free_gpu_memory(self) -> int:
        """Get free GPU memory in MB."""
        if not torch.cuda.is_available():
            return 0
        
        # Get free memory from first GPU
        free_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory -= torch.cuda.memory_allocated(0)
        free_memory -= torch.cuda.memory_reserved(0)
        
        return free_memory // (1024 * 1024)  # Convert to MB
    
    def load_model(
        self, model_id: str, quantization: str = "none", use_gpu: Optional[bool] = None
    ) -> ModelInstance:
        """Load a model by ID with specified quantization."""
        with self.lock:
            # Check if model is already loaded
            model_key = f"{model_id}_{quantization}"
            if model_key in self.loaded_models:
                # Update last used time
                self.loaded_models[model_key].update_usage()
                return self.loaded_models[model_key]
            
            # Get model configuration
            model_config = get_model_by_id(model_id, self.models_config)
            if not model_config:
                raise ValueError(f"Model not found: {model_id}")
            
            # Check if quantization is valid for this model
            if quantization not in model_config.quantization:
                raise ValueError(
                    f"Quantization '{quantization}' not supported for model '{model_id}'. "
                    f"Supported quantizations: {model_config.quantization}"
                )
            
            # Determine device
            if use_gpu is None:
                use_gpu = self.settings.use_gpu
            
            device = self.get_device() if use_gpu else "cpu"
            
            # Check if we need to unload models to make space
            if len(self.loaded_models) >= self.settings.max_loaded_models:
                self._unload_least_recently_used()
            
            # Import model loading functions here to avoid slow startup
            if model_config.type == "causal_lm":
                self._load_causal_lm(model_id, model_config, quantization, device)
            elif model_config.type == "embedding":
                self._load_embedding_model(model_id, model_config, quantization, device)
            else:
                raise ValueError(f"Unsupported model type: {model_config.type}")
            
            return self.loaded_models[model_key]
    
    def _load_causal_lm(
        self, model_id: str, model_config: ModelConfig, quantization: str, device: str
    ) -> None:
        """Load a causal language model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Log loading
        log_model_load(model_id, quantization, device)
        
        # Define quantization settings
        quantization_config = None
        if quantization == "4-bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8-bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_config.huggingface_repo,
            device_map=device if device == "cuda" else None,
            cache_dir=self.settings.cache_dir,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.huggingface_repo,
            cache_dir=self.settings.cache_dir,
        )
        
        # Move model to device if not quantized and not on cuda
        if quantization == "none" and device != "cuda":
            model = model.to(device)
        
        # Store model instance
        model_key = f"{model_id}_{quantization}"
        self.loaded_models[model_key] = ModelInstance(
            model_id=model_id,
            model_config=model_config,
            quantization=quantization,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
    
    def _load_embedding_model(
        self, model_id: str, model_config: ModelConfig, quantization: str, device: str
    ) -> None:
        """Load an embedding model."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Log loading
        log_model_load(model_id, quantization, device)
        
        # Load model
        model = AutoModel.from_pretrained(
            model_config.huggingface_repo,
            cache_dir=self.settings.cache_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.huggingface_repo,
            cache_dir=self.settings.cache_dir,
        )
        
        # Move model to device
        model = model.to(device)
        
        # Store model instance
        model_key = f"{model_id}_{quantization}"
        self.loaded_models[model_key] = ModelInstance(
            model_id=model_id,
            model_config=model_config,
            quantization=quantization,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
    
    def unload_model(self, model_id: str, quantization: str = "none") -> bool:
        """Unload a model by ID."""
        with self.lock:
            model_key = f"{model_id}_{quantization}"
            if model_key not in self.loaded_models:
                logger.warning(f"Model not loaded: {model_key}")
                return False
            
            # Log unloading
            log_model_unload(model_id)
            
            # Remove from loaded models
            model_instance = self.loaded_models.pop(model_key)
            
            # Delete model references
            del model_instance.model
            del model_instance.tokenizer
            del model_instance
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
    
    def _unload_least_recently_used(self) -> None:
        """Unload the least recently used model."""
        if not self.loaded_models:
            return
        
        # Find the least recently used model
        lru_key = min(
            self.loaded_models.keys(),
            key=lambda k: self.loaded_models[k].last_used,
        )
        
        # Unload the model
        model_id, quantization = lru_key.split("_", 1)
        self.unload_model(model_id, quantization)
    
    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get information about all loaded models."""
        with self.lock:
            return [
                {
                    "model_id": instance.model_id,
                    "name": instance.model_config.name,
                    "quantization": instance.quantization,
                    "device": instance.device,
                    "type": instance.model_config.type,
                    "family": instance.model_config.family,
                    **instance.get_stats(),
                }
                for instance in self.loaded_models.values()
            ]
    
    def get_model_instance(
        self, model_id: str, quantization: str = "none"
    ) -> Optional[ModelInstance]:
        """Get a loaded model instance by ID."""
        model_key = f"{model_id}_{quantization}"
        return self.loaded_models.get(model_key)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        # Memory info
        memory = psutil.virtual_memory()
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # GPU info
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info["count"] = gpu_count
            gpu_info["devices"] = []
            
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                device_mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
                device_mem_used = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                device_mem_free = device_mem_total - device_mem_used
                
                gpu_info["devices"].append({
                    "id": i,
                    "name": device_name,
                    "memory_total_gb": device_mem_total,
                    "memory_used_gb": device_mem_used,
                    "memory_free_gb": device_mem_free,
                })
        
        return {
            "cpu": {
                "count": psutil.cpu_count(),
                "percent": cpu_percent,
            },
            "memory": {
                "total_gb": memory.total / (1024 ** 3),
                "available_gb": memory.available / (1024 ** 3),
                "used_gb": memory.used / (1024 ** 3),
                "percent": memory.percent,
            },
            "gpu": gpu_info if torch.cuda.is_available() else {"available": False},
            "models": {
                "loaded_count": len(self.loaded_models),
                "max_loaded": self.settings.max_loaded_models,
            },
        }