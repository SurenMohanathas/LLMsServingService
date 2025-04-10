"""
Inference functionality for models.
"""
import time
import threading
from typing import Dict, Optional, List, Any, Union, Callable
import torch
from loguru import logger

from ..utils.config import ModelConfig, InferenceSettings
from .model_manager import ModelInstance, ModelManager


class Inference:
    """Inference functionality for language models."""
    
    def __init__(self, model_manager: ModelManager, settings: InferenceSettings):
        self.model_manager = model_manager
        self.settings = settings
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def generate_text(
        self,
        model_id: str,
        prompt: str,
        quantization: str = "none",
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        num_return_sequences: int = 1,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate text using a causal language model."""
        # Check cache if enabled and key provided
        if self.settings.cache_responses and cache_key:
            with self.lock:
                cache_item = self.response_cache.get(cache_key)
                if cache_item:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cache_item
        
        # Get model instance (loading if necessary)
        model_instance = self.model_manager.load_model(model_id, quantization)
        
        # Make sure model is causal LM
        if model_instance.model_config.type != "causal_lm":
            raise ValueError(f"Model {model_id} is not a causal language model")
        
        # Use default settings if not provided
        max_length = max_length or self.settings.max_length
        temperature = temperature or self.settings.temperature
        top_p = top_p or self.settings.top_p
        top_k = top_k or self.settings.top_k
        repetition_penalty = repetition_penalty or self.settings.repetition_penalty
        
        # Start timing
        start_time = time.time()
        
        # Run inference
        with model_instance.lock:
            input_ids = model_instance.tokenizer.encode(prompt, return_tensors="pt")
            
            # Move input to model's device
            input_ids = input_ids.to(model_instance.device)
            
            # Prepare sampling parameters
            generation_args = {
                "input_ids": input_ids,
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0.0,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": model_instance.tokenizer.eos_token_id,
            }
            
            # Add specific configuration based on model family
            if model_instance.model_config.family == "llama":
                generation_args["eos_token_id"] = model_instance.tokenizer.eos_token_id
            
            # Generate text
            with torch.no_grad():
                outputs = model_instance.model.generate(**generation_args)
            
            # Decode outputs
            generated_texts = []
            for i in range(num_return_sequences):
                output_sequence = outputs[i]
                # Remove input tokens from output
                if output_sequence.shape[0] > input_ids.shape[1]:
                    output_sequence = output_sequence[input_ids.shape[1]:]
                
                # Decode to text
                text = model_instance.tokenizer.decode(output_sequence, skip_special_tokens=True)
                generated_texts.append(text)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Update model usage stats
        tokens_processed = input_ids.shape[1] + sum(len(text.split()) for text in generated_texts)
        model_instance.update_usage(tokens_processed)
        
        # Prepare response
        response = {
            "model_id": model_id,
            "generated_text": generated_texts[0] if num_return_sequences == 1 else generated_texts,
            "elapsed_time": elapsed_time,
            "input_tokens": input_ids.shape[1],
            "output_tokens": sum(len(text.split()) for text in generated_texts),
        }
        
        # Cache response if caching is enabled
        if self.settings.cache_responses and cache_key:
            with self.lock:
                self.response_cache[cache_key] = response
        
        return response
    
    def generate_embeddings(
        self,
        model_id: str,
        texts: List[str],
        quantization: str = "none",
        pooling_method: str = "mean",
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """Generate embeddings for a list of texts."""
        # Get model instance (loading if necessary)
        model_instance = self.model_manager.load_model(model_id, quantization)
        
        # Make sure model is embedding model
        if model_instance.model_config.type != "embedding":
            raise ValueError(f"Model {model_id} is not an embedding model")
        
        # Start timing
        start_time = time.time()
        
        # Run inference
        with model_instance.lock:
            embeddings = []
            token_count = 0
            
            for text in texts:
                # Tokenize
                encoded = model_instance.tokenizer(
                    text,
                    truncation=True,
                    max_length=model_instance.model_config.context_length,
                    return_tensors="pt",
                )
                
                # Move tensors to the right device
                input_ids = encoded["input_ids"].to(model_instance.device)
                attention_mask = encoded["attention_mask"].to(model_instance.device)
                
                token_count += input_ids.shape[1]
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model_instance.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    
                    # Get last hidden state or the specified layer
                    hidden_states = outputs.last_hidden_state
                    
                    # Apply pooling
                    if pooling_method == "mean":
                        # Apply attention mask
                        masked_hidden_states = hidden_states * attention_mask.unsqueeze(-1)
                        # Sum and divide by number of tokens
                        embedding = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                    elif pooling_method == "cls":
                        # Use [CLS] token embedding
                        embedding = hidden_states[:, 0]
                    else:
                        raise ValueError(f"Unknown pooling method: {pooling_method}")
                    
                    # Normalize if requested
                    if normalize:
                        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                    
                    # Convert to list
                    embedding_list = embedding[0].cpu().numpy().tolist()
                    embeddings.append(embedding_list)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Update model usage stats
        model_instance.update_usage(token_count)
        
        return {
            "model_id": model_id,
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "input_texts": len(texts),
            "total_tokens": token_count,
            "elapsed_time": elapsed_time,
        }
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        with self.lock:
            self.response_cache.clear()