# LLMsServingService

A service for deploying and serving open-source LLMs in a local GPU-enabled environment, with a user-friendly UI for model management and a powerful API for integration.

## Features

- **Model Management**: Load, unload, and manage multiple LLMs on local hardware
- **GPU Acceleration**: Leverage local GPU resources for faster inference
- **Quantization Support**: Run models with 4-bit or 8-bit quantization to reduce memory usage
- **Multiple Model Types**: Support for causal language models and embedding models
- **RESTful API**: Well-documented API with comprehensive endpoints
- **Web UI**: Intuitive Gradio UI for model management and testing
- **System Monitoring**: Track resource usage and model performance
- **Configurable**: Easily configure available models and server settings

## Screenshots

![Model Manager](./docs/images/model_manager.png)
![Model Stats](./docs/images/model_stats.png)
![Text Generation](./docs/images/text_generation.png)

## Requirements

- Python 3.8+
- 16+ GB RAM
- NVIDIA GPU with 8+ GB VRAM (optional, but recommended)
- CUDA 11.8+ (for GPU acceleration)

## Installation

### Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/LLMsServingService.git
   cd LLMsServingService
   ```

2. Run the setup script:
   ```
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. Start the service:
   ```
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

4. Access the UI at http://localhost:7860 and the API at http://localhost:8000

### Manual Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the service:
   ```
   python main.py
   ```

### With GPU Support

To enable CUDA support for GPU acceleration:

```
./scripts/setup.sh --cuda
```

## Configuration

### Server Configuration

Edit `config/server_config.yaml` to customize server settings:

- API host and port
- Model settings (cache directory, default model, etc.)
- Inference parameters (max length, temperature, etc.)
- Security settings (authentication, rate limiting, etc.)

### Model Configuration

Edit `config/models_config.yaml` to add or modify available models:

- Model ID and name
- Hugging Face repository
- Model type and family
- Resource requirements
- Supported quantization methods
- Tags and description

## Usage

### Web UI

The web UI is available at http://localhost:7860 and provides:

- Model Manager: View available models, load and unload models
- Model Stats: Monitor system resources and model usage
- Demo: Test text generation and embeddings with loaded models
- API Docs: View API documentation and example code

### API

The API is available at http://localhost:8000 with interactive documentation at http://localhost:8000/api/v1/docs.

#### List Available Models

```python
import requests

# Get all available models
response = requests.get("http://localhost:8000/api/v1/models/available")
models = response.json()

# Filter for embedding models
response = requests.get("http://localhost:8000/api/v1/models/available?type=embedding")
embedding_models = response.json()
```

#### Load a Model

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
```

#### Generate Text

```python
import requests

# Generate text
response = requests.post(
    "http://localhost:8000/api/v1/generation/text",
    json={
        "model_id": "mistral-7b",
        "prompt": "Write a short poem about artificial intelligence:",
        "temperature": 0.7,
        "max_length": 512
    }
)
result = response.json()
print(result["generated_text"])
```

#### Generate Embeddings

```python
import requests

# Generate embeddings
response = requests.post(
    "http://localhost:8000/api/v1/generation/embeddings",
    json={
        "model_id": "bge-small-en",
        "texts": ["Hello, world!", "This is a test."]
    }
)
result = response.json()
embeddings = result["embeddings"]
```

## Command-Line Interface

```
usage: main.py [-h] [--config CONFIG] [--models-config MODELS_CONFIG]
               [--api-only | --ui-only] [--api-host API_HOST]
               [--api-port API_PORT] [--workers WORKERS] [--reload] [--debug]
               [--ui-port UI_PORT] [--share]

LLMsServingService

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to server configuration file
  --models-config MODELS_CONFIG
                        Path to models configuration file
  --api-only            Run only the API server
  --ui-only             Run only the UI server
  --api-host API_HOST   API server host
  --api-port API_PORT   API server port
  --workers WORKERS     Number of API server workers
  --reload              Enable API server auto-reload (development)
  --debug               Enable debug mode
  --ui-port UI_PORT     UI server port
  --share               Share the UI via Gradio
```

## Advanced Usage

### Adding Custom Models

To add a custom model:

1. Edit `config/models_config.yaml` and add a new entry following the existing format
2. Restart the service

### Running in Production

For production deployments:

1. Configure proper authentication in `config/server_config.yaml`
2. Use multiple workers for better performance: `--workers 4`
3. Consider using a reverse proxy (Nginx, Caddy) for HTTPS
4. Set appropriate resource limits

## License

This project is licensed under the MIT License - see the LICENSE file for details.