"""
Gradio UI for LLM Service management.
"""
import time
import json
import threading
import requests
from typing import Dict, List, Any, Optional, Tuple
import gradio as gr
import pandas as pd
import plotly.graph_objects as go

# Constants
API_URL = "http://localhost:8000/api/v1"
REFRESH_INTERVAL_SECONDS = 5


def format_size(size_bytes: float) -> str:
    """Format size in bytes to human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def get_available_models() -> List[Dict[str, Any]]:
    """Get list of available models from API."""
    try:
        response = requests.get(f"{API_URL}/models/available")
        return response.json()
    except Exception as e:
        print(f"Error fetching available models: {str(e)}")
        return []


def get_loaded_models() -> List[Dict[str, Any]]:
    """Get list of loaded models from API."""
    try:
        response = requests.get(f"{API_URL}/models/loaded")
        return response.json()
    except Exception as e:
        print(f"Error fetching loaded models: {str(e)}")
        return []


def get_system_info() -> Dict[str, Any]:
    """Get system information from API."""
    try:
        response = requests.get(f"{API_URL}/models/system-info")
        return response.json()
    except Exception as e:
        print(f"Error fetching system info: {str(e)}")
        return {
            "cpu": {"count": 0, "percent": 0},
            "memory": {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0},
            "gpu": {"available": False},
            "models": {"loaded_count": 0, "max_loaded": 0},
        }


def load_model(model_id: str, quantization: str, use_gpu: bool) -> str:
    """Load a model using the API."""
    try:
        response = requests.post(
            f"{API_URL}/models/load",
            json={"model_id": model_id, "quantization": quantization, "use_gpu": use_gpu},
        )
        result = response.json()
        if result.get("success"):
            return f"✅ Model {model_id} loaded successfully on {result.get('device')}"
        else:
            return f"❌ Failed to load model: {result.get('message', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def unload_model(model_id: str, quantization: str) -> str:
    """Unload a model using the API."""
    try:
        response = requests.post(
            f"{API_URL}/models/unload",
            json={"model_id": model_id, "quantization": quantization},
        )
        result = response.json()
        if result.get("success"):
            return f"✅ Model {model_id} unloaded successfully"
        else:
            return f"❌ {result.get('message', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error unloading model: {str(e)}"


def create_model_manager_tab() -> gr.Blocks:
    """Create the model manager tab."""
    available_models = get_available_models()
    model_ids = sorted([model["id"] for model in available_models])
    
    with gr.Blocks() as manager_tab:
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### Available Models")
                
                # Filter options
                with gr.Row():
                    model_type_filter = gr.Dropdown(
                        choices=["All", "causal_lm", "embedding"],
                        value="All",
                        label="Model Type",
                    )
                    model_family_filter = gr.Dropdown(
                        choices=["All"] + sorted(list(set(model["family"] for model in available_models))),
                        value="All",
                        label="Model Family",
                    )
                
                # Available models table
                available_df = pd.DataFrame([
                    {
                        "ID": model["id"],
                        "Name": model["name"],
                        "Type": model["type"],
                        "Family": model["family"],
                        "Context Length": model["context_length"],
                        "RAM Required (GB)": model["ram_required"],
                        "GPU Required (GB)": model["gpu_required"],
                    }
                    for model in available_models
                ])
                
                available_models_table = gr.DataFrame(
                    value=available_df,
                    headers=["ID", "Name", "Type", "Family", "Context Length", "RAM Required (GB)", "GPU Required (GB)"],
                    datatype=["str", "str", "str", "str", "number", "number", "number"],
                    interactive=False,
                )
                
                def filter_models(model_type, model_family):
                    filtered_models = available_models
                    if model_type != "All":
                        filtered_models = [m for m in filtered_models if m["type"] == model_type]
                    if model_family != "All":
                        filtered_models = [m for m in filtered_models if m["family"] == model_family]
                    
                    return pd.DataFrame([
                        {
                            "ID": model["id"],
                            "Name": model["name"],
                            "Type": model["type"],
                            "Family": model["family"],
                            "Context Length": model["context_length"],
                            "RAM Required (GB)": model["ram_required"],
                            "GPU Required (GB)": model["gpu_required"],
                        }
                        for model in filtered_models
                    ])
                
                # Connect the filters to the table
                model_type_filter.change(
                    filter_models,
                    inputs=[model_type_filter, model_family_filter],
                    outputs=[available_models_table],
                )
                model_family_filter.change(
                    filter_models,
                    inputs=[model_type_filter, model_family_filter],
                    outputs=[available_models_table],
                )
                
                # Model details section
                gr.Markdown("### Model Details")
                
                model_selector = gr.Dropdown(
                    choices=model_ids,
                    label="Select Model",
                    value=model_ids[0] if model_ids else None,
                )
                
                model_details = gr.JSON(label="Model Details")
                
                def get_model_details(model_id):
                    for model in available_models:
                        if model["id"] == model_id:
                            return model
                    return {}
                
                model_selector.change(
                    get_model_details,
                    inputs=[model_selector],
                    outputs=[model_details],
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Load Model")
                
                load_model_selector = gr.Dropdown(
                    choices=model_ids,
                    label="Select Model to Load",
                    value=model_ids[0] if model_ids else None,
                )
                
                # Get available quantization options for selected model
                def get_quantization_options(model_id):
                    for model in available_models:
                        if model["id"] == model_id:
                            return model["quantization"]
                    return ["none"]
                
                quantization_selector = gr.Dropdown(
                    choices=["none", "4-bit", "8-bit"],
                    label="Quantization Method",
                    value="none",
                )
                
                load_model_selector.change(
                    lambda model_id: gr.Dropdown(choices=get_quantization_options(model_id)),
                    inputs=[load_model_selector],
                    outputs=[quantization_selector],
                )
                
                use_gpu = gr.Checkbox(label="Use GPU (if available)", value=True)
                
                load_button = gr.Button("Load Model", variant="primary")
                load_result = gr.Textbox(label="Load Result", interactive=False)
                
                load_button.click(
                    load_model,
                    inputs=[load_model_selector, quantization_selector, use_gpu],
                    outputs=[load_result],
                )
                
                gr.Markdown("### Unload Model")
                
                # Function to get loaded models for dropdown
                def get_loaded_model_options():
                    loaded = get_loaded_models()
                    return [f"{model['model_id']}_{model['quantization']}" for model in loaded]
                
                unload_model_selector = gr.Dropdown(
                    choices=get_loaded_model_options(),
                    label="Select Model to Unload",
                    value=None,
                )
                
                refresh_loaded_button = gr.Button("Refresh Loaded Models")
                
                refresh_loaded_button.click(
                    lambda: gr.Dropdown(choices=get_loaded_model_options()),
                    outputs=[unload_model_selector],
                )
                
                unload_button = gr.Button("Unload Model", variant="stop")
                unload_result = gr.Textbox(label="Unload Result", interactive=False)
                
                def handle_unload(model_key):
                    if not model_key:
                        return "⚠️ No model selected"
                    
                    parts = model_key.split("_")
                    model_id = parts[0]
                    quantization = "_".join(parts[1:]) if len(parts) > 1 else "none"
                    
                    result = unload_model(model_id, quantization)
                    
                    # Update the dropdown after unloading
                    return result
                
                unload_button.click(
                    handle_unload,
                    inputs=[unload_model_selector],
                    outputs=[unload_result],
                )
        
        return manager_tab


def create_model_stats_tab() -> gr.Blocks:
    """Create the model statistics tab."""
    with gr.Blocks() as stats_tab:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### System Resources")
                
                cpu_plot = gr.Plot(label="CPU Usage")
                memory_plot = gr.Plot(label="Memory Usage")
                
                # Auto refresh checkbox
                auto_refresh = gr.Checkbox(label="Auto Refresh", value=True)
                
                refresh_button = gr.Button("Refresh Stats")
                
                def update_system_charts():
                    system_info = get_system_info()
                    
                    # CPU chart
                    cpu_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=system_info["cpu"]["percent"],
                        title={"text": f"CPU Usage ({system_info['cpu']['count']} cores)"},
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "blue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgreen"},
                                {"range": [50, 75], "color": "orange"},
                                {"range": [75, 100], "color": "red"},
                            ],
                        },
                    ))
                    
                    # Memory chart
                    memory_fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=system_info["memory"]["percent"],
                        title={"text": f"Memory Usage ({system_info['memory']['total_gb']:.1f} GB total)"},
                        delta={"reference": 50},
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgreen"},
                                {"range": [50, 75], "color": "orange"},
                                {"range": [75, 100], "color": "red"},
                            ],
                        },
                    ))
                    
                    memory_fig.add_annotation(
                        text=f"Available: {system_info['memory']['available_gb']:.1f} GB",
                        x=0.5,
                        y=0.7,
                        showarrow=False,
                    )
                    
                    return cpu_fig, memory_fig
                
                refresh_button.click(
                    update_system_charts,
                    outputs=[cpu_plot, memory_plot],
                )
                
                # Initialize with data
                cpu_plot, memory_plot = update_system_charts()
                
                # GPU info section
                with gr.Row():
                    gr.Markdown("### GPU Information")
                    
                    gpu_info = gr.JSON(label="GPU Status")
                    
                    def update_gpu_info():
                        system_info = get_system_info()
                        return system_info.get("gpu", {"available": False})
                    
                    gpu_info.value = update_gpu_info()
                    
                    refresh_button.click(
                        update_gpu_info,
                        outputs=[gpu_info],
                    )
            
            with gr.Column(scale=1):
                gr.Markdown("### Loaded Models")
                
                loaded_models_table = gr.DataFrame(
                    headers=["Model ID", "Name", "Quantization", "Device", "Uptime", "Requests", "Tokens"],
                    datatype=["str", "str", "str", "str", "str", "number", "number"],
                    interactive=False,
                )
                
                def update_loaded_models_table():
                    loaded_models = get_loaded_models()
                    if not loaded_models:
                        return pd.DataFrame(columns=["Model ID", "Name", "Quantization", "Device", "Uptime", "Requests", "Tokens"])
                    
                    return pd.DataFrame([
                        {
                            "Model ID": model["model_id"],
                            "Name": model["name"],
                            "Quantization": model["quantization"],
                            "Device": model["device"],
                            "Uptime": f"{model['uptime_seconds'] / 60:.1f} min",
                            "Requests": model["requests_processed"],
                            "Tokens": model["total_tokens_processed"],
                        }
                        for model in loaded_models
                    ])
                
                loaded_models_table.value = update_loaded_models_table()
                
                refresh_button.click(
                    update_loaded_models_table,
                    outputs=[loaded_models_table],
                )
                
                # Model usage chart
                model_usage_chart = gr.Plot(label="Model Usage")
                
                def update_model_usage_chart():
                    loaded_models = get_loaded_models()
                    if not loaded_models:
                        # Return empty figure
                        fig = go.Figure()
                        fig.update_layout(title="No models loaded")
                        return fig
                    
                    # Create chart
                    model_ids = [f"{model['model_id']} ({model['quantization']})" for model in loaded_models]
                    requests = [model["requests_processed"] for model in loaded_models]
                    tokens = [model["total_tokens_processed"] for model in loaded_models]
                    
                    fig = go.Figure(data=[
                        go.Bar(name="Requests", x=model_ids, y=requests),
                        go.Bar(name="Tokens (K)", x=model_ids, y=[t / 1000 for t in tokens]),
                    ])
                    
                    fig.update_layout(
                        title="Model Usage Statistics",
                        xaxis_title="Model",
                        yaxis_title="Count",
                        barmode="group",
                    )
                    
                    return fig
                
                model_usage_chart.value = update_model_usage_chart()
                
                refresh_button.click(
                    update_model_usage_chart,
                    outputs=[model_usage_chart],
                )
                
                # Function for auto-refresh
                refresh_thread = None
                refresh_thread_active = False
                
                def auto_refresh_func(interval=REFRESH_INTERVAL_SECONDS):
                    nonlocal refresh_thread_active
                    try:
                        refresh_thread_active = True
                        while refresh_thread_active:
                            time.sleep(interval)
                            # This won't directly update the UI, used as a flag
                            refresh_button.click()
                    except Exception as e:
                        print(f"Error in refresh thread: {str(e)}")
                    finally:
                        refresh_thread_active = False
                
                def toggle_auto_refresh(auto_refresh_enabled):
                    nonlocal refresh_thread, refresh_thread_active
                    if auto_refresh_enabled:
                        if refresh_thread is None or not refresh_thread.is_alive():
                            refresh_thread_active = True
                            refresh_thread = threading.Thread(target=auto_refresh_func)
                            refresh_thread.daemon = True
                            refresh_thread.start()
                    else:
                        refresh_thread_active = False
                
                auto_refresh.change(toggle_auto_refresh, inputs=[auto_refresh])
                
                # Initial auto-refresh if enabled
                if auto_refresh.value:
                    toggle_auto_refresh(True)
        
        return stats_tab


def create_demo_tab() -> gr.Blocks:
    """Create the demo tab for testing models."""
    with gr.Blocks() as demo_tab:
        gr.Markdown("### Test Text Generation")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Get loaded models for dropdown
                def get_loaded_model_choices():
                    loaded_models = get_loaded_models()
                    return [
                        f"{model['model_id']}_{model['quantization']}"
                        for model in loaded_models
                        if model["type"] == "causal_lm"
                    ]
                
                loaded_model_selector = gr.Dropdown(
                    choices=get_loaded_model_choices(),
                    label="Select Loaded Model",
                    value=None,
                )
                
                refresh_models_button = gr.Button("Refresh Models")
                
                refresh_models_button.click(
                    lambda: gr.Dropdown(choices=get_loaded_model_choices()),
                    outputs=[loaded_model_selector],
                )
                
                # Generation parameters
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                )
                
                max_length = gr.Slider(
                    minimum=10,
                    maximum=4096,
                    value=512,
                    step=10,
                    label="Max Length",
                )
                
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                )
            
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5,
                )
                
                generate_button = gr.Button("Generate", variant="primary")
                
                output_text = gr.Textbox(
                    label="Generated Text",
                    lines=10,
                    interactive=False,
                )
                
                generation_info = gr.JSON(label="Generation Info")
                
                def generate_text(model_key, prompt, temperature, max_length, top_p):
                    if not model_key or not prompt:
                        return "Please select a model and enter a prompt.", {}
                    
                    try:
                        # Split model key to get model_id and quantization
                        parts = model_key.split("_")
                        model_id = parts[0]
                        quantization = "_".join(parts[1:]) if len(parts) > 1 else "none"
                        
                        # Make API request
                        response = requests.post(
                            f"{API_URL}/generation/text",
                            json={
                                "model_id": model_id,
                                "prompt": prompt,
                                "quantization": quantization,
                                "temperature": temperature,
                                "max_length": max_length,
                                "top_p": top_p,
                            },
                        )
                        
                        result = response.json()
                        
                        # Extract generation info
                        generation_info = {
                            "model_id": result.get("model_id"),
                            "elapsed_time": f"{result.get('elapsed_time', 0):.3f} seconds",
                            "input_tokens": result.get("input_tokens", 0),
                            "output_tokens": result.get("output_tokens", 0),
                            "total_tokens": (result.get("input_tokens", 0) + result.get("output_tokens", 0)),
                        }
                        
                        return result.get("generated_text", "No text generated"), generation_info
                    except Exception as e:
                        return f"Error generating text: {str(e)}", {}
                
                generate_button.click(
                    generate_text,
                    inputs=[loaded_model_selector, prompt_input, temperature, max_length, top_p],
                    outputs=[output_text, generation_info],
                )
        
        gr.Markdown("### Test Embeddings")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Get loaded embedding models
                def get_loaded_embedding_models():
                    loaded_models = get_loaded_models()
                    return [
                        f"{model['model_id']}_{model['quantization']}"
                        for model in loaded_models
                        if model["type"] == "embedding"
                    ]
                
                embedding_model_selector = gr.Dropdown(
                    choices=get_loaded_embedding_models(),
                    label="Select Embedding Model",
                    value=None,
                )
                
                refresh_models_button.click(
                    lambda: gr.Dropdown(choices=get_loaded_embedding_models()),
                    outputs=[embedding_model_selector],
                )
                
                # Embedding parameters
                normalize = gr.Checkbox(label="Normalize Embeddings", value=True)
                
                pooling_method = gr.Radio(
                    choices=["mean", "cls"],
                    label="Pooling Method",
                    value="mean",
                )
            
            with gr.Column(scale=2):
                texts_input = gr.Textbox(
                    label="Input Texts (one text per line)",
                    placeholder="Enter texts for embedding here...",
                    lines=5,
                )
                
                embed_button = gr.Button("Generate Embeddings", variant="primary")
                
                embeddings_output = gr.JSON(label="Embeddings (truncated)")
                
                embedding_info = gr.JSON(label="Embedding Info")
                
                def generate_embeddings(model_key, texts_input, normalize, pooling_method):
                    if not model_key or not texts_input:
                        return {}, {}
                    
                    try:
                        # Split texts by newline
                        texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
                        
                        # Split model key to get model_id and quantization
                        parts = model_key.split("_")
                        model_id = parts[0]
                        quantization = "_".join(parts[1:]) if len(parts) > 1 else "none"
                        
                        # Make API request
                        response = requests.post(
                            f"{API_URL}/generation/embeddings",
                            json={
                                "model_id": model_id,
                                "texts": texts,
                                "quantization": quantization,
                                "normalize": normalize,
                                "pooling_method": pooling_method,
                            },
                        )
                        
                        result = response.json()
                        
                        # Format embeddings for display (truncate)
                        embeddings = result.get("embeddings", [])
                        truncated_embeddings = []
                        
                        for i, emb in enumerate(embeddings):
                            # Get first 5 and last 5 dimensions, with ... in between
                            dims = len(emb)
                            if dims <= 10:
                                truncated = emb
                            else:
                                truncated = emb[:5] + ["..."] + emb[-5:]
                            
                            truncated_embeddings.append({
                                f"Text {i+1}": truncated,
                            })
                        
                        # Extract embedding info
                        embedding_info = {
                            "model_id": result.get("model_id"),
                            "dimensions": result.get("dimensions", 0),
                            "input_texts": result.get("input_texts", 0),
                            "total_tokens": result.get("total_tokens", 0),
                            "elapsed_time": f"{result.get('elapsed_time', 0):.3f} seconds",
                        }
                        
                        return truncated_embeddings, embedding_info
                    except Exception as e:
                        return {}, {"error": str(e)}
                
                embed_button.click(
                    generate_embeddings,
                    inputs=[embedding_model_selector, texts_input, normalize, pooling_method],
                    outputs=[embeddings_output, embedding_info],
                )
        
        return demo_tab


def create_api_docs_tab() -> gr.Blocks:
    """Create the API documentation tab."""
    with gr.Blocks() as docs_tab:
        gr.Markdown("### API Documentation")
        
        gr.Markdown("""
        The API provides endpoints for model management and text generation. 
        Below are code examples showing how to use the API with Python.
        
        **API Base URL:** http://localhost:8000/api/v1
        
        For interactive API documentation, visit: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
        """)
        
        with gr.Accordion("List Available Models", open=False):
            gr.Code(
                """
import requests

# Get all available models
response = requests.get("http://localhost:8000/api/v1/models/available")
models = response.json()

# Print model names
for model in models:
    print(f"{model['name']} ({model['id']})")

# Filter for embedding models
response = requests.get("http://localhost:8000/api/v1/models/available?type=embedding")
embedding_models = response.json()
                """,
                language="python",
            )
        
        with gr.Accordion("Load and Unload Models", open=False):
            gr.Code(
                """
import requests

# Load a model with default settings
response = requests.post(
    "http://localhost:8000/api/v1/models/load",
    json={"model_id": "mistral-7b"}
)
result = response.json()
print(f"Model loaded: {result['success']}, Device: {result['device']}")

# Load with 4-bit quantization
response = requests.post(
    "http://localhost:8000/api/v1/models/load",
    json={"model_id": "llama3-8b", "quantization": "4-bit", "use_gpu": True}
)
result = response.json()

# Unload a model
response = requests.post(
    "http://localhost:8000/api/v1/models/unload",
    json={"model_id": "mistral-7b"}
)
result = response.json()
print(f"Model unloaded: {result['success']}")
                """,
                language="python",
            )
        
        with gr.Accordion("Generate Text", open=False):
            gr.Code(
                """
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
                """,
                language="python",
            )
        
        with gr.Accordion("Generate Embeddings", open=False):
            gr.Code(
                """
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Generate embeddings for texts
response = requests.post(
    "http://localhost:8000/api/v1/generation/embeddings",
    json={
        "model_id": "bge-small-en",
        "texts": ["Hello, world!", "This is a test."]
    }
)
result = response.json()

# Get embeddings
embeddings = result["embeddings"]
dimensions = result["dimensions"]
print(f"Generated {len(embeddings)} embeddings with {dimensions} dimensions each")

# Compute similarity between embeddings
embeddings_array = np.array(embeddings)
similarity = cosine_similarity(embeddings_array)[0, 1]
print(f"Cosine similarity: {similarity:.4f}")
                """,
                language="python",
            )
        
        return docs_tab


def create_ui() -> gr.Blocks:
    """Create the main Gradio UI."""
    with gr.Blocks(title="LLMsServingService Manager") as ui:
        gr.Markdown("# LLMsServingService Manager")
        gr.Markdown("Manage and test your locally deployed Language Models")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Model Manager", id="manager"):
                manager_tab = create_model_manager_tab()
            
            with gr.TabItem("Model Stats", id="stats"):
                stats_tab = create_model_stats_tab()
            
            with gr.TabItem("Demo", id="demo"):
                demo_tab = create_demo_tab()
            
            with gr.TabItem("API Docs", id="docs"):
                docs_tab = create_api_docs_tab()
    
    return ui