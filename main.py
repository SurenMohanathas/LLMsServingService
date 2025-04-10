#!/usr/bin/env python3
"""
Main entry point for LLMsServingService.
"""
import os
import argparse
import threading
import uvicorn
import gradio as gr
import time
from typing import Optional

from src.api.app import create_app
from src.ui.gradio_app import create_ui
from src.utils.config import load_server_config, load_models_config


def run_api_server(
    host: str, port: int, workers: int = 1, reload: bool = False, debug: bool = False
) -> None:
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.app:create_app",
        host=host,
        port=port,
        factory=True,
        workers=workers,
        reload=reload,
        log_level="debug" if debug else "info",
    )


def run_ui_server(ui_port: int, share: bool = False) -> None:
    """Run the Gradio UI server."""
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", server_port=ui_port, share=share, show_api=False)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLMsServingService")
    
    # Common arguments
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/server_config.yaml",
        help="Path to server configuration file",
    )
    parser.add_argument(
        "--models-config", 
        type=str, 
        default="config/models_config.yaml",
        help="Path to models configuration file",
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--api-only", 
        action="store_true", 
        help="Run only the API server",
    )
    mode_group.add_argument(
        "--ui-only", 
        action="store_true", 
        help="Run only the UI server",
    )
    
    # API server arguments
    parser.add_argument(
        "--api-host", 
        type=str, 
        help="API server host",
    )
    parser.add_argument(
        "--api-port", 
        type=int, 
        help="API server port",
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        help="Number of API server workers",
    )
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable API server auto-reload (development)",
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode",
    )
    
    # UI server arguments
    parser.add_argument(
        "--ui-port", 
        type=int, 
        default=7860,
        help="UI server port",
    )
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Share the UI via Gradio",
    )
    
    args = parser.parse_args()
    
    # Load configurations
    try:
        server_config = load_server_config(args.config)
        models_config = load_models_config(args.models_config)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return
    
    # Determine mode and settings
    run_api = not args.ui_only
    run_ui = not args.api_only
    
    api_host = args.api_host or server_config.server.host
    api_port = args.api_port or server_config.server.port
    workers = args.workers or server_config.server.workers
    debug = args.debug or server_config.server.debug
    
    # Print banner
    print("=" * 60)
    print(f"LLMsServingService")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Models: {args.models_config} ({len(models_config.models)} models available)")
    
    if run_api:
        print(f"API Server: http://{api_host}:{api_port}")
        print(f"API Documentation: http://{api_host}:{api_port}{server_config.server.api_prefix}/docs")
    
    if run_ui:
        print(f"UI Server: http://localhost:{args.ui_port}")
    
    print("=" * 60)
    
    # Start servers based on mode
    api_thread = None
    ui_thread = None
    
    if run_api:
        if run_ui:
            # Run API in a separate thread if we're also running UI
            api_thread = threading.Thread(
                target=run_api_server,
                args=(api_host, api_port, workers, args.reload, debug),
                daemon=True,
            )
            api_thread.start()
            print("API server started in background thread.")
            
            # Give API server time to start
            time.sleep(2)
            
            # Run UI in main thread
            print("Starting UI server...")
            run_ui_server(args.ui_port, args.share)
        else:
            # Run API in main thread
            print("Starting API server...")
            run_api_server(api_host, api_port, workers, args.reload, debug)
    else:
        # Run UI in main thread
        print("Starting UI server...")
        run_ui_server(args.ui_port, args.share)


if __name__ == "__main__":
    main()