#!/bin/bash
# Start LLMsServingService

# Default settings
API_PORT=8000
UI_PORT=7860
MODE="both" # both, api, ui

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-only)
      MODE="api"
      shift
      ;;
    --ui-only)
      MODE="ui"
      shift
      ;;
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --ui-port)
      UI_PORT="$2"
      shift 2
      ;;
    --debug)
      DEBUG="--debug"
      shift
      ;;
    --reload)
      RELOAD="--reload"
      shift
      ;;
    --share)
      SHARE="--share"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found."
    exit 1
fi

# Check for virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Build command based on mode
if [ "$MODE" = "api" ]; then
    CMD="python3 main.py --api-only --api-port $API_PORT $DEBUG $RELOAD"
elif [ "$MODE" = "ui" ]; then
    CMD="python3 main.py --ui-only --ui-port $UI_PORT $SHARE"
else
    CMD="python3 main.py --api-port $API_PORT --ui-port $UI_PORT $DEBUG $RELOAD $SHARE"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the service
echo "Starting LLMsServingService..."
echo "Command: $CMD"
exec $CMD