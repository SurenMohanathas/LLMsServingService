#!/bin/bash
# Setup script for LLMsServingService

# Default settings
PYTHON_VERSION="3.8"
CREATE_VENV=true
INSTALL_DEPS=true
CUDA_SUPPORT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-venv)
      CREATE_VENV=false
      shift
      ;;
    --no-deps)
      INSTALL_DEPS=false
      shift
      ;;
    --cuda)
      CUDA_SUPPORT=true
      shift
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set the python command based on version
PYTHON_CMD="python$PYTHON_VERSION"

# Check for Python
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Python $PYTHON_VERSION is required but not found."
    echo "Please install Python $PYTHON_VERSION and try again."
    exit 1
fi

# Create virtual environment if requested
if [ "$CREATE_VENV" = true ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
fi

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    echo "Installing dependencies..."
    
    if [ "$CUDA_SUPPORT" = true ]; then
        echo "Installing with CUDA support..."
        # Install PyTorch with CUDA
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        # Install other dependencies
        pip install -r requirements.txt
    else
        # Install all dependencies
        pip install -r requirements.txt
    fi
fi

# Create cache directory for models
mkdir -p model_cache

# Create logs directory
mkdir -p logs

echo "Setup completed successfully!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the service, run:"
echo "  ./scripts/start.sh"