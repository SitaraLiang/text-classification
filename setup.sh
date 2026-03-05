#!/bin/bash
# =============================================================
# setup_env.sh
# Usage:
#   First time  : bash setup_env.sh      (creates venv + installs deps)
#   Every time  : source setup_env.sh    (just activates the venv)
# =============================================================

PROJECT_DIR="/tempory/NUMERO_ETUDIANT/RITAL/text-classification"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

# ── Activate if venv already exists ──
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment found at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Activated: $(which python)"
    echo "Python version: $(python --version)"
    return 0 2>/dev/null || exit 0
fi

# ── First time: create venv + install dependencies ──
echo "No virtual environment found. Creating one..."

# Check python3 is available
if ! command -v python3 &> /dev/null; then
    echo "python3 not found. Please load the python module first:"
    echo "  module load python/3.10"
    return 1 2>/dev/null || exit 1
fi

# Create venv
python3 -m venv "$VENV_DIR"
echo "Virtual environment created at $VENV_DIR"

# Activate
source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA first (check your GPU's CUDA version with: nvcc --version)
# Adjust cu118 to match your server's CUDA version (cu117, cu118, cu121, etc.)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
if [ -f "$REQUIREMENTS" ]; then
    echo "Installing from requirements.txt..."
    pip install -r "$REQUIREMENTS"
else
    echo "requirements.txt not found at $REQUIREMENTS"
fi

# Download French spaCy model
echo "Downloading French spaCy model..."
python -m spacy download fr_core_news_sm

echo ""
echo "============================================"
echo "Setup complete!"
echo "To active the virtural environment: source setup_env.sh"
echo "============================================"