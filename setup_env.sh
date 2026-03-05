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

# ── Force all cache/temp away from home directory (avoids quota errors) ──
export PIP_CACHE_DIR="/tempory/NUMERO_ETUDIANT/.pip_cache"
export TMPDIR="/tempory/NUMERO_ETUDIANT/tmp"
export PIP_NO_USER=1                  # never install to ~/.local
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# ── Activate if venv already exists ──
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment found at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Activated: $(which python) — $(python --version)"
    return 0 2>/dev/null || exit 0
fi

# ── First time: create venv + install dependencies ──
echo "No virtual environment found. Creating one..."

if ! command -v python3 &> /dev/null; then
    echo "python3 not found."
    return 1 2>/dev/null || exit 1
fi

# No --system-site-packages: fully isolated venv to avoid permission errors
python3 -m venv "$VENV_DIR"
echo "Virtual environment created at $VENV_DIR"

source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# Upgrade pip inside venv only
python -m pip install --upgrade pip

echo "Installing PyTorch (adjust cu118 to match: nvcc --version)..."
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118

if [ -f "$REQUIREMENTS" ]; then
    echo "Installing from requirements.txt..."
    pip install -r "$REQUIREMENTS"
else
    echo "requirements.txt not found at $REQUIREMENTS"
fi

echo "Downloading French spaCy model..."
python -m spacy download fr_core_news_sm

echo ""
echo "============================================"
echo "Setup complete!"
echo "Next time: source setup_env.sh"
echo "============================================"