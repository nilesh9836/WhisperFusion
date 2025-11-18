#!/bin/bash

echo "========================================"
echo "WhisperFusion CPU-only Setup Script"
echo "========================================"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        echo "Please ensure python3-venv is installed: sudo apt-get install python3-venv"
        exit 1
    fi
else
    echo "[1/5] Virtual environment already exists, skipping..."
fi

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

# Install CPU-only PyTorch first
echo "[3/5] Installing CPU-only PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install requirements
echo "[4/5] Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "[5/5] Creating directories..."
mkdir -p input_audio
mkdir -p input_audio/processed
mkdir -p outputs

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To run WhisperFusion in CPU mode:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. (Optional) Set OpenAI API key for OpenAI LLM provider:"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "3. Run with OpenAI provider:"
echo "   python main.py --cpu --llm_provider openai"
echo ""
echo "   OR run with HuggingFace provider:"
echo "   python main.py --cpu --llm_provider hf"
echo ""
echo "4. Drop WAV files into the input_audio/ directory"
echo "   The pipeline will process them and write outputs to outputs/"
echo ""
echo "Optional: Install espeak for audio playback:"
echo "   sudo apt-get install espeak"
echo ""
echo "========================================"
