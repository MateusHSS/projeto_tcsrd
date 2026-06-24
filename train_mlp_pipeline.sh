#!/bin/bash

# Exit on error
set -e

# Detect virtual environment python, fallback to system python3
PYTHON_EXEC=".venv/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

echo "=== MLP RL Training Pipeline ==="
echo "Stage 1: Training MLP model on NetworkX (Mathematical graph)..."
$PYTHON_EXEC train_ppo.py

MODEL_PATH="modelos_pre_treinados/escala_50_ppo_mlp.zip"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[Error] Pretrained model not found at $MODEL_PATH. Aborting NS-3 stage."
    exit 1
fi

echo ""
echo "Stage 2: Fine-tuning MLP model on NS-3 (Packet-level simulator)..."
$PYTHON_EXEC train_ppo.py --use_ns3

echo ""
echo "=== MLP Training Pipeline Finished Successfully ==="
