#!/usr/bin/env bash
# scripts/download_models.sh

BASE="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
MODELS="models"

curl -L -o "$MODELS/melspectrogram.onnx" "$BASE/melspectrogram.onnx"
curl -L -o "$MODELS/embedding_model.onnx" "$BASE/embedding_model.onnx"
curl -L -o "$MODELS/hey_jarvis_v0.1.onnx" "$BASE/hey_jarvis_v0.1.onnx"

# silero VAD v5 — must be v5 specifically, master is v6
curl -L -o "$MODELS/silero_vad.onnx" \
  "https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx"
