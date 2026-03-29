#!/bin/bash
# Full preprocessing pipeline: scrape → chunk → tag → prepare dataset
#
# Usage:
#   cd /path/to/ACE-Step
#   bash preprocessing/run_all.sh [name] [max_files]
#
# Examples:
#   bash preprocessing/run_all.sh psytrance 500
#   bash preprocessing/run_all.sh darkpsy 1000
#   bash preprocessing/run_all.sh                   # defaults: psytrance, 500
#
# Prerequisites:
#   pip install -r preprocessing/requirements.txt
#   ffmpeg must be installed (brew install ffmpeg / apt install ffmpeg)

set -e

NAME="${1:-psytrance}"
MAX_FILES="${2:-500}"
DATA_DIR="./data/${NAME}"
DATASET_NAME="data/${NAME}_lora_dataset"

echo "============================================"
echo "  ACE-Step Data Preprocessing: ${NAME}"
echo "============================================"

echo ""
echo "[1/4] Scraping MP3s from archive (max: $MAX_FILES)..."
python preprocessing/01_scrape.py \
    --output_dir "${DATA_DIR}/raw_mp3" \
    --max_files "$MAX_FILES"

echo ""
echo "[2/4] Chunking into 240s segments..."
python preprocessing/02_chunk.py \
    --input_dir "${DATA_DIR}/raw_mp3" \
    --output_dir "${DATA_DIR}/chunked_mp3"

echo ""
echo "[3/4] Generating trance prompts (Essentia + CLAP)..."
python preprocessing/03_generate_prompts.py \
    --input_dir "${DATA_DIR}/chunked_mp3"

echo ""
echo "[4/4] Moving to data dir and creating HF dataset..."
python preprocessing/04_prepare_dataset.py \
    --input_dir "${DATA_DIR}/chunked_mp3" \
    --data_dir "${DATA_DIR}" \
    --repeat_count 50 \
    --output_name "${DATASET_NAME}"

echo ""
echo "============================================"
echo "  Done! Start training with:"
echo ""
echo "  python trainer.py \\"
echo "    --dataset_path ./${DATASET_NAME} \\"
echo "    --exp_name ${NAME}_lora \\"
echo "    --lora_config_path config/zh_rap_lora_config.json \\"
echo "    --learning_rate 1e-4 \\"
echo "    --max_steps 200000"
echo "============================================"
