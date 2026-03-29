#!/bin/bash
# Full preprocessing pipeline: scrape → chunk → tag → prepare dataset
#
# Usage:
#   cd /path/to/ACE-Step
#   bash preprocessing/run_all.sh [max_files]    # e.g. bash preprocessing/run_all.sh 500
#
# Prerequisites:
#   pip install -r preprocessing/requirements.txt
#   ffmpeg must be installed (brew install ffmpeg / apt install ffmpeg)

set -e

MAX_FILES="${1:-500}"

echo "============================================"
echo "  ACE-Step Psytrance Data Preprocessing"
echo "============================================"

echo ""
echo "[1/4] Scraping MP3s from archive (max: $MAX_FILES)..."
python preprocessing/01_scrape.py \
    --output_dir ./preprocessing/raw_mp3 \
    --max_files "$MAX_FILES"

echo ""
echo "[2/4] Chunking into 240s segments..."
python preprocessing/02_chunk.py \
    --input_dir ./preprocessing/raw_mp3 \
    --output_dir ./preprocessing/chunked_mp3

echo ""
echo "[3/4] Generating trance prompts (Essentia + CLAP)..."
python preprocessing/03_generate_prompts.py \
    --input_dir ./preprocessing/chunked_mp3

echo ""
echo "[4/4] Moving to data dir and creating HF dataset..."
python preprocessing/04_prepare_dataset.py \
    --input_dir ./preprocessing/chunked_mp3 \
    --data_dir ./data_psytrance \
    --repeat_count 50 \
    --output_name psytrance_lora_dataset

echo ""
echo "============================================"
echo "  Done! Start training with:"
echo ""
echo "  python trainer.py \\"
echo "    --dataset_path ./psytrance_lora_dataset \\"
echo "    --exp_name psytrance_lora \\"
echo "    --lora_config_path config/zh_rap_lora_config.json \\"
echo "    --learning_rate 1e-4 \\"
echo "    --max_steps 200000"
echo "============================================"
