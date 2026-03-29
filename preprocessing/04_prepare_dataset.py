"""
Step 4: Move processed MP3s + prompts + lyrics into the ACE-Step data directory
and convert to HuggingFace dataset format for LoRA fine-tuning.

Usage:
    python preprocessing/04_prepare_dataset.py \
        --input_dir ./preprocessing/chunked_mp3 \
        --data_dir ./data \
        --repeat_count 50 \
        --output_name psytrance_lora_dataset
"""

import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Add parent dir to path so we can import the converter
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert2hf_dataset import create_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Move processed files to data dir and create HF dataset"
    )
    parser.add_argument("--input_dir", default="./preprocessing/chunked_mp3")
    parser.add_argument("--data_dir", default="./data_psytrance",
                        help="ACE-Step data directory for training (separate from default ./data)")
    parser.add_argument("--repeat_count", type=int, default=50,
                        help="Dataset repeat count (lower for large datasets)")
    parser.add_argument("--output_name", default="psytrance_lora_dataset",
                        help="Output HuggingFace dataset name")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    data_dir = Path(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Find all complete triplets (mp3 + prompt + lyrics)
    mp3_files = sorted(input_dir.glob("*.mp3"))
    transferred = 0
    skipped = 0

    print(f"Found {len(mp3_files)} MP3 files in {input_dir}")

    for mp3 in tqdm(mp3_files, desc="Transferring"):
        stem = mp3.stem
        prompt_file = input_dir / f"{stem}_prompt.txt"
        lyrics_file = input_dir / f"{stem}_lyrics.txt"

        if not prompt_file.exists():
            print(f"  [SKIP] No prompt for {stem}")
            skipped += 1
            continue

        if not lyrics_file.exists():
            print(f"  [SKIP] No lyrics for {stem}")
            skipped += 1
            continue

        # Transfer files
        transfer = shutil.copy2 if args.copy else shutil.move
        for src in [mp3, prompt_file, lyrics_file]:
            dst = data_dir / src.name
            if dst.exists() and dst != src:
                # Don't overwrite
                continue
            transfer(str(src), str(dst))

        transferred += 1

    print(f"Transferred {transferred} tracks to {data_dir} ({skipped} skipped)")

    if transferred == 0:
        print("No files transferred. Skipping dataset creation.")
        return

    # Create HuggingFace dataset
    print(f"Creating HuggingFace dataset with repeat_count={args.repeat_count}...")
    create_dataset(
        data_dir=str(data_dir),
        repeat_count=args.repeat_count,
        output_name=args.output_name,
    )
    print(f"Dataset saved to ./{args.output_name}")
    print(f"\nReady for training! Run:")
    print(f"  python trainer.py \\")
    print(f"    --dataset_path ./{args.output_name} \\")
    print(f"    --exp_name psytrance_lora \\")
    print(f"    --lora_config_path config/zh_rap_lora_config.json \\")
    print(f"    --learning_rate 1e-4 \\")
    print(f"    --max_steps 200000")


if __name__ == "__main__":
    main()
