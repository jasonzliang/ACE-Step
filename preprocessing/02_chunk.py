"""
Step 2: Split MP3 files into 240-second chunks.
Skips files that are already <= 240 seconds.

Usage:
    python preprocessing/02_chunk.py --input_dir ./preprocessing/raw_mp3 --output_dir ./preprocessing/chunked_mp3
"""

import argparse
import os
from pathlib import Path
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from tqdm import tqdm


MAX_DURATION_MS = 240 * 1000  # 240 seconds
MIN_DURATION_MS = 10 * 1000   # Skip chunks shorter than 10 seconds


def chunk_mp3(input_path, output_dir):
    """Split a single MP3 into 240s chunks. Returns list of output paths."""
    try:
        audio = AudioSegment.from_mp3(input_path)
    except (CouldntDecodeError, Exception) as e:
        print(f"  [WARN] Could not decode {input_path}: {e}")
        return []

    stem = Path(input_path).stem
    duration_ms = len(audio)
    outputs = []

    if duration_ms <= MAX_DURATION_MS:
        # No splitting needed, just copy
        out_path = os.path.join(output_dir, f"{stem}.mp3")
        if not os.path.exists(out_path):
            audio.export(out_path, format="mp3", bitrate="192k")
        outputs.append(out_path)
    else:
        # Split into chunks
        chunk_idx = 0
        for start_ms in range(0, duration_ms, MAX_DURATION_MS):
            chunk = audio[start_ms:start_ms + MAX_DURATION_MS]

            # Skip very short trailing chunks
            if len(chunk) < MIN_DURATION_MS:
                continue

            out_path = os.path.join(output_dir, f"{stem}_chunk{chunk_idx:02d}.mp3")
            if not os.path.exists(out_path):
                chunk.export(out_path, format="mp3", bitrate="192k")
            outputs.append(out_path)
            chunk_idx += 1

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Chunk MP3s into 240s segments")
    parser.add_argument("--input_dir", default="./preprocessing/raw_mp3")
    parser.add_argument("--output_dir", default="./preprocessing/chunked_mp3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mp3_files = sorted(Path(args.input_dir).glob("*.mp3"))
    print(f"Found {len(mp3_files)} MP3 files to process")

    total_chunks = 0
    for mp3 in tqdm(mp3_files, desc="Chunking"):
        outputs = chunk_mp3(str(mp3), args.output_dir)
        total_chunks += len(outputs)

    print(f"Produced {total_chunks} chunks from {len(mp3_files)} files in {args.output_dir}")


if __name__ == "__main__":
    main()
