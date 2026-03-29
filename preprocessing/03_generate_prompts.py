"""
Step 3: Generate trance-specific prompt tags for each MP3 chunk.
Uses Essentia for BPM/key + CLAP for zero-shot trance-specific tagging.

Usage:
    python preprocessing/03_generate_prompts.py --input_dir ./data/psytrance/chunked_mp3
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Trance-specific tag vocabulary for CLAP zero-shot classification
# ---------------------------------------------------------------------------

TRANCE_TAGS = {
    "subgenre": [
        "uplifting trance", "progressive trance", "psytrance",
        "psychedelic trance", "goa trance", "vocal trance",
        "tech trance", "acid trance", "hard trance",
        "full-on psytrance", "dark psytrance", "forest psytrance",
        "melodic trance", "minimal psytrance", "suomi trance",
    ],
    "sounds": [
        "supersaw lead", "acid bassline", "pluck synth",
        "atmospheric pad", "rolling bassline", "arpeggio synth",
        "trance gate", "white noise sweep", "sub bass",
        "TB-303 acid", "detuned lead", "layered synths",
        "resonant filter sweep", "saw lead", "squelchy bass",
    ],
    "energy": [
        "euphoric", "driving", "hypnotic", "dark",
        "atmospheric", "dreamy", "aggressive", "emotional",
        "peak time", "melodic", "intense", "uplifting",
        "meditative", "chaotic", "tribal",
    ],
    "vocals": [
        "female vocal", "male vocal", "vocal chops",
        "ethereal vocal", "no vocals", "instrumental",
        "spoken word sample", "vocal pad", "chanted vocal",
    ],
    "production": [
        "heavy reverb", "sidechain compression", "delay effects",
        "stereo wide", "punchy kick", "clean mix",
        "lo-fi texture", "crisp highs", "deep low end",
    ],
}

# Flatten for CLAP scoring
ALL_CANDIDATES = []
CATEGORY_RANGES = {}
idx = 0
for cat, tags in TRANCE_TAGS.items():
    CATEGORY_RANGES[cat] = (idx, idx + len(tags))
    ALL_CANDIDATES.extend(tags)
    idx += len(tags)

# How many tags to pick per category
TAGS_PER_CATEGORY = {
    "subgenre": 2,
    "sounds": 3,
    "energy": 2,
    "vocals": 1,
    "production": 2,
}


def load_essentia():
    """Load Essentia extractors."""
    import essentia.standard as es
    return es


def load_clap():
    """Load LAION CLAP model."""
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    model.load_ckpt()
    return model


def analyze_essentia(mp3_path, es):
    """Extract BPM and key using Essentia."""
    tags = []
    try:
        audio = es.MonoLoader(filename=mp3_path, sampleRate=44100)()

        # BPM
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        result = rhythm_extractor(audio)
        bpm = round(result[0])
        if 60 <= bpm <= 200:
            tags.append(f"{bpm} bpm")

        # Key
        key_extractor = es.KeyExtractor()
        key, scale, key_strength = key_extractor(audio)
        if key_strength > 0.3:
            tags.append(f"{key} {scale}")

    except Exception as e:
        print(f"  [WARN] Essentia failed on {mp3_path}: {e}")

    return tags


def analyze_clap_batch(mp3_paths, clap_model, text_embeddings):
    """Score all trance tags for a batch of MP3s using CLAP."""
    import torch

    # Get audio embeddings
    audio_embeddings = clap_model.get_audio_embedding_from_filelist(
        x=mp3_paths, use_tensor=True
    )

    # Compute cosine similarity
    audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
    similarities = (audio_embeddings @ text_embeddings.T).detach().cpu().numpy()

    results = []
    for i in range(len(mp3_paths)):
        scores = similarities[i]
        tags = []
        for cat, (start, end) in CATEGORY_RANGES.items():
            cat_scores = scores[start:end]
            cat_tags = ALL_CANDIDATES[start:end]
            n_pick = TAGS_PER_CATEGORY.get(cat, 2)

            # Pick top-N tags from this category
            top_indices = np.argsort(cat_scores)[::-1][:n_pick]
            for j in top_indices:
                # Only include tags with reasonable scores
                if cat_scores[j] > 0.1:
                    tags.append(cat_tags[j])

        results.append(tags)

    return results


def write_prompt(mp3_path, tags):
    """Write prompt tags to a _prompt.txt file next to the MP3."""
    prompt_path = mp3_path.replace(".mp3", "_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(", ".join(tags))
    return prompt_path


def write_empty_lyrics(mp3_path):
    """Write an empty _lyrics.txt file (trance is mostly instrumental)."""
    lyrics_path = mp3_path.replace(".mp3", "_lyrics.txt")
    if not os.path.exists(lyrics_path):
        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write("")
    return lyrics_path


def main():
    parser = argparse.ArgumentParser(description="Generate trance prompts for MP3 chunks")
    parser.add_argument("--input_dir", default="./data/psytrance/chunked_mp3")
    parser.add_argument("--clap_batch_size", type=int, default=16,
                        help="Batch size for CLAP inference")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already have a _prompt.txt")
    args = parser.parse_args()

    mp3_files = sorted(Path(args.input_dir).glob("*.mp3"))
    if args.skip_existing:
        mp3_files = [
            f for f in mp3_files
            if not os.path.exists(str(f).replace(".mp3", "_prompt.txt"))
        ]

    print(f"Processing {len(mp3_files)} MP3 files")

    if not mp3_files:
        print("No files to process.")
        return

    # Load models
    print("Loading Essentia...")
    es = load_essentia()

    print("Loading CLAP model...")
    clap_model = load_clap()

    # Phase 1: Essentia analysis (per-file, fast)
    print("Phase 1: Essentia BPM/key analysis...")
    essentia_tags = {}
    for mp3 in tqdm(mp3_files, desc="Essentia"):
        essentia_tags[str(mp3)] = analyze_essentia(str(mp3), es)

    # Phase 2: CLAP analysis (batched, GPU-accelerated)
    print("Phase 2: CLAP zero-shot tagging...")
    import torch
    # Pre-compute text embeddings once (shared across all batches)
    text_embeddings = clap_model.get_text_embedding(ALL_CANDIDATES, use_tensor=True)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    clap_tags = {}
    mp3_paths = [str(f) for f in mp3_files]

    for batch_start in tqdm(range(0, len(mp3_paths), args.clap_batch_size), desc="CLAP"):
        batch = mp3_paths[batch_start:batch_start + args.clap_batch_size]
        try:
            batch_results = analyze_clap_batch(batch, clap_model, text_embeddings)
            for path, tags in zip(batch, batch_results):
                clap_tags[path] = tags
        except Exception as e:
            print(f"  [WARN] CLAP batch failed ({len(batch)} files): {e}")
            # Fall back to processing files individually
            for path in batch:
                try:
                    single_result = analyze_clap_batch([path], clap_model, text_embeddings)
                    clap_tags[path] = single_result[0]
                except Exception as e2:
                    print(f"  [WARN] CLAP failed on {path}: {e2}")
                    clap_tags[path] = []

    # Phase 3: Combine and write
    print("Phase 3: Writing prompt files...")
    for mp3 in tqdm(mp3_files, desc="Writing"):
        path = str(mp3)
        tags = essentia_tags.get(path, []) + clap_tags.get(path, [])

        if not tags:
            tags = ["psytrance", "electronic"]

        write_prompt(path, tags)
        write_empty_lyrics(path)

    print(f"Done. Generated prompts for {len(mp3_files)} files in {args.input_dir}")


if __name__ == "__main__":
    main()
