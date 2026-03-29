"""
Step 1: Scrape MP3 files from the goa_psytrance archive.
Recursively crawls Apache directory listings and downloads MP3s.

Usage:
    python preprocessing/01_scrape.py --output_dir ./data/psytrance/raw_mp3 --max_files 500
"""

import argparse
import os
import time
import re
from urllib.parse import urljoin, unquote
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_links(url, session, base_url):
    """Parse an Apache directory listing page and return (dirs, mp3s).
    Only returns links that stay within base_url to prevent upward traversal.
    """
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return [], []

    soup = BeautifulSoup(resp.text, "html.parser")
    dirs = []
    mp3s = []

    for a in soup.find_all("a"):
        href = a.get("href", "")
        # Skip parent dir, sort links, and absolute paths
        if href in ("", "/") or href.startswith("?") or href.startswith("/"):
            continue
        # Skip parent directory links
        if href.startswith(".."):
            continue

        full_url = urljoin(url, href)

        # Safety: only follow links within the base URL
        if not full_url.startswith(base_url):
            continue

        if href.endswith("/"):
            dirs.append(full_url)
        elif href.lower().endswith(".mp3"):
            mp3s.append(full_url)

    return dirs, mp3s


def crawl(base_url, session, max_files=None, max_depth=3):
    """Recursively crawl directories and collect MP3 URLs."""
    # Ensure base_url ends with / for prefix matching
    if not base_url.endswith("/"):
        base_url = base_url + "/"

    collected = []
    collected_set = set()
    visited = set()
    queue = [(base_url, 0)]

    pbar = tqdm(desc="Crawling directories", unit=" dirs")

    while queue:
        if max_files and len(collected) >= max_files:
            break

        url, depth = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        dirs, mp3s = get_links(url, session, base_url)
        for mp3_url in mp3s:
            if mp3_url not in collected_set:
                collected.append(mp3_url)
                collected_set.add(mp3_url)
        pbar.set_postfix(mp3s=len(collected))
        pbar.update(1)

        if max_files and len(collected) >= max_files:
            collected = collected[:max_files]
            break

        for d in dirs:
            if d not in visited:
                queue.append((d, depth + 1))

        # Be polite
        time.sleep(0.2)

    pbar.close()
    return collected


def sanitize_filename(url, base_url):
    """Create a safe filename from a URL, incorporating parent dir to avoid collisions."""
    # Get the path relative to base_url for uniqueness
    rel_path = unquote(url.replace(base_url, "", 1))
    # Replace path separators and problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '_', rel_path)
    # Collapse multiple underscores/spaces
    name = re.sub(r'[\s_]+', '_', name).strip('_')
    # Ensure it ends with .mp3
    if not name.lower().endswith(".mp3"):
        name += ".mp3"
    return name


def download_mp3s(urls, output_dir, session, base_url):
    """Download MP3 files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0

    for url in tqdm(urls, desc="Downloading MP3s"):
        filename = sanitize_filename(url, base_url)
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            downloaded += 1
            continue

        try:
            resp = session.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded += 1
        except requests.RequestException as e:
            print(f"  [WARN] Failed to download {url}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            continue

        # Be polite
        time.sleep(0.3)

    print(f"Downloaded {downloaded}/{len(urls)} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Scrape MP3s from goa_psytrance archive")
    parser.add_argument("--base_url", default="http://b1g-arch1ve.buho.ch/goa_psytrance/")
    parser.add_argument("--output_dir", default="./data/psytrance/raw_mp3")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max number of MP3s to download (None = all)")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Max directory recursion depth")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "ACE-Step-DataPrep/1.0"})

    # Normalize base URL to always end with /
    base_url = args.base_url
    if not base_url.endswith("/"):
        base_url += "/"

    print(f"Crawling {base_url} ...")
    mp3_urls = crawl(base_url, session, args.max_files, args.max_depth)
    print(f"Found {len(mp3_urls)} MP3 files")

    if mp3_urls:
        download_mp3s(mp3_urls, args.output_dir, session, base_url)


if __name__ == "__main__":
    main()
