"""Download the Unsplash Lite dataset images for FlowBender preprocessing.

Usage:
    python data/download_unsplash.py --data-root /path/to/unsplash_25k
"""

import csv
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cyclopts
from tqdm import tqdm

LITE_URL = "https://unsplash.com/data/lite/latest"
_USER_AGENT = "flowbender-dataset-downloader/1.0"
_WIDTH = 1024  # requested image width in pixels (matches the 1024 preprocessing)
_NUM_WORKERS = 16  # concurrent download threads
_TEST_IDS_FILE = Path(__file__).with_name("unsplash_test_ids.txt")

app = cyclopts.App(help="Download Unsplash Lite images for FlowBender preprocessing.")


def _urlretrieve(url: str, dest: Path, timeout: int = 60) -> None:
    """Download ``url`` to ``dest`` atomically (via a temp file + rename)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    tmp.rename(dest)


def ensure_metadata(metadata_dir: Path) -> list[Path]:
    """Return the ``photos.tsv*`` files, downloading + extracting the Lite zip if needed."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tsv_files = sorted(metadata_dir.glob("photos.tsv*"))
    if tsv_files:
        return tsv_files

    zip_path = metadata_dir / "unsplash-lite.zip"
    if not zip_path.exists():
        print(f"Downloading Lite metadata (~700MB) to {zip_path} ...")
        _urlretrieve(LITE_URL, zip_path, timeout=600)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(metadata_dir)

    tsv_files = sorted(metadata_dir.glob("photos.tsv*"))
    if not tsv_files:
        sys.exit(f"No photos.tsv* found in {metadata_dir} after extraction.")
    return tsv_files


def read_photos(tsv_files: list[Path]) -> list[tuple[str, str]]:
    """Read (photo_id, photo_image_url) rows from the photos TSV table(s)."""
    rows: list[tuple[str, str]] = []
    for path in tsv_files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                url = row.get("photo_image_url")
                pid = row.get("photo_id")
                if pid and url:
                    rows.append((pid, url))
    return rows


def load_test_ids() -> set[str]:
    """Load the fixed set of ``test`` photo ids from the committed manifest."""
    if not _TEST_IDS_FILE.exists():
        sys.exit(f"Missing fixed split manifest: {_TEST_IDS_FILE}")
    return {line.strip() for line in _TEST_IDS_FILE.read_text().splitlines() if line.strip()}


@app.default
def main(*, data_root: str) -> None:
    """Download Unsplash Lite images into ``{data_root}/data/{train,test}``.

    Args:
        data_root: Dataset root (matches ``data_root`` in your user config).
            Raw images are written to ``{data_root}/data/{train,test}`` and the
            Lite metadata is cached under ``{data_root}/unsplash-lite``.
    """
    root = Path(data_root)
    tsv_files = ensure_metadata(root / "unsplash-lite")
    photos = read_photos(tsv_files)
    print(f"Found {len(photos)} photos in metadata.")

    test_ids = load_test_ids()

    # Build the download work list, skipping images that already exist.
    jobs: list[tuple[str, Path]] = []
    for photo_id, url in photos:
        split = "test" if photo_id in test_ids else "train"
        dest = root / "data" / split / f"{photo_id}.jpg"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((f"{url}?w={_WIDTH}&fm=jpg", dest))

    print(f"{len(photos) - len(jobs)} already present, downloading {len(jobs)} ...")

    failures = 0

    def _download(job: tuple[str, Path]) -> bool:
        url, dest = job
        try:
            _urlretrieve(url, dest)
            return True
        except Exception as e:  # noqa: BLE001 - keep going on individual failures
            tqdm.write(f"Failed {dest.name}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as pool:
        futures = [pool.submit(_download, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            if not fut.result():
                failures += 1

    print(f"Done. {len(jobs) - failures} downloaded, {failures} failed.")
    print(f"Images under: {root}/data/train and {root}/data/test")


if __name__ == "__main__":
    app()
