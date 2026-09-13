"""
Downloads the two MaleCNS v1.0 connectome files we need, straight from
Google's public storage bucket over plain HTTPS. No API key, no account,
no neuPrint token -- this is a static, public, CC-BY-licensed dataset.

Total download: ~1.1 GB. Run this once. After this, everything in the
project works fully offline.
"""
import pathlib
import sys

import requests

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

FILES = {
    # cell type / class / side annotations for every neuron -- 13 MB
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": f"{BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    # the full segment-to-segment connection weight graph -- 1.1 GB
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": f"{BASE}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}


def download(url: str, dest: pathlib.Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"[downloading] {dest.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    sys.stdout.write(f"\r  {written / 1e6:8.1f} MB / {total / 1e6:.1f} MB ({pct:5.1f}%)")
                    sys.stdout.flush()
        tmp.rename(dest)
        print()


def main() -> None:
    for filename, url in FILES.items():
        download(url, DATA_DIR / filename)
    print("\nDone. Data lives in:", DATA_DIR)
    print("Nothing else in this project ever calls out to the internet again.")


if __name__ == "__main__":
    main()
