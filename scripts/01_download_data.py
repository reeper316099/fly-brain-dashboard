"""
Downloads the MaleCNS v1.0 connectome files we need, straight from Google's
public storage bucket over plain HTTPS. No API key, no account, no neuPrint
token -- this is a static, public, CC-BY-licensed dataset.

Files (default set, ~560 MB total):
  - body-annotations-*.feather          cell type / class / side per neuron   (14 MB)
  - body-neurotransmitters-*.feather    predicted neurotransmitter per neuron (43 MB)
  - connectome-weights-*-significant-only.feather
                                        body-to-body connection weights,
                                        strong ("significant") edges only    (500 MB)

Pass --full-weights to fetch the complete 1.05 GB weight table instead. It
includes every 1- and 2-synapse edge as well; for an 800-neuron hub subgraph
that makes no visible difference, so the smaller file is the default.

Downloads are resumable: an interrupted transfer leaves a *.part file that is
picked up where it left off the next time you run this. Run this once. After
this, everything in the project works fully offline.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import requests

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS_SIGNIFICANT = "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather"
WEIGHTS_FULL = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

CHUNK = 1 << 20  # 1 MB
MAX_ATTEMPTS = 5


def _fmt_mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        return int(r.headers["content-length"])
    except (requests.RequestException, KeyError, ValueError):
        return None


def download(url: str, dest: pathlib.Path) -> None:
    total = remote_size(url)

    if dest.exists():
        if total is None or dest.stat().st_size == total:
            print(f"[skip] {dest.name} already downloaded ({_fmt_mb(dest.stat().st_size)})")
            return
        print(f"[redo] {dest.name} is {_fmt_mb(dest.stat().st_size)} but should be "
              f"{_fmt_mb(total)} -- re-downloading")
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[downloading] {dest.name}" + (f"  ({_fmt_mb(total)})" if total else ""))

    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have > total:
            tmp.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if have and r.status_code != 206:
                    # Server ignored the Range header: start over.
                    have = 0
                    tmp.unlink(missing_ok=True)
                r.raise_for_status()
                mode = "ab" if have else "wb"
                written = have
                last_print = 0.0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        f.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if total and now - last_print > 0.2:
                            pct = written / total * 100
                            sys.stdout.write(f"\r  {_fmt_mb(written):>12} / {_fmt_mb(total)} ({pct:5.1f}%)")
                            sys.stdout.flush()
                            last_print = now
                if total:
                    sys.stdout.write(f"\r  {_fmt_mb(written):>12} / {_fmt_mb(total)} (100.0%)\n")
                else:
                    print(f"  {_fmt_mb(written)}")
            if total is not None and tmp.stat().st_size != total:
                raise IOError(f"size mismatch: got {tmp.stat().st_size}, expected {total}")
            tmp.replace(dest)
            return
        except (requests.RequestException, IOError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise SystemExit(f"\nGave up on {dest.name} after {MAX_ATTEMPTS} attempts: {exc}")
            wait = 2 ** attempt
            print(f"\n  [retry {attempt}/{MAX_ATTEMPTS - 1}] {exc} -- resuming in {wait}s")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full-weights", action="store_true",
                        help=f"download the complete weight table ({WEIGHTS_FULL}, ~1.05 GB) "
                             f"instead of the significant-only one (~500 MB)")
    parser.add_argument("--data-dir", type=pathlib.Path, default=DATA_DIR,
                        help=f"where to put the files (default: {DATA_DIR})")
    args = parser.parse_args()

    files = [ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS_FULL if args.full_weights else WEIGHTS_SIGNIFICANT]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in files:
            download(f"{BASE}/{name}", args.data_dir / name)
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted. Re-run to resume where this left off.")

    print("\nDone. Data lives in:", args.data_dir)
    print("Nothing else in this project ever calls out to the internet again.")
    print("Next: python scripts/02_build_subgraph.py")


if __name__ == "__main__":
    main()
