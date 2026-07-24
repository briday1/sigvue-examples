#!/usr/bin/env python3
"""Download and unpack the Allen Telescope Array RFI survey from Zenodo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import sys
from urllib.request import Request, urlopen

import certifi
from sigvue.helpers import RemoteFile, download_file, safe_extract_tar


RECORD_ID = 8242048
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_BYTES = 1024 * 1024
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def record_files() -> list[RemoteFile]:
    request = Request(RECORD_API, headers={"User-Agent": "Sigvue-Examples/0.3"})
    with urlopen(request, context=TLS_CONTEXT) as response:
        payload = json.load(response)
    return sorted(
        (
            RemoteFile(
                url=str(item["links"]["self"]),
                filename=str(item["key"]),
                size=int(item["size"]),
                checksum=str(item["checksum"]),
            )
            for item in payload["files"]
            if str(item["key"]).endswith(".sigmf")
        ),
        key=lambda remote: remote.filename,
    )


def _progress(filename: str):
    def report(received: int, total: int | None) -> None:
        status = (
            f"{received / total:6.1%}"
            if total
            else f"{received / 1_000_000:.1f} MB"
        )
        print(f"\r{filename}: {status}", end="", flush=True)

    return report


def is_unpacked(remote: RemoteFile, output: Path) -> bool:
    directory = output / Path(remote.filename).stem
    return directory.is_dir() and any(directory.rglob("*.sigmf-meta")) and any(directory.rglob("*.sigmf-data"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/radio-astronomy"))
    parser.add_argument("--first", action="store_true", help="download only the first ~559 MB recording")
    parser.add_argument("--list", action="store_true", help="list remote files without downloading")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    files = record_files()
    if args.list:
        for remote in files:
            print(
                f"{remote.filename}  {remote.size / 1e6:.1f} MB  "
                f"{remote.checksum}"
            )
        return
    if args.first:
        files = files[:1]
    total_gb = sum(remote.size or 0 for remote in files) / 1e9
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} SigMF archive(s), {total_gb:.2f} GB total, to {args.output.resolve()}")
    for remote in files:
        if is_unpacked(remote, args.output):
            print(f"Using unpacked {remote.filename}")
            continue
        archive = download_file(
            remote,
            args.output,
            user_agent="Sigvue-Examples/0.3",
            chunk_bytes=CHUNK_BYTES,
            progress=_progress(remote.filename),
            tls_context=TLS_CONTEXT,
        )
        print()
        safe_extract_tar(archive, args.output)
        print(f"Unpacked {archive.name}")
        if not args.keep_archives:
            archive.unlink()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
