#!/usr/bin/env python3
"""Download and unpack the Allen Telescope Array RFI survey from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import ssl
import sys
import tarfile
from urllib.request import Request, urlopen

import certifi


RECORD_ID = 8242048
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK_BYTES = 1024 * 1024
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def record_files() -> list[dict[str, object]]:
    request = Request(RECORD_API, headers={"User-Agent": "Scientific-Workspace-Browser-Examples/0.1"})
    with urlopen(request, context=TLS_CONTEXT) as response:
        payload = json.load(response)
    return sorted((item for item in payload["files"] if str(item["key"]).endswith(".sigmf")), key=lambda item: item["key"])


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(file: dict[str, object], destination: Path) -> None:
    expected = str(file["checksum"]).removeprefix("md5:")
    if destination.is_file() and md5(destination) == expected:
        print(f"Using verified {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(str(file["links"]["self"]), headers={"User-Agent": "Scientific-Workspace-Browser-Examples/0.1"})
    size = int(file["size"])
    received = 0
    with urlopen(request, context=TLS_CONTEXT) as response, temporary.open("wb") as output:
        while chunk := response.read(CHUNK_BYTES):
            output.write(chunk)
            received += len(chunk)
            print(f"\r{destination.name}: {received / size:6.1%}", end="", flush=True)
    print()
    if md5(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {destination.name}")
    temporary.replace(destination)


def unpack(archive: Path, output: Path) -> None:
    root = output.resolve()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (output / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not supported: {member.name}")
        bundle.extractall(output, members=members)
    print(f"Unpacked {archive.name}")


def is_unpacked(file: dict[str, object], output: Path) -> bool:
    directory = output / Path(str(file["key"])).stem
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
        for file in files:
            print(f"{file['key']}  {int(file['size']) / 1e6:.1f} MB  {file['checksum']}")
        return
    if args.first:
        files = files[:1]
    total_gb = sum(int(file["size"]) for file in files) / 1e9
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(files)} SigMF archive(s), {total_gb:.2f} GB total, to {args.output.resolve()}")
    for file in files:
        if is_unpacked(file, args.output):
            print(f"Using unpacked {file['key']}")
            continue
        archive = args.output / str(file["key"])
        download(file, archive)
        unpack(archive, args.output)
        if not args.keep_archives:
            archive.unlink()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
