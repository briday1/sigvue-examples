#!/usr/bin/env python3
"""Download pinned public NOAA NEXRAD Level III example scans."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from sigvue.helpers import RemoteFile, download_file


NEXRAD_BASE_URL = "https://unidata-nexrad-level3.s3.amazonaws.com"
USER_AGENT = "Sigvue-Examples/0.3"
MANIFEST_PATH = Path(__file__).with_name("weather_radar_manifest.tsv")


def _load_manifest(path: Path = MANIFEST_PATH) -> dict[str, tuple[RemoteFile, ...]]:
    sequences: dict[str, list[RemoteFile]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        if rows.fieldnames != ["radar", "filename", "size", "sha256"]:
            raise ValueError(f"Invalid weather-radar manifest columns: {path}")
        for row in rows:
            radar = row["radar"]
            filename = row["filename"]
            if not filename.startswith(f"{radar}_N0B_"):
                raise ValueError(f"Invalid weather-radar manifest entry: {filename}")
            sequences.setdefault(radar, []).append(
                RemoteFile(
                    url=f"{NEXRAD_BASE_URL}/{filename}",
                    filename=filename,
                    size=int(row["size"]),
                    checksum=f"sha256:{row['sha256']}",
                )
            )
    if not sequences:
        raise ValueError(f"Empty weather-radar manifest: {path}")
    return {radar: tuple(scans) for radar, scans in sequences.items()}


WEATHER_RADAR_SEQUENCES = _load_manifest()
DEFAULT_RADARS = tuple(WEATHER_RADAR_SEQUENCES)
WEATHER_RADAR_FILES = tuple(
    remote for radar in DEFAULT_RADARS for remote in WEATHER_RADAR_SEQUENCES[radar]
)
WEATHER_RADAR_FILE = next(
    remote
    for remote in WEATHER_RADAR_SEQUENCES["TLX"]
    if remote.filename == "TLX_N0B_2024_05_20_03_10_54"
)


def _progress(filename: str):
    def report(received: int, total: int | None) -> None:
        status = (
            f"{received / total:6.1%}" if total else f"{received / 1_000_000:.1f} MB"
        )
        print(f"\r{filename}: {status}", end="", flush=True)

    return report


def _download_scan(remote: RemoteFile, output: str | Path) -> Path:
    return download_file(
        remote,
        Path(output).expanduser(),
        user_agent=USER_AGENT,
        progress=_progress(remote.filename),
    )


def download_weather_radar(output: str | Path) -> Path:
    """Download the original pinned scan for API compatibility."""
    return _download_scan(WEATHER_RADAR_FILE, output)


def download_weather_radar_scans(
    output: str | Path,
    radars: tuple[str, ...] = DEFAULT_RADARS,
    *,
    workers: int = 8,
) -> tuple[Path, ...]:
    """Materialize selected checksum-verified N0B sequences in ``output``."""
    unknown = tuple(radar for radar in radars if radar not in WEATHER_RADAR_SEQUENCES)
    if unknown:
        raise ValueError(f"Unknown weather radar: {', '.join(unknown)}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    remotes = tuple(
        remote
        for radar in radars
        for remote in WEATHER_RADAR_SEQUENCES[radar]
    )
    if workers == 1:
        return tuple(_download_scan(remote, output) for remote in remotes)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(partial(_download_scan, output=output), remotes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/weather-radar"),
    )
    parser.add_argument(
        "--radars",
        nargs="+",
        choices=DEFAULT_RADARS,
        default=DEFAULT_RADARS,
        help="radar sequences to download (default: all pinned radars)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="concurrent downloads (default: 8)",
    )
    args = parser.parse_args()
    paths = download_weather_radar_scans(
        args.output,
        tuple(args.radars),
        workers=args.workers,
    )
    print()
    print(
        f"Ready: {len(paths)} scans from {', '.join(args.radars)} "
        f"in {paths[0].parent.resolve()}"
    )


if __name__ == "__main__":
    main()
