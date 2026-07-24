#!/usr/bin/env python3
"""Download the public LTE SigMF recordings used by the examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from sigvue.helpers import RemoteFile, download_file


BASE_URL = "http://nas.destevez.net/~daniel/LTE"
USER_AGENT = "Sigvue-Examples/0.3"
LTE_MANIFEST = {
    "lte/downlink": (
        RemoteFile(
            f"{BASE_URL}/LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta",
            "LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta",
            size=1_022,
            checksum=(
                "sha256:"
                "2f591862e15ef67f4f7aceda3457320db839e4de424d793edf3d30a971479b45"
            ),
        ),
        RemoteFile(
            f"{BASE_URL}/LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-data",
            "LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-data",
            size=122_880_000,
            checksum=(
                "sha256:"
                "d2dfecfec0cdf346d2264ae61ddc80ceba373893b0e8a2d6ebafc87b7215f26c"
            ),
        ),
    ),
    "lte/uplink": (
        RemoteFile(
            f"{BASE_URL}/LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-meta",
            "LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-meta",
            size=799,
            checksum=(
                "sha256:"
                "4593e878261b5f9040195f854d906b9197f1dc0ccf8a84b2d6634b871051eb91"
            ),
        ),
        RemoteFile(
            f"{BASE_URL}/LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-data",
            "LTE_uplink_847MHz_2022-01-30_30720ksps.sigmf-data",
            size=108_871_680,
            checksum=(
                "sha256:"
                "85d3cf17552581eae161491e9a633cce056a9019495805c526f3975592d96e2a"
            ),
        ),
    ),
}


def _progress(filename: str):
    def report(received: int, total: int | None) -> None:
        status = (
            f"{received / total:6.1%}"
            if total
            else f"{received / 1_000_000:.1f} MB"
        )
        print(f"\r{filename}: {status}", end="", flush=True)

    return report


def download_lte_recordings(data_root: Path) -> tuple[Path, ...]:
    """Materialize every file in the LTE manifest under its workspace."""
    downloaded = []
    for relative_directory, files in LTE_MANIFEST.items():
        destination = data_root / relative_directory
        for remote in files:
            print(f"Preparing {remote.filename}")
            path = download_file(
                remote,
                destination,
                user_agent=USER_AGENT,
                progress=_progress(remote.filename),
                preserve_existing=remote.filename.endswith(".sigmf-meta"),
            )
            print(f"\rReady {path}")
            downloaded.append(path)
    return tuple(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    download_lte_recordings(args.output.expanduser())
    print(f"LTE SigMF recordings are available under {args.output}")


if __name__ == "__main__":
    main()
