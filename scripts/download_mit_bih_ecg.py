#!/usr/bin/env python3
"""Download pinned MIT-BIH records used by the annotated ECG example."""

from __future__ import annotations

import argparse
from pathlib import Path

from sigvue.helpers import RemoteFile
from sigvue_examples.plugins.wfdb import download_wfdb_record


PHYSIONET_BASE_URL = "https://physionet.org/files/mitdb/1.0.0"
USER_AGENT = "Sigvue-Examples/0.3"


def _record_file(
    record: str,
    extension: str,
    size: int,
    digest: str,
) -> RemoteFile:
    filename = f"{record}.{extension}"
    return RemoteFile(
        url=f"{PHYSIONET_BASE_URL}/{filename}",
        filename=filename,
        size=size,
        checksum=f"sha256:{digest}",
    )


MIT_BIH_RECORDS = {
    "100": (
        _record_file(
            "100",
            "hea",
            143,
            "60ebc904c7bf3e04d142638d3fd5c903e8dc9f10f1ea3264e07926aa089ee75e",
        ),
        _record_file(
            "100",
            "dat",
            1_950_000,
            "b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639",
        ),
        _record_file(
            "100",
            "atr",
            4_558,
            "8d8a5349fb16638ebbf649f1779d12e96d91b736b2aafe59db43719ae583d471",
        ),
    ),
    "101": (
        _record_file(
            "101",
            "hea",
            131,
            "d5f02fbe8673fa05465442191b98ca0d28a1670e7ef0e83fb9ef8723113a311c",
        ),
        _record_file(
            "101",
            "dat",
            1_950_000,
            "698d1ea6f472d23ca50317c72c96cda2698badd8578220ed0380cdf241e39006",
        ),
        _record_file(
            "101",
            "atr",
            3_768,
            "441cdd6486cfdf4c53e344d1048ab81296773e7058d53069022adc68543a0663",
        ),
    ),
    "200": (
        _record_file(
            "200",
            "hea",
            306,
            "9e0c2ff5b790cf624deab0ccb8a9f211a9e29a748d8197da3c1ee7c5b596b40c",
        ),
        _record_file(
            "200",
            "dat",
            1_950_000,
            "a9e203b3807b9fcd3647cde03444437cb8eec7f5128a8eb413edafb394272e0f",
        ),
        _record_file(
            "200",
            "atr",
            8_114,
            "f9624a11696760427d75314a78c31f89ca3c446af855890c8f8b66cddd8b3a3f",
        ),
    ),
    "207": (
        _record_file(
            "207",
            "hea",
            546,
            "7645d488d4c304760aae0a709193ffa13692c317b551bcbdcbb37011032178d8",
        ),
        _record_file(
            "207",
            "dat",
            1_950_000,
            "139f99250366fbf347cba4d8ea1fbe788f98cb1b93b70f50ccba70122d908605",
        ),
        _record_file(
            "207",
            "atr",
            4_958,
            "cceb64d68033a277d2d5669458d49258295102dd3e20614a0ca7d63b67009404",
        ),
    ),
}
DEFAULT_RECORDS = tuple(MIT_BIH_RECORDS)
MIT_BIH_RECORD_100 = MIT_BIH_RECORDS["100"]


def _progress(filename: str):
    def report(received: int, total: int | None) -> None:
        status = (
            f"{received / total:6.1%}" if total else f"{received / 1_000_000:.1f} MB"
        )
        print(f"\r{filename}: {status}", end="", flush=True)

    return report


def download_mit_bih_record(
    output: str | Path,
    record: str = "100",
) -> tuple[Path, ...]:
    """Download and verify all companion files for one pinned record."""
    try:
        manifest = MIT_BIH_RECORDS[record]
    except KeyError as error:
        choices = ", ".join(MIT_BIH_RECORDS)
        raise ValueError(
            f"Unknown MIT-BIH record {record!r}; choose {choices}"
        ) from error
    return download_wfdb_record(
        manifest,
        Path(output).expanduser(),
        user_agent=USER_AGENT,
        progress_factory=_progress,
    )


def download_mit_bih_records(
    output: str | Path,
    records: tuple[str, ...] = DEFAULT_RECORDS,
) -> tuple[Path, ...]:
    """Download a selection of pinned records into one discoverable directory."""
    return tuple(
        path for record in records for path in download_mit_bih_record(output, record)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ecg/mit-bih"),
        help="directory that will contain the WFDB companion files",
    )
    parser.add_argument(
        "--records",
        nargs="+",
        choices=DEFAULT_RECORDS,
        default=DEFAULT_RECORDS,
        help="records to download (default: all pinned records)",
    )
    args = parser.parse_args()
    paths = download_mit_bih_records(args.output, tuple(args.records))
    print()
    print(
        f"MIT-BIH records {', '.join(args.records)} are available "
        f"in {paths[0].parent.resolve()}"
    )


if __name__ == "__main__":
    main()
