"""Verified download composition for file-based WFDB records."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from sigvue.helpers import RemoteFile, download_file


def download_wfdb_record(
    files: Iterable[RemoteFile],
    directory: str | Path,
    *,
    user_agent: str = "Sigvue-Examples/0.3",
    progress_factory: Callable[[str], Callable[[int, int | None], None]] | None = None,
) -> tuple[Path, ...]:
    """Download every verified companion file for one WFDB record."""
    manifest = tuple(files)
    if not manifest:
        raise ValueError("A WFDB download manifest cannot be empty")
    stems = {Path(remote.filename).stem for remote in manifest}
    if len(stems) != 1:
        raise ValueError("WFDB companion files must share one record name")
    suffixes = {Path(remote.filename).suffix for remote in manifest}
    if ".hea" not in suffixes or ".dat" not in suffixes:
        raise ValueError("WFDB manifests require header and signal data files")
    output = Path(directory).expanduser()
    return tuple(
        (
            download_file(
                remote,
                output,
                user_agent=user_agent,
                progress=(
                    None
                    if progress_factory is None
                    else progress_factory(remote.filename)
                ),
            )
        )
        for remote in manifest
    )


__all__ = ["download_wfdb_record"]
