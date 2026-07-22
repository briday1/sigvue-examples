"""SigMF collection discovery and loading for calibrated radar recordings."""

import json
from pathlib import Path

from sigvue.plugin import DataResource, DirectorySource

from ..io.sigmf.capabilities import sigmf_discovery_summary
from .domain import CollectionMember, LfmCollection


def describe_collection(path: Path) -> DataResource:
    payload = json.loads(path.read_text(encoding="utf-8"))
    member = next((value for value in payload["members"] if value["role"] == "ota"), None)
    member_path = path.parent / member["metadata"] if member is not None else None
    member_metadata = (
        json.loads(member_path.read_text(encoding="utf-8"))
        if member_path is not None and member_path.is_file()
        else {
            "global": {"core:sample_rate": payload["collection"].get("sample_rate")},
            "captures": [],
        }
    )
    channel_count = sum(1 for value in payload["members"] if value["role"] == "ota")
    return DataResource(
        path.stem,
        payload["collection"]["name"],
        source=path,
        tags=("sigmf-collection", "ci16", f"{channel_count}-channel"),
        summary={
            **sigmf_discovery_summary(member_metadata),
            "members": "calibration, terminated-noise, ota",
        },
    )

def read_collection(path: Path) -> LfmCollection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[CollectionMember]] = {}
    for value in payload["members"]:
        member = CollectionMember(
            value["role"], int(value["channel"]), path.parent / value["metadata"], path.parent / value["data"], float(value["duration_seconds"])
        )
        grouped.setdefault(member.role, []).append(member)
    members = {role: tuple(sorted(values, key=lambda member: member.channel)) for role, values in grouped.items()}
    required = {"calibration", "terminated-noise", "ota"}
    if set(members) != required:
        raise ValueError(f"Collection must define exactly {sorted(required)}")
    sample_rate = float(payload["collection"]["sample_rate"])
    adc_bits = 16
    channel_count = len(members["ota"])
    if channel_count < 1:
        raise ValueError("Collection must define at least one channel")
    expected_channels = list(range(1, channel_count + 1))
    for role, records in members.items():
        if [record.channel for record in records] != expected_channels:
            raise ValueError(
                f"{role} must define the same contiguous channels 1 through {channel_count}"
            )
        for member in records:
            metadata = json.loads(member.metadata_path.read_text(encoding="utf-8"))["global"]
            if metadata.get("core:datatype") != "ci16_le" or int(metadata.get("core:num_channels", 0)) != 1:
                raise ValueError(f"{member.metadata_path.name} must be single-channel ci16_le")
            if float(metadata["core:sample_rate"]) != sample_rate:
                raise ValueError(f"{member.metadata_path.name} has a different sample rate")
            expected_bytes = round(member.duration * sample_rate) * 4
            if role != "ota" and member.data_path.stat().st_size < expected_bytes:
                raise ValueError(f"{member.data_path.name} is shorter than its declared duration")
    collection_metadata = payload["collection"]
    return LfmCollection(
        sample_rate,
        float(collection_metadata["calibration_dbm"]),
        adc_bits,
        members,
        float(collection_metadata.get("ota_prf_hz", 1_000.0)),
        float(collection_metadata.get("ota_pulse_width_seconds", 50e-6)),
        path,
    )


def collection_source(root: Path) -> DirectorySource:
    return DirectorySource(
        root, pattern="*.sigmf-collection", loader=read_collection,
        describe=describe_collection, recursive=True,
    )


__all__ = ["LfmCollection", "collection_source", "describe_collection", "read_collection"]
