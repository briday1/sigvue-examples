"""Exactness tests for the reusable, dependency-free WFDB helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sigvue_examples.plugins.wfdb import (
    WFDBWindow,
    load_wfdb_record,
    peak_to_peak_overview,
    read_mit_annotations,
)


def _signed_checksum(values: np.ndarray) -> int:
    checksum = int(np.sum(values, dtype=np.int64)) & 0xFFFF
    return checksum - 0x10000 if checksum >= 0x8000 else checksum


def _pack_format_212(channels: np.ndarray) -> bytes:
    """Pack two signed 12-bit channels exactly as WFDB format 212."""
    if channels.shape[0] != 2:
        raise ValueError("The test encoder requires exactly two channels")
    packed = bytearray()
    for first, second in channels.T:
        first_unsigned = int(first) & 0x0FFF
        second_unsigned = int(second) & 0x0FFF
        packed.extend(
            (
                first_unsigned & 0xFF,
                ((first_unsigned >> 8) & 0x0F) | ((second_unsigned >> 4) & 0xF0),
                second_unsigned & 0xFF,
            )
        )
    return bytes(packed)


def _annotation_word(interval: int, code: int) -> bytes:
    if not 0 <= interval <= 0x3FF:
        raise ValueError("Ordinary annotation intervals are ten-bit values")
    return bytes((interval & 0xFF, (code << 2) | (interval >> 8)))


def _skip_word(interval: int) -> bytes:
    unsigned = interval & 0xFFFFFFFF
    return b"".join(
        (
            _annotation_word(0, 59),
            bytes(((unsigned >> 16) & 0xFF, (unsigned >> 24) & 0xFF)),
            bytes((unsigned & 0xFF, (unsigned >> 8) & 0xFF)),
        )
    )


class WFDBHelperTests(unittest.TestCase):
    def _write_record(
        self,
        root: Path,
        channels: np.ndarray,
        *,
        annotations: bytes = b"\x00\x00",
    ) -> Path:
        first_checksum, second_checksum = (
            _signed_checksum(channel) for channel in channels
        )
        header = root / "fixture.hea"
        header.write_text(
            "\n".join(
                (
                    f"fixture 2 360 {channels.shape[1]}",
                    "fixture.dat 212 200(0)/mV 12 0 "
                    f"{int(channels[0, 0])} {first_checksum} 0 MLII",
                    "fixture.dat 212 100(0)/mV 12 0 "
                    f"{int(channels[1, 0])} {second_checksum} 0 V5",
                    "# exact synthetic fixture",
                    "",
                )
            ),
            encoding="ascii",
        )
        (root / "fixture.dat").write_bytes(_pack_format_212(channels))
        (root / "fixture.atr").write_bytes(annotations)
        return header

    def test_format_212_decodes_every_signed_edge_and_ranged_reads(self):
        expected = np.asarray(
            (
                (-2048, -1025, -1, 0, 1, 1024, 2047),
                (2047, 1024, 1, 0, -1, -1025, -2048),
            ),
            dtype=np.int16,
        )
        with TemporaryDirectory() as directory:
            record = load_wfdb_record(self._write_record(Path(directory), expected))

            decoded = record.read_digital(0, expected.shape[1])
            ranged = record.read_digital(1, 4)
            physical = record.read_physical(1, 4)
            verified_checksums = record.verify_signal_checksums()

        np.testing.assert_array_equal(expected, decoded)
        np.testing.assert_array_equal(expected[:, 1:5], ranged)
        np.testing.assert_array_equal(
            expected[:, 1:5] / np.asarray(((200.0,), (100.0,))),
            physical,
        )
        self.assertEqual(
            tuple(_signed_checksum(channel) for channel in expected),
            verified_checksums,
        )

    def test_mit_annotations_preserve_extended_sample_and_metadata(self):
        auxiliary = b"abc"
        payload = b"".join(
            (
                _annotation_word(10, 1),
                bytes((0xFE, 61 << 2)),
                bytes((3, 62 << 2)),
                bytes((5, 60 << 2)),
                bytes((len(auxiliary), 63 << 2)),
                auxiliary + b"\x00",
                _skip_word(70_000),
                _annotation_word(0, 5),
                _annotation_word(0, 0),
            )
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.atr"
            path.write_bytes(payload)
            annotations = read_mit_annotations(path)

        self.assertEqual((10, 70_010), tuple(a.sample for a in annotations))
        self.assertEqual(("N", "V"), tuple(a.symbol for a in annotations))
        self.assertEqual(("NORMAL", "PVC"), tuple(a.short_name for a in annotations))
        self.assertTrue(all(a.is_beat for a in annotations))
        self.assertEqual(-2, annotations[0].subtype)
        self.assertEqual("abc", annotations[0].auxiliary_note)
        self.assertEqual(0, annotations[1].subtype)
        self.assertEqual((3, 3), tuple(a.channel for a in annotations))
        self.assertEqual((5, 5), tuple(a.number for a in annotations))
        self.assertEqual(70_010 / 360, annotations[1].time_seconds(360))

    def test_window_memory_and_overview_are_based_on_exact_native_samples(self):
        samples = np.asarray(
            (
                (-8, 4, -2, 9, 1, 3),
                (10, 20, 30, 40, 50, 60),
            ),
            dtype=np.int16,
        )
        annotations = b"".join(
            (
                _annotation_word(1, 1),
                _annotation_word(3, 5),
                _annotation_word(0, 0),
            )
        )
        with TemporaryDirectory() as directory:
            record = load_wfdb_record(
                self._write_record(
                    Path(directory),
                    samples,
                    annotations=annotations,
                )
            )
            native = record.read_digital(1, 4)
            window = WFDBWindow(record, 1, native)
            overview = peak_to_peak_overview(record, bins=3, channel=0)

        self.assertEqual(native.nbytes, window.buffer_nbytes)
        np.testing.assert_array_equal(samples[:, 1:5], window.digital_samples)
        self.assertEqual((1, 4), tuple(a.sample for a in window.annotations))
        # Two exact source samples land in each overview bin. Channel zero has
        # 200 ADC units per mV, so these are the exact physical peak-to-peaks.
        np.testing.assert_allclose(
            np.asarray((12.0, 11.0, 2.0)) / 200.0,
            overview,
            rtol=0.0,
            atol=np.finfo(np.float64).eps,
        )


if __name__ == "__main__":
    unittest.main()
