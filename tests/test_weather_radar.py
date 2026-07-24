import bz2
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np

from sigvue_examples.plugins.nexrad import (
    NexradFormatError,
    describe_level3,
    level3_sequence_source,
    read_level3_header,
    read_level3_radial,
)
from sigvue_examples.weather_radar.plots import ppi_figure


def _set_u16(message: bytearray, halfword: int, value: int) -> None:
    struct.pack_into(">H", message, (halfword - 1) * 2, value)


def _set_i16(message: bytearray, halfword: int, value: int) -> None:
    struct.pack_into(">h", message, (halfword - 1) * 2, value)


def _set_u32(message: bytearray, halfword: int, value: int) -> None:
    struct.pack_into(">I", message, (halfword - 1) * 2, value)


def _set_i32(message: bytearray, halfword: int, value: int) -> None:
    struct.pack_into(">i", message, (halfword - 1) * 2, value)


def synthetic_n0b(
    *,
    first_radial_bytes: int = 4,
    scan_seconds: int = 11_454,
    radar_id: str = "TLX",
) -> bytes:
    radial_data = (
        struct.pack(">HHH", first_radial_bytes, 3595, 5)
        + bytes((0, 1, 2, 255))
        + struct.pack(">HHH", 4, 0, 5)
        + bytes((4, 5, 6, 7))
    )
    packet = (
        struct.pack(
            ">HHHhhHH",
            16,
            0,
            4,
            0,
            0,
            999,
            2,
        )
        + radial_data
    )
    symbology = (
        struct.pack(">hHIH", -1, 1, 16 + len(packet), 1)
        + struct.pack(">hI", -1, len(packet))
        + packet
    )
    compressed = bz2.compress(symbology)

    message = bytearray(120)
    _set_u16(message, 1, 153)
    _set_u16(message, 2, 19_864)
    _set_u32(message, 3, 11_473)
    _set_u32(message, 5, 120 + len(compressed))
    _set_u16(message, 7, 1)
    _set_u16(message, 9, 3)
    _set_i16(message, 10, -1)
    _set_i32(message, 11, 35_333)
    _set_i32(message, 13, -97_278)
    _set_i16(message, 15, 1_277)
    _set_u16(message, 16, 153)
    _set_u16(message, 17, 2)
    _set_u16(message, 18, 212)
    _set_u16(message, 19, 2_930)
    _set_u16(message, 20, 32)
    _set_u16(message, 21, 19_864)
    _set_u32(message, 22, scan_seconds)
    _set_u16(message, 24, 19_864)
    _set_u32(message, 25, scan_seconds + 18)
    _set_u16(message, 29, 3)
    _set_i16(message, 30, 5)
    _set_i16(message, 31, -320)
    _set_i16(message, 32, 5)
    _set_u16(message, 33, 254)
    _set_i16(message, 47, 69)
    _set_u16(message, 51, 1)
    _set_u32(message, 52, len(symbology))
    _set_u32(message, 55, 60)
    heading = f"SDUS54 KOUN 200310\r\r\nN0B{radar_id}\r\r\n".encode("ascii")
    return heading + bytes(message) + compressed


class NexradLevel3ReaderTests(unittest.TestCase):
    def test_header_and_packet_16_preserve_native_codes_and_coordinates(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "TLX_N0B_2024_05_20_03_10_54"
            path.write_bytes(synthetic_n0b())

            header = read_level3_header(path)
            scan = read_level3_radial(path)

        self.assertEqual("N0B", header.product_id)
        self.assertEqual("TLX", header.radar_id)
        self.assertEqual(153, header.message_code)
        self.assertEqual("2024-05-20T03:10:54+00:00", header.scan_time.isoformat())
        self.assertEqual((2, 4), scan.level_codes.shape)
        np.testing.assert_array_equal(
            scan.level_codes,
            np.asarray(((0, 1, 2, 255), (4, 5, 6, 7)), dtype=np.uint8),
        )
        np.testing.assert_allclose(scan.azimuth_start_deg, (359.5, 0.0))
        np.testing.assert_allclose(scan.azimuth_width_deg, (0.5, 0.5))
        np.testing.assert_allclose(
            scan.slant_range_edges_km,
            (0.0, 0.25, 0.5, 0.75, 1.0),
        )
        decoded = scan.reflectivity_dbz()
        self.assertTrue(np.isnan(decoded[0, 0]))
        self.assertTrue(np.isnan(decoded[0, 1]))
        self.assertEqual(-32.0, float(decoded[0, 2]))
        self.assertEqual(94.5, float(decoded[0, 3]))
        self.assertEqual(
            {
                "measured": 6,
                "below_threshold": 1,
                "range_folded": 1,
                "padding": 0,
            },
            scan.code_counts(),
        )

    def test_discovery_description_reads_metadata_without_gate_inflation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "TLX_N0B_2024_05_20_03_10_54"
            path.write_bytes(synthetic_n0b())
            resource = describe_level3(path)

        self.assertEqual(path.name, resource.identifier)
        self.assertIn("TLX N0B", resource.title)
        self.assertEqual(
            resource.timestamp.isoformat(),
            resource.summary["date"],
        )
        json.dumps(resource.summary)
        self.assertEqual(("NOAA", "NEXRAD", "Level III", "N0B"), resource.tags)

    def test_sequence_source_groups_chronological_scans_into_one_item(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "TLX_N0B_2024_05_20_03_10_54").write_bytes(
                synthetic_n0b(),
            )
            (root / "TLX_N0B_2024_05_20_03_20_54").write_bytes(
                synthetic_n0b(scan_seconds=12_054),
            )
            (root / "FDR_N0B_2024_05_20_03_10_05").write_bytes(
                synthetic_n0b(
                    scan_seconds=11_405,
                    radar_id="FDR",
                ),
            )
            source = level3_sequence_source(root)
            resources = source.discover()
            sequences = {
                resource.identifier: source.open(resource) for resource in resources
            }

        self.assertEqual(("FDR-N0B", "TLX-N0B"), tuple(sequences))
        self.assertEqual(1, sequences["FDR-N0B"].scan_count)
        self.assertEqual(2, sequences["TLX-N0B"].scan_count)
        self.assertEqual((0.0, 600.0), sequences["TLX-N0B"].elapsed_seconds)
        self.assertEqual(
            ("03:10:54", "03:20:54"),
            tuple(
                header.scan_time.strftime("%H:%M:%S")
                for header in sequences["TLX-N0B"].headers
            ),
        )

    def test_ppi_uses_responsive_browser_height_and_keeps_axis_titles(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "TLX_N0B_2024_05_20_03_10_54"
            path.write_bytes(synthetic_n0b())
            scan = read_level3_radial(path)

        figure = ppi_figure(
            scan,
            maximum_range_km=1.0,
            pixels=32,
            colormap="NEXRAD",
            theme="light",
        )

        self.assertIsNone(figure.layout.height)
        self.assertEqual("East of radar (km)", figure.layout.xaxis.title.text)
        self.assertEqual("North of radar (km)", figure.layout.yaxis.title.text)
        self.assertEqual((-1.0, 1.0), tuple(figure.layout.xaxis.range))
        self.assertEqual((-1.0, 1.0), tuple(figure.layout.yaxis.range))

    def test_radial_byte_count_must_match_declared_gate_count(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.nids"
            path.write_bytes(synthetic_n0b(first_radial_bytes=3))
            with self.assertRaisesRegex(
                NexradFormatError,
                "radial byte count",
            ):
                read_level3_radial(path)


if __name__ == "__main__":
    unittest.main()
