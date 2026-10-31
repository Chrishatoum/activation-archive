from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from python.activation_archive import (
    ActivationArchive,
    ArchiveWriter,
    build_archive_from_parquet,
    compress_archive,
)
from python.load_activations import METADATA_COLUMNS
from python.codec_archive import CodecActivationArchive, compress_archive_codec
from python.quantized_archive import (
    QuantizedActivationArchive,
    _group_rows,
    compress_archive_quant8,
    fit_quant8_groups,
)


def make_metadata(start: int, rows: int) -> dict[str, np.ndarray]:
    return {
        "game_id": np.full(rows, "fixture", dtype="U7"),
        "round": np.zeros(rows, dtype=np.int16),
        "layer": np.arange(start, start + rows, dtype=np.int16) % 3,
        "activation_site": np.where(np.arange(start, start + rows) % 2, "post_mlp", "post_attention"),
        "token_idx": np.arange(start, start + rows, dtype=np.int32),
    }


def make_bits(rows: int, dimensions: int) -> np.ndarray:
    values = np.arange(rows * dimensions, dtype=np.uint16).reshape(rows, dimensions)
    values[0, 0] = 0x7FC1
    return values


def write_fixture_parquet(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    metadata = make_metadata(0, 7)
    bits = make_bits(7, 4)
    columns: dict[str, pa.Array] = {
        "game_id": pa.array(metadata["game_id"].tolist()),
        "round": pa.array(metadata["round"]),
        "layer": pa.array(metadata["layer"]),
        "activation_site": pa.array(metadata["activation_site"].tolist()),
        "token_idx": pa.array(metadata["token_idx"]),
    }
    for dimension in range(bits.shape[1]):
        columns[f"activation_dim{dimension:04d}"] = pa.array(
            [bits[row, dimension].tobytes() for row in range(bits.shape[0])],
            type=pa.binary(2),
        )
    table = pa.table(columns)
    pq.write_table(table, path, row_group_size=4)
    return metadata, bits


class ActivationArchiveTests(unittest.TestCase):
    def test_incremental_writer_round_trip_and_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.archive"
            metadata = make_metadata(0, 7)
            bits = make_bits(7, 4)
            with ArchiveWriter(archive_path, "fixture", chunk_rows=3) as writer:
                writer.append({name: values[:2] for name, values in metadata.items()}, bits[:2])
                writer.append({name: values[2:] for name, values in metadata.items()}, bits[2:])
                writer.finalize()

            archive = ActivationArchive(archive_path)
            archive.verify()
            actual_metadata, actual_bits = archive.read()
            np.testing.assert_array_equal(actual_bits, bits)
            for name in METADATA_COLUMNS:
                np.testing.assert_array_equal(actual_metadata[name], metadata[name])
            yielded = list(archive.iter_chunks())
            self.assertGreater(len(yielded), 1)
            self.assertTrue(all(len(chunk_bits) <= 3 for _, chunk_bits in yielded))
            np.testing.assert_array_equal(
                np.concatenate([chunk_bits for _, chunk_bits in yielded]), bits
            )

            selected_metadata, selected_bits = archive.read(
                round_index=0, token_start=2, token_stop=6, layer=1
            )
            expected = (metadata["token_idx"] >= 2) & (metadata["token_idx"] < 6)
            expected &= metadata["layer"] == 1
            np.testing.assert_array_equal(selected_bits, bits[expected])
            np.testing.assert_array_equal(selected_metadata["token_idx"], metadata["token_idx"][expected])
            empty_metadata, empty_bits = archive.read(layer=99)
            self.assertEqual(empty_bits.shape, (0, bits.shape[1]))
            self.assertEqual(empty_metadata["layer"].dtype, metadata["layer"].dtype)

    def test_bounded_parquet_conversion_preserves_bits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "source.parquet"
            archive_path = root / "archive"
            metadata, bits = write_fixture_parquet(parquet_path)
            result = build_archive_from_parquet(
                parquet_path, archive_path, "fixture", batch_rows=2
            )
            self.assertEqual(result["rows"], bits.shape[0])
            self.assertLess(result["chunk_rows_max"], result["rows"])

            archive = ActivationArchive(archive_path)
            archive.verify()
            actual_metadata, actual_bits = archive.read()
            np.testing.assert_array_equal(actual_bits, bits)
            for name in METADATA_COLUMNS:
                np.testing.assert_array_equal(actual_metadata[name], metadata[name])

            selected_metadata, selected_bits = archive.read(activation_site="post_attention")
            expected = metadata["activation_site"] == "post_attention"
            np.testing.assert_array_equal(selected_bits, bits[expected])
            np.testing.assert_array_equal(
                selected_metadata["token_idx"], metadata["token_idx"][expected]
            )

    def test_verify_detects_modified_activation_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.archive"
            metadata = make_metadata(0, 2)
            bits = make_bits(2, 4)
            with ArchiveWriter(archive_path, "fixture", chunk_rows=2) as writer:
                writer.append(metadata, bits)
                writer.finalize()
            import zarr

            array = zarr.open_array(str(archive_path / "activations.zarr"), mode="r+")
            array[0, 0] = array[0, 0] ^ np.uint16(1)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                ActivationArchive(archive_path).verify()

    def test_zstandard_archive_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.archive"
            compressed_path = root / "compressed.archive"
            metadata = make_metadata(0, 7)
            bits = make_bits(7, 4)
            with ArchiveWriter(raw_path, "fixture", chunk_rows=3) as writer:
                writer.append(metadata, bits)
                writer.finalize()

            compressed_index = compress_archive(raw_path, compressed_path)
            self.assertEqual(compressed_index["compression"], "zstd")
            self.assertEqual(compressed_index["version"], 6)
            self.assertTrue((compressed_path / "activations.zarr").exists())
            compressed_archive = ActivationArchive(compressed_path)
            compressed_archive.verify()
            actual_metadata, actual_bits = compressed_archive.read()
            np.testing.assert_array_equal(actual_bits, bits)
            for name in METADATA_COLUMNS:
                np.testing.assert_array_equal(actual_metadata[name], metadata[name])

    def test_quant8_archive_reconstructs_with_recorded_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.archive"
            quantized_path = root / "quant8.archive"
            metadata = make_metadata(0, 7)
            bits = make_bits(7, 4)
            bits[0, 0] = 0x3F80  # Quant8 requires finite values; replace the fixture NaN with 1.0.
            with ArchiveWriter(raw_path, "fixture", chunk_rows=3) as writer:
                writer.append(metadata, bits)
                writer.finalize()

            quantized_index = compress_archive_quant8(raw_path, quantized_path)
            quantized = QuantizedActivationArchive(quantized_path)
            quantized.verify()
            actual_metadata, values = quantized.read()
            original = (bits.astype(np.uint32) << 16).view(np.float32)
            self.assertEqual(values.shape, original.shape)
            self.assertFalse(np.array_equal(values, original))
            self.assertGreater(quantized_index["relative_frobenius_error"], 0)
            for name in METADATA_COLUMNS:
                np.testing.assert_array_equal(actual_metadata[name], metadata[name])
            selected_metadata, selected_values = quantized.read(layer=1)
            expected = metadata["layer"] == 1
            np.testing.assert_array_equal(selected_metadata["token_idx"], metadata["token_idx"][expected])
            np.testing.assert_array_equal(selected_values, values[expected])

    def test_quant8_layer_site_scales_reduce_error_across_layer_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.archive"
            metadata = make_metadata(0, 12)
            rng = np.random.default_rng(0)
            # Layer 2 spans a range 1000x wider than layer 0, as in the pilot data.
            values = rng.standard_normal((12, 8)).astype(np.float32)
            values *= (10.0 ** (3 * metadata["layer"] / 2)).astype(np.float32)[:, None]
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            with ArchiveWriter(raw_path, "fixture", chunk_rows=5) as writer:
                writer.append(metadata, bits)
                writer.finalize()

            global_index = compress_archive_quant8(raw_path, root / "global.archive")
            grouped_index = compress_archive_quant8(
                raw_path, root / "grouped.archive", granularity="layer_site"
            )
            self.assertEqual(len(grouped_index["groups"]), 6)
            self.assertLess(
                grouped_index["relative_frobenius_error"], global_index["relative_frobenius_error"]
            )
            grouped = QuantizedActivationArchive(root / "grouped.archive")
            grouped.verify()
            actual_metadata, decoded = grouped.read()
            original = (bits.astype(np.uint32) << 16).view(np.float32)
            layer_zero = metadata["layer"] == 0
            relative = np.linalg.norm(decoded[layer_zero] - original[layer_zero]) / np.linalg.norm(
                original[layer_zero]
            )
            self.assertLess(relative, 0.05)
            for name in METADATA_COLUMNS:
                np.testing.assert_array_equal(actual_metadata[name], metadata[name])

    def test_quant8_held_out_calibration_clips_and_rejects_source_game(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = make_metadata(0, 12)
            rng = np.random.default_rng(1)
            calibration_values = rng.uniform(-1, 1, (12, 4)).astype(np.float32)
            source_values = calibration_values * 2  # Half the source range lies outside calibration.
            for name, game_id, values in [
                ("calibration.archive", "other", calibration_values),
                ("source.archive", "fixture", source_values),
            ]:
                with ArchiveWriter(root / name, game_id, chunk_rows=5) as writer:
                    writer.append(metadata, (values.view(np.uint32) >> 16).astype(np.uint16))
                    writer.finalize()

            index = compress_archive_quant8(
                root / "source.archive",
                root / "held_out.archive",
                granularity="layer_site",
                calibration_paths=[root / "calibration.archive"],
            )
            self.assertEqual(index["calibration"]["mode"], "held_out")
            self.assertEqual(index["calibration"]["game_ids"], ["other"])
            self.assertGreater(index["clipped_values"], 0)
            QuantizedActivationArchive(root / "held_out.archive").verify()

            with self.assertRaisesRegex(ValueError, "must not include the source game"):
                compress_archive_quant8(
                    root / "source.archive",
                    root / "leaky.archive",
                    calibration_paths=[root / "source.archive"],
                )
            self.assertFalse((root / "leaky.archive").exists())

    def test_quant8_clipped_exceptions_recover_exact_outlier_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = make_metadata(0, 12)
            rng = np.random.default_rng(4)
            calibration_values = rng.uniform(-1, 1, (12, 4)).astype(np.float32)
            source_values = calibration_values * 2  # Half the source range lies outside calibration.
            source_bits = (source_values.view(np.uint32) >> 16).astype(np.uint16)
            original = (source_bits.astype(np.uint32) << 16).view(np.float32)
            for name, game_id, values in [
                ("calibration.archive", "other", calibration_values),
                ("source.archive", "fixture", source_values),
            ]:
                with ArchiveWriter(root / name, game_id, chunk_rows=5) as writer:
                    writer.append(metadata, (values.view(np.uint32) >> 16).astype(np.uint16))
                    writer.finalize()

            without = compress_archive_quant8(
                root / "source.archive",
                root / "without_exceptions.archive",
                granularity="layer_site",
                calibration_paths=[root / "calibration.archive"],
            )
            with_exceptions = compress_archive_quant8(
                root / "source.archive",
                root / "with_exceptions.archive",
                granularity="layer_site",
                calibration_paths=[root / "calibration.archive"],
                store_clipped_exceptions=True,
            )
            self.assertGreater(without["clipped_values"], 0)
            self.assertEqual(with_exceptions["clipped_values"], without["clipped_values"])
            self.assertTrue(with_exceptions["clipped_exceptions_stored"])
            self.assertGreater(with_exceptions["exception_bytes_total"], 0)
            self.assertLess(with_exceptions["relative_frobenius_error"], without["relative_frobenius_error"])

            QuantizedActivationArchive(root / "with_exceptions.archive").verify()
            _, decoded_with = QuantizedActivationArchive(root / "with_exceptions.archive").read()
            _, decoded_without = QuantizedActivationArchive(root / "without_exceptions.archive").read()

            groups = fit_quant8_groups([root / "calibration.archive"], "layer_site")
            clipped_mask = np.zeros(original.shape, dtype=bool)
            for key, rows in _group_rows("layer_site", metadata):
                group = groups[key]
                raw_codes = np.rint((original[rows].astype(np.float64) - group["minimum"]) / group["scale"])
                clipped_mask[rows] = (raw_codes < 0) | (raw_codes > 255)
            self.assertTrue(clipped_mask.any())

            np.testing.assert_array_equal(decoded_with[clipped_mask], original[clipped_mask])
            self.assertFalse(np.array_equal(decoded_without[clipped_mask], original[clipped_mask]))

    def test_quant8_per_dimension_scales_handle_outlier_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.archive"
            metadata = make_metadata(0, 12)
            rng = np.random.default_rng(2)
            values = rng.standard_normal((12, 8)).astype(np.float32)
            values[:, 3] *= 1000  # One outlier dimension, as in the pilot data.
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            with ArchiveWriter(raw_path, "fixture", chunk_rows=5) as writer:
                writer.append(metadata, bits)
                writer.finalize()

            grouped = compress_archive_quant8(raw_path, root / "grouped.archive", granularity="layer_site")
            per_dim = compress_archive_quant8(raw_path, root / "per_dim.archive", granularity="layer_site_dim")
            self.assertIn("parameters_file", per_dim)
            self.assertEqual(per_dim["clipped_values"], 0)
            archive = QuantizedActivationArchive(root / "per_dim.archive")
            archive.verify()
            actual_metadata, decoded = archive.read()
            original = (bits.astype(np.uint32) << 16).view(np.float32)
            ordinary = np.delete(np.arange(8), 3)
            per_dim_error = np.linalg.norm(decoded[:, ordinary] - original[:, ordinary]) / np.linalg.norm(
                original[:, ordinary]
            )
            _, grouped_decoded = QuantizedActivationArchive(root / "grouped.archive").read()
            grouped_error = np.linalg.norm(
                grouped_decoded[:, ordinary] - original[:, ordinary]
            ) / np.linalg.norm(original[:, ordinary])
            self.assertLess(per_dim_error, grouped_error / 10)
            self.assertLessEqual(per_dim["relative_frobenius_error"], grouped["relative_frobenius_error"])
            for name in METADATA_COLUMNS:
                np.testing.assert_array_equal(actual_metadata[name], metadata[name])

    def test_codec_archives_round_trip_within_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.archive"
            metadata = make_metadata(0, 24)
            rng = np.random.default_rng(3)
            values = rng.standard_normal((24, 16)).astype(np.float32)
            values *= (10.0 ** metadata["layer"]).astype(np.float32)[:, None]
            bits = (values.view(np.uint32) >> 16).astype(np.uint16)
            original = (bits.astype(np.uint32) << 16).view(np.float32)
            with ArchiveWriter(raw_path, "fixture", chunk_rows=10) as writer:
                writer.append(metadata, bits)
                writer.finalize()

            cases = [
                ("bf16_trunc_zstd", {"mantissa_bits": 7}),
                ("bf16_trunc_zstd", {"mantissa_bits": 3}),
                ("fp8_e4m3_block_scale", {}),
                ("zfp_fixed_rate", {"rate": 16}),
                ("sz3_abs_rms", {"tolerance": 0.01}),
            ]
            for codec, params in cases:
                with self.subTest(codec=codec, params=params):
                    path = root / f"{codec}-{len(params) and list(params.values())[0]}.archive"
                    index = compress_archive_codec(raw_path, path, codec, **params)
                    archive = CodecActivationArchive(path)
                    archive.verify()
                    actual_metadata, decoded = archive.read()
                    for name in METADATA_COLUMNS:
                        np.testing.assert_array_equal(actual_metadata[name], metadata[name])
                    relative = np.abs(decoded - original) / np.abs(original)
                    if codec == "bf16_trunc_zstd" and params["mantissa_bits"] == 7:
                        np.testing.assert_array_equal(decoded, original)
                    elif codec == "bf16_trunc_zstd":
                        self.assertLessEqual(relative.max(), 2.0 ** -(params["mantissa_bits"] + 1))
                    elif codec == "sz3_abs_rms":
                        for chunk in index["chunks"]:
                            self.assertTrue(all("abs_error_bound" in block for block in chunk["blocks"]))
                        for layer in range(3):
                            rows = metadata["layer"] == layer
                            rms = np.sqrt(np.mean(np.square(original[rows].astype(np.float64))))
                            # Bounds are per (chunk, group) block; allow the spread between block RMS values.
                            self.assertLess(np.abs(decoded[rows] - original[rows]).max(), 0.05 * rms)
                    else:
                        error = np.linalg.norm(decoded - original) / np.linalg.norm(original)
                        self.assertLess(error, 0.1)
                    selected_metadata, selected = archive.read(layer=2, token_start=4, token_stop=20)
                    expected = (metadata["layer"] == 2) & (metadata["token_idx"] >= 4) & (metadata["token_idx"] < 20)
                    np.testing.assert_array_equal(selected_metadata["token_idx"], metadata["token_idx"][expected])
                    np.testing.assert_array_equal(selected, decoded[expected])

            with self.assertRaisesRegex(ValueError, "mantissa_bits"):
                compress_archive_codec(raw_path, root / "bad.archive", "bf16_trunc_zstd", mantissa_bits=8)

    def test_quant8_archive_rejects_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.archive"
            quantized_path = root / "quant8.archive"
            with ArchiveWriter(raw_path, "fixture", chunk_rows=3) as writer:
                writer.append(make_metadata(0, 7), make_bits(7, 4))
                writer.finalize()

            with self.assertRaisesRegex(ValueError, "finite"):
                compress_archive_quant8(raw_path, quantized_path)
            self.assertFalse(quantized_path.exists())


if __name__ == "__main__":
    unittest.main()