from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from cts1_mo_tools.cts1_decode_satnogs_packets import (
    _PACKET_DECODERS,  # pyright: ignore[reportPrivateUsage]
    _PACKET_MIN_PAYLOAD_SIZES,  # pyright: ignore[reportPrivateUsage]
    crc32c,
    decode_raw_packet_safe,
)
from cts1_mo_tools.cts1_processing_pipeline import partitions
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    pipeline as step_2_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets.pipeline import (
    CTS1_CSP_HEADER_HEX,
    compute_distinct_packets,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets import (
    pipeline as step_3_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets.pipeline import (
    compute_decoded_packets,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets.schema import (
    DECODED_SCHEMA,
)
from polars.testing import assert_frame_equal


def _read_output(data_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(data_dir / step_3_pipeline.OUTPUT_FILENAME)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _log_message_hex(message: bytes) -> str:
    """A CSP-wrapped, CRC-complete LOG_MESSAGE packet (packet_type 0x03).

    `decode_log_message_packet` splits on the *last* newline, keeping
    everything before it as the message and discarding the rest (including
    the trailing CRC-32C bytes this appends) -- so `message` must end with
    `\\n` for it to decode back out cleanly.
    """
    body = bytes.fromhex(CTS1_CSP_HEADER_HEX) + bytes([0x03]) + message + b"\n"
    crc = crc32c(body)
    return (body + crc.to_bytes(4, "big")).hex()


def _distinct_packets_df() -> pl.DataFrame:
    packets_df = pl.DataFrame(
        [
            {
                "observation_id": 1,
                "decoder": "sso_rx_replay",
                "data_hex": _log_message_hex(b"hello"),
                "data_length_bytes": len(_log_message_hex(b"hello")) // 2,
                "csp_crc_valid": True,
                "rssi_db": -1.0,
                "rs_corrected_error_count": 0,
                "rs_correctable": True,
                "time_in_file_ms": 1000.0,
                "received_at": _dt("2026-01-01T00:00:00"),
            },
            {
                # Starts with the right CSP header (so step 2 keeps it) but
                # far too short a payload for any known packet type --
                # exercises the "still gets a row, just with nulls" path.
                "observation_id": 2,
                "decoder": "sso_rx_replay",
                "data_hex": CTS1_CSP_HEADER_HEX,
                "data_length_bytes": 4,
                "csp_crc_valid": False,
                "rssi_db": -2.0,
                "rs_corrected_error_count": 0,
                "rs_correctable": True,
                "time_in_file_ms": 2000.0,
                "received_at": _dt("2026-01-01T00:01:00"),
            },
        ],
        infer_schema_length=None,
    )
    observations_df = pl.DataFrame(
        [
            {
                "id": oid,
                "start": _dt("2026-01-01T00:00:00"),
                "end": _dt("2026-01-01T00:10:00"),
            }
            for oid in (1, 2)
        ],
        infer_schema_length=None,
    )
    return compute_distinct_packets(packets_df, observations_df)


def _empty_distinct_packets_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"data_hex": pl.String, "received_at": pl.Datetime("us", "UTC")}
    )


def test_empty_input_returns_empty() -> None:
    result = compute_decoded_packets(_empty_distinct_packets_df())
    assert result.is_empty()


def test_decodes_and_preserves_step2_columns() -> None:
    distinct = _distinct_packets_df()
    result = compute_decoded_packets(distinct)

    assert result.height == distinct.height
    # The CLI-shaped bridging columns don't leak into this pipeline's schema.
    assert "hex_payload" not in result.columns
    assert "received_timestamp" not in result.columns
    assert "observation_id" not in result.columns
    assert "csp_crc_valid_right" not in result.columns
    assert "data_hex" in result.columns
    assert "received_at" in result.columns


def test_log_message_packet_decodes() -> None:
    result = compute_decoded_packets(_distinct_packets_df())
    decoded = result.filter(pl.col("packet_type") == "LOG_MESSAGE")
    assert decoded.height == 1
    assert decoded["log_message"][0] == "hello"
    # Step 2's own verified csp_crc_valid must survive untouched -- not
    # clobbered or duplicated by the decoder's redundant recomputation.
    assert decoded["csp_crc_valid"][0] is True


def test_undecodable_packet_still_gets_a_row_with_nulls() -> None:
    result = compute_decoded_packets(_distinct_packets_df())
    undecoded = result.filter(pl.col("data_length_bytes") == 4)
    assert undecoded.height == 1
    assert undecoded["packet_type"][0] is None


def test_row_order_matches_input() -> None:
    distinct = _distinct_packets_df()
    result = compute_decoded_packets(distinct)
    assert result["received_at"].to_list() == distinct["received_at"].to_list()


def test_output_always_has_the_fixed_schema() -> None:
    distinct = _distinct_packets_df()
    result = compute_decoded_packets(distinct)

    step_2_columns = [col for col in distinct.columns if col != "data_hex"]
    assert result.columns == step_2_columns + list(DECODED_SCHEMA)
    assert dict(result.select(DECODED_SCHEMA.names()).schema) == dict(DECODED_SCHEMA)


@pytest.mark.parametrize("packet_type", sorted(_PACKET_DECODERS))
def test_decoded_schema_lists_every_decoded_field(packet_type: int) -> None:
    """If this fails, a decoder has grown a field: add it to
    `DECODED_SCHEMA` (step 3 otherwise rebuilds its whole output, every
    run, until it's there)."""
    body = (
        bytes.fromhex(CTS1_CSP_HEADER_HEX)
        + bytes([packet_type])
        + bytes(_PACKET_MIN_PAYLOAD_SIZES.get(packet_type, 0) + 8)
    )
    decoded = decode_raw_packet_safe(
        body + crc32c(body).to_bytes(4, "big"), crc_valid=True
    )

    assert decoded is not None
    # `csp_crc_valid` is the decoder's own verdict, which step 3 drops.
    assert set(decoded) - {"csp_crc_valid"} <= set(DECODED_SCHEMA)


def test_incremental_run_matches_a_full_one(tmp_path: Path) -> None:
    def distinct_month(*messages: tuple[str, bytes]) -> pl.DataFrame:
        base = _distinct_packets_df().head(1)
        return pl.concat(
            base.with_columns(
                received_at=pl.lit(_dt(at)), data_hex=pl.lit(_log_message_hex(msg))
            )
            for at, msg in messages
        )

    august = distinct_month(("2026-08-01T00:00:00", b"aug"))
    september = distinct_month(("2026-09-01T00:00:00", b"sep"))
    september_grown = distinct_month(
        ("2026-09-01T00:00:00", b"sep"), ("2026-09-02T00:00:00", b"sep2")
    )
    distinct_path = tmp_path / step_2_pipeline.OUTPUT_FILENAME
    partitions.write_frame(pl.concat([august, september]), distinct_path)
    step_3_pipeline.run(data_dir=tmp_path, incremental=True)  # First: full.
    august_file = partitions.month_files(tmp_path / step_3_pipeline.OUTPUT_FILENAME)[
        "2026-08"
    ]
    august_mtime = august_file.stat().st_mtime_ns

    partitions.write_frame(september_grown, distinct_path, months={"2026-09"})
    step_3_pipeline.run(data_dir=tmp_path, incremental=True)

    assert august_file.stat().st_mtime_ns == august_mtime
    assert_frame_equal(
        _read_output(tmp_path),
        compute_decoded_packets(pl.concat([august, september_grown])),
    )
