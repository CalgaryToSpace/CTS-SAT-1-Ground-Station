from datetime import UTC, datetime

import polars as pl
from cts1_mo_tools.cts1_decode_satnogs_packets import crc32c
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets.pipeline import (
    CTS1_CSP_HEADER_HEX,
    compute_distinct_packets,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets.pipeline import (
    compute_decoded_packets,
)


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
