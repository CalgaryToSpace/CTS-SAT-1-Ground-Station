import json
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_decode_satnogs_packets import crc32c
from cts1_mo_tools.cts1_processing_pipeline.step_6_deduplicate_gnss_samples.pipeline import (  # noqa: E501
    compute_distinct_gnss_samples,
    run,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.data import (
    load_gnss_sample_counts_per_window,
)

_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_CSP = bytes.fromhex("c2a28a00")

_packet_counter = 0


def _gnss_hex(*, seq_num: int, ring_position: int = 7, sample: bytes) -> str:
    """A CSP-wrapped, CRC-complete GNSS_BESTXYZB_SAMPLE packet."""
    body = _CSP + struct.pack("<B H H", 0x30, seq_num, ring_position) + sample
    return (body + crc32c(body).to_bytes(4, "big")).hex()


def _gnss(  # noqa: PLR0913
    *,
    seq_num: int,
    sample: bytes = b"sample-a",
    ring_position: int = 7,
    seconds: float = 0.0,
    packet_count: int = 1,
    csp_crc_valid: bool = True,
    position_x_m: float = 1.0,
) -> dict[str, Any]:
    global _packet_counter  # noqa: PLW0603
    _packet_counter += 1
    return {
        "packet_id": f"pkt{_packet_counter:04d}",
        "data_hex": _gnss_hex(
            seq_num=seq_num, ring_position=ring_position, sample=sample
        ),
        "packet_type": "GNSS_BESTXYZB_SAMPLE",
        "csp_crc_valid": csp_crc_valid,
        "received_at": _BASE + timedelta(seconds=seconds),
        "packet_count": packet_count,
        "gnss_downlink_seq_num": seq_num,
        "gnss_ring_position": ring_position,
        "gnss_position_x_m": position_x_m,
        # Beacon-only GNSS fields: always null on GNSS sample rows.
        "gnss_rx_mode": None,
        "gnss_uart_interrupt_enabled": None,
    }


def _beacon(*, seconds: float = 0.0) -> dict[str, Any]:
    return {
        "packet_id": "beacon",
        "data_hex": (_CSP + b"\x01beacon").hex(),
        "packet_type": "BEACON_BASIC",
        "csp_crc_valid": True,
        "received_at": _BASE + timedelta(seconds=seconds),
        "packet_count": 1,
        "gnss_downlink_seq_num": None,
        "gnss_ring_position": None,
        "gnss_position_x_m": None,
        "gnss_rx_mode": "FIREHOSE_MODE",
        "gnss_uart_interrupt_enabled": True,
    }


def test_empty_input_returns_empty_with_expected_schema() -> None:
    result = compute_distinct_gnss_samples(pl.DataFrame())
    assert result.height == 0
    assert result.schema["receive_count"] == pl.Int64
    assert result.schema["first_received_at"] == pl.Datetime("us", "UTC")


def test_groups_copies_differing_only_in_downlink_counter() -> None:
    df = pl.DataFrame(
        [
            _gnss(seq_num=10, seconds=100, packet_count=3, position_x_m=1.0),
            _gnss(seq_num=3, seconds=0, packet_count=2, position_x_m=1.0),
            _gnss(seq_num=55, seconds=50, packet_count=1, position_x_m=1.0),
            _gnss(seq_num=11, sample=b"sample-b", seconds=20, position_x_m=2.0),
            _beacon(),
        ]
    )
    result = compute_distinct_gnss_samples(df)

    assert result.height == 2
    a = result.filter(pl.col("gnss_position_x_m") == 1.0).row(0, named=True)
    assert a["receive_count"] == 3
    assert a["decode_count"] == 6
    assert a["first_received_at"] == _BASE
    assert a["last_received_at"] == _BASE + timedelta(seconds=100)
    assert json.loads(a["downlink_seq_nums"]) == [3, 10, 55]
    assert len(json.loads(a["packet_ids"])) == 3
    # ring_position (0x0007, little-endian) + the raw sample bytes.
    assert a["gnss_sample_hex"] == (b"\x07\x00" + b"sample-a").hex()
    assert a["gnss_ring_position"] == 7

    b = result.filter(pl.col("gnss_position_x_m") == 2.0).row(0, named=True)
    assert b["receive_count"] == 1
    assert a["gnss_sample_id"] != b["gnss_sample_id"]

    # Per-copy and beacon-only columns aren't carried through.
    for column in ("gnss_downlink_seq_num", "gnss_rx_mode", "data_hex"):
        assert column not in result.columns

    # Newest first.
    assert result["first_received_at"].to_list() == [
        _BASE + timedelta(seconds=20),
        _BASE,
    ]


def test_different_ring_position_is_a_different_sample() -> None:
    df = pl.DataFrame([_gnss(seq_num=1, ring_position=1), _gnss(seq_num=2)])
    assert compute_distinct_gnss_samples(df).height == 2


def test_invalid_crc_packets_are_skipped() -> None:
    df = pl.DataFrame(
        [_gnss(seq_num=1), _gnss(seq_num=2, sample=b"flipped", csp_crc_valid=False)]
    )
    result = compute_distinct_gnss_samples(df)
    assert result.height == 1
    assert result["receive_count"].to_list() == [1]


def test_no_gnss_packets_gives_empty_result() -> None:
    result = compute_distinct_gnss_samples(pl.DataFrame([_beacon()]))
    assert result.height == 0
    assert "gnss_sample_hex" in result.columns


def test_run_writes_parquet_and_dashboard_counts_it(tmp_path: Path) -> None:
    pl.DataFrame(
        [
            _gnss(seq_num=1, seconds=0),
            _gnss(seq_num=2, seconds=60),
            _gnss(seq_num=3, sample=b"sample-b", seconds=120),
        ]
    ).write_parquet(tmp_path / "everything_decoded.parquet")
    run(data_dir=tmp_path)

    out_path = tmp_path / "distinct_gnss_samples.parquet"
    out = pl.read_parquet(out_path)
    assert sorted(out["receive_count"].to_list()) == [1, 2]

    counts = load_gnss_sample_counts_per_window(out_path, every=timedelta(hours=6))
    assert counts.select("times_received", "count").rows() == [("1x", 1), ("2x", 1)]


def test_dashboard_counts_without_step_6_output(tmp_path: Path) -> None:
    counts = load_gnss_sample_counts_per_window(
        tmp_path / "missing.parquet", every=timedelta(hours=6)
    )
    assert counts.height == 0
