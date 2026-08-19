import json
from datetime import UTC, datetime
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_decode_satnogs_packets import crc32c, verify_csp_packet_crc32c
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets.pipeline import (
    compute_distinct_packets,
)

_DEFAULT_PACKET = {
    "observation_id": 1,
    "decoder": "sso_rx_replay",
    "data_hex": "c2a28a00aa",
    "data_length_bytes": 1,
    "csp_crc_valid": True,
    "rssi_db": -1.0,
    "rs_corrected_error_count": 0,
    "rs_correctable": True,
    "time_in_file_ms": 1000.0,
}


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _packet(*, received_at: str, **overrides: Any) -> dict[str, Any]:
    return {**_DEFAULT_PACKET, "received_at": _dt(received_at), **overrides}


def _packets_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def _observations_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


# A default observation window wide enough to cover the default packet times
# used across most tests below.
_OBS_1 = {
    "id": 1,
    "start": _dt("2026-08-12T19:30:00"),
    "end": _dt("2026-08-12T19:40:00"),
}


def test_sso_rs_uncorrectable_is_dropped() -> None:
    observations = _observations_df([_OBS_1])
    packets = _packets_df(
        [
            _packet(
                received_at="2026-08-12T19:35:00",
                rs_corrected_error_count=-1,
                rs_correctable=False,
            )
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert result.is_empty()


def test_sso_rs_corrected_is_kept() -> None:
    observations = _observations_df([_OBS_1])
    packets = _packets_df(
        [_packet(received_at="2026-08-12T19:35:00", rs_corrected_error_count=0)]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    assert result["rs_corrected_error_count"][0] == 0


def test_sso_dedupes_within_one_minute_across_observations() -> None:
    """Two ground stations catching the same overpass, both sso_rx_replay,
    within BASELINE_DEDUPE_TOLERANCE of each other, merge into one row.
    """
    observations = _observations_df(
        [
            _OBS_1,
            {
                "id": 2,
                "start": _dt("2026-08-12T19:30:30"),
                "end": _dt("2026-08-12T19:40:30"),
            },
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=1,
                received_at="2026-08-12T19:35:00",
                data_hex="c2a28a00aa",
            ),
            _packet(
                observation_id=2,
                received_at="2026-08-12T19:35:20",
                data_hex="c2a28a00aa",
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["received_at"] == _dt("2026-08-12T19:35:00")  # earliest
    assert row["received_at_source"] == "sso_rx_replay"
    assert row["packet_count"] == 2
    assert json.loads(row["observation_ids"]) == [1, 2]
    assert json.loads(row["decoders"]) == ["sso_rx_replay"]


def test_askew_outranks_sso_as_baseline_within_same_cluster() -> None:
    """When both askew_demod_from_file and sso_rx_replay decode the same
    content within BASELINE_DEDUPE_TOLERANCE, askew's own received_at/decoder
    wins as the cluster's authoritative source, even though it arrives later.
    """
    observations = _observations_df([_OBS_1])
    packets = _packets_df(
        [
            _packet(
                observation_id=1,
                received_at="2026-08-12T19:35:10",  # later than askew's
                data_hex="c2a28a00aa",
            ),
            _packet(
                observation_id=1,
                decoder="askew_demod_from_file",
                received_at="2026-08-12T19:35:00",  # earlier -- still wins
                data_hex="c2a28a00aa",
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["received_at"] == _dt("2026-08-12T19:35:00")
    assert row["received_at_source"] == "askew_demod_from_file"
    assert row["packet_count"] == 2
    assert json.loads(row["decoders"]) == ["askew_demod_from_file", "sso_rx_replay"]


def test_sso_does_not_dedupe_beyond_one_minute() -> None:
    """Two identical-content sso_rx_replay decodes 45 minutes apart are two
    distinct receptions, not a single duplicate detection.
    """
    observations = _observations_df(
        [
            _OBS_1,
            {
                "id": 3,
                "start": _dt("2026-08-12T20:15:00"),
                "end": _dt("2026-08-12T20:25:00"),
            },
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=1,
                received_at="2026-08-12T19:35:00",
                data_hex="c2a28a00aa",
            ),
            _packet(
                observation_id=3,
                received_at="2026-08-12T20:20:00",
                data_hex="c2a28a00aa",
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 2
    assert set(result["received_at"].to_list()) == {
        _dt("2026-08-12T19:35:00"),
        _dt("2026-08-12T20:20:00"),
    }
    assert (result["packet_count"] == 1).all()


def test_other_decoder_matches_baseline_via_observation_window_overlap() -> None:
    """A low-trust decoder's packet, way outside the 15-minute tolerance but
    from an observation whose window overlaps the baseline's, still merges.
    """
    observations = _observations_df(
        [
            _OBS_1,
            {
                "id": 4,
                "start": _dt("2026-08-12T19:31:00"),
                "end": _dt("2026-08-12T19:41:00"),
            },
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=1,
                received_at="2026-08-12T19:37:00",
                data_hex="c2a28a00cc",
            ),
            _packet(
                observation_id=4,
                decoder="gr_satellites_pdu",
                received_at="2026-08-12T23:59:00",  # way outside 15 min
                data_hex="c2a28a00cc",
                csp_crc_valid=None,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["packet_count"] == 2
    assert json.loads(row["decoders"]) == ["gr_satellites_pdu", "sso_rx_replay"]
    assert row["received_at"] == _dt("2026-08-12T19:37:00")  # baseline's own time


def test_other_decoder_matches_baseline_via_fifteen_minute_tolerance() -> None:
    """A low-trust decoder's packet from a non-overlapping observation still
    merges if it's within OTHER_DECODER_TOLERANCE of the baseline.
    """
    observations = _observations_df(
        [
            _OBS_1,
            {
                "id": 5,
                "start": _dt("2026-08-12T05:00:00"),
                "end": _dt("2026-08-12T05:10:00"),
            },
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=1,
                received_at="2026-08-12T19:35:00",
                data_hex="c2a28a00dd",
            ),
            _packet(
                observation_id=5,
                decoder="gr_satellites_kiss",
                received_at="2026-08-12T19:40:00",  # 5 min after baseline
                data_hex="c2a28a00dd",
                csp_crc_valid=None,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["packet_count"] == 2
    assert json.loads(row["decoders"]) == ["gr_satellites_kiss", "sso_rx_replay"]


def test_other_decoder_not_matched_becomes_its_own_leftover_row() -> None:
    """Content with no sso_rx_replay baseline anywhere is still kept, but as
    its own row with an 'estimated' (low-trust) received_at.
    """
    observations = _observations_df(
        [
            {
                "id": 6,
                "start": _dt("2026-08-12T23:50:00"),
                "end": _dt("2026-08-13T00:00:00"),
            }
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=6,
                decoder="satnogs_data_demod",
                received_at="2026-08-12T23:59:00",
                data_hex="c2a28a00ee",
                csp_crc_valid=True,  # already has a valid CRC -- no completion needed
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            )
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["received_at_source"] == "estimated"
    assert row["received_at"] == _dt("2026-08-12T23:59:00")
    assert json.loads(row["decoders"]) == ["satnogs_data_demod"]
    assert row["rssi_db"] is None
    assert row["rs_corrected_error_count"] is None


def test_leftover_content_clusters_by_fifteen_minute_gaps() -> None:
    """Two low-trust-only decodes of the same content close together cluster
    into one row; a third, far away, stays separate.
    """
    observations = _observations_df(
        [
            {
                "id": 7,
                "start": _dt("2026-08-12T10:00:00"),
                "end": _dt("2026-08-12T10:10:00"),
            },
            {
                "id": 8,
                "start": _dt("2026-08-12T10:05:00"),
                "end": _dt("2026-08-12T10:15:00"),
            },
            {
                "id": 9,
                "start": _dt("2026-08-13T10:00:00"),
                "end": _dt("2026-08-13T10:10:00"),
            },
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=7,
                decoder="satnogs_data_demod",
                received_at="2026-08-12T10:05:00",
                data_hex="c2a28a00ff",
                csp_crc_valid=True,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            ),
            _packet(
                observation_id=8,
                decoder="satnogs_data_demod",
                received_at="2026-08-12T10:12:00",  # 7 min later, within 15 min
                data_hex="c2a28a00ff",
                csp_crc_valid=True,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            ),
            _packet(
                observation_id=9,
                decoder="satnogs_data_demod",
                received_at="2026-08-13T10:05:00",  # a day later
                data_hex="c2a28a00ff",
                csp_crc_valid=True,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 2
    counts = sorted(result["packet_count"].to_list())
    assert counts == [1, 2]


def test_demod_packet_missing_crc_merges_with_sso_baseline() -> None:
    """satnogs_data_demod sometimes reports a packet with its trailing CSP
    CRC-32C already stripped; step 2 should complete it and match it to an
    sso_rx_replay baseline of the same underlying payload.
    """
    payload = bytes.fromhex("c2a28a0010")
    full_hex = (payload + crc32c(payload).to_bytes(4, "big")).hex()
    payload_hex = payload.hex()

    observations = _observations_df(
        [
            _OBS_1,
            {
                "id": 2,
                "start": _dt("2026-08-12T19:30:30"),
                "end": _dt("2026-08-12T19:40:30"),
            },
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=1,
                received_at="2026-08-12T19:35:00",
                data_hex=full_hex,
                data_length_bytes=len(payload) + 4,
            ),
            _packet(
                observation_id=2,
                decoder="satnogs_data_demod",
                received_at="2026-08-12T19:35:05",
                data_hex=payload_hex,  # missing its trailing CRC
                data_length_bytes=len(payload),
                csp_crc_valid=False,  # step 1 already flagged this invalid
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            ),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["data_hex"] == full_hex
    assert row["csp_crc_valid"] is True
    assert row["packet_count"] == 2
    assert json.loads(row["decoders"]) == ["satnogs_data_demod", "sso_rx_replay"]

    sources = json.loads(row["sources"])
    sources_by_decoder = {s["decoder"]: s for s in sources}
    assert sources_by_decoder["sso_rx_replay"]["csp_crc_source"] == "decoded"
    assert sources_by_decoder["satnogs_data_demod"]["csp_crc_source"] == "computed"


def test_demod_only_packet_gets_a_completed_self_consistent_crc() -> None:
    """A demod-only packet (no sso baseline at all) still gets its CRC
    completed, purely so the result is internally consistent -- not because
    it was independently verified by the satellite.
    """
    payload = bytes.fromhex("c2a28a00aabbccddeeff")
    observations = _observations_df(
        [
            {
                "id": 5,
                "start": _dt("2026-08-12T19:30:00"),
                "end": _dt("2026-08-12T19:40:00"),
            }
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=5,
                decoder="satnogs_data_demod",
                received_at="2026-08-12T19:35:05",
                data_hex=payload.hex(),
                data_length_bytes=len(payload),
                csp_crc_valid=False,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            )
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["csp_crc_valid"] is True
    assert row["csp_crc_source"] == "computed"

    is_valid, _computed, _received = verify_csp_packet_crc32c(
        bytes.fromhex(row["data_hex"])
    )
    assert is_valid
    assert row["data_length_bytes"] == len(payload) + 4


def test_demod_packet_with_already_valid_crc_is_left_alone() -> None:
    """A satnogs_data_demod row that already carries a valid CRC shouldn't
    have a second one appended on top.
    """
    payload = bytes.fromhex("c2a28a0011223344")
    full_hex = (payload + crc32c(payload).to_bytes(4, "big")).hex()

    observations = _observations_df(
        [
            {
                "id": 6,
                "start": _dt("2026-08-12T19:30:00"),
                "end": _dt("2026-08-12T19:40:00"),
            }
        ]
    )
    packets = _packets_df(
        [
            _packet(
                observation_id=6,
                decoder="satnogs_data_demod",
                received_at="2026-08-12T19:35:05",
                data_hex=full_hex,
                data_length_bytes=len(payload) + 4,
                csp_crc_valid=True,
                rssi_db=None,
                rs_corrected_error_count=None,
                rs_correctable=True,
                time_in_file_ms=None,
            )
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert len(result) == 1
    row = result.row(0, named=True)
    assert row["data_hex"] == full_hex
    assert row["data_length_bytes"] == len(payload) + 4

    sources = json.loads(row["sources"])
    assert sources[0]["csp_crc_source"] == "decoded"


def test_packet_with_wrong_csp_header_is_dropped() -> None:
    observations = _observations_df([_OBS_1])
    packets = _packets_df(
        [_packet(received_at="2026-08-12T19:35:00", data_hex="deadbeef00")]
    )

    result = compute_distinct_packets(packets, observations)

    assert result.is_empty()


def test_empty_data_hex_rows_are_dropped() -> None:
    observations = _observations_df([_OBS_1])
    packets = _packets_df(
        [
            _packet(received_at="2026-08-12T19:35:00", data_hex=""),
        ]
    )

    result = compute_distinct_packets(packets, observations)

    assert result.is_empty()


def test_output_schema_and_json_columns() -> None:
    observations = _observations_df([_OBS_1])
    packets = _packets_df([_packet(received_at="2026-08-12T19:35:00")])

    result = compute_distinct_packets(packets, observations)

    assert result.columns == [
        "packet_id",
        "data_hex",
        "data_length_bytes",
        "csp_crc_valid",
        "csp_crc_source",
        "received_at",
        "received_at_source",
        "rssi_db",
        "rs_corrected_error_count",
        "rs_correctable",
        "packet_count",
        "decoders",
        "observation_ids",
        "sources",
    ]

    row = result.row(0, named=True)
    assert isinstance(json.loads(row["decoders"]), list)
    assert isinstance(json.loads(row["observation_ids"]), list)
    assert isinstance(json.loads(row["sources"]), list)
    assert row["packet_id"] is not None
