import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from cts1_mo_tools.cts1_decode_satnogs_packets import crc32c, decode_packet_safe
from cts1_mo_tools.cts1_processing_pipeline import partitions
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    pipeline as step_2_pipeline,
)
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    sat_clock,
)
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets.pipeline import (
    compute_distinct_packets,
)
from cts1_mo_tools.cts1_processing_pipeline.step_3_decode_packets.pipeline import (
    compute_decoded_packets,
)
from polars.testing import assert_frame_equal

# Just before a month boundary, for the incremental tests.
_START = datetime(2026, 8, 31, 23, 55, tzinfo=UTC)
# Nothing like `_START`: the result mustn't come from the ground's clock.
_SAT_START = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)

_BASIC = (0x01, b"END\0", 130)
_V5 = (0x21, b" X5\0", 198)
_LEAD_MS = {0x01: 390, 0x20: 430, 0x21: 430}


def _with_crc(body: bytes) -> str:
    return (body + crc32c(body).to_bytes(4, "big")).hex()


def _beacon_hex(
    sat_ms: int,
    kind: tuple[int, bytes, int] = _BASIC,
    *,
    uptime_ms: int | None = None,
) -> str:
    """A beacon transmitted at `sat_ms`, so stamped a little before that."""
    packet_type, end_message, size = kind
    sat_ms -= _LEAD_MS.get(packet_type, 0)
    payload = bytearray(size)
    uptime_ms = 10_000_000 + sat_ms if uptime_ms is None else uptime_ms
    epoch_ms = int(_SAT_START.timestamp() * 1000) + sat_ms
    struct.pack_into(
        "<B4sBBIIQ", payload, 0, packet_type, b"CTS1", 1, 0, uptime_ms, 0, epoch_ms
    )
    payload[126:130] = end_message
    return _with_crc(bytes.fromhex("c2a28a00") + payload)


def _other_hex(message: bytes = b"hello") -> str:
    return _with_crc(bytes.fromhex("c2a28a00") + b"\x03" + message + b"\n")


def _log_hex(
    sat_ms: int, *, synced_ms: int = -600_000, source: bytes = b"E", text: bytes = b""
) -> str:
    """A log message transmitted at `sat_ms`, of a clock set at `synced_ms`."""
    synced_at = int(_SAT_START.timestamp() * 1000) + synced_ms
    prefix = f"{synced_at}+{sat_ms - synced_ms:010d}_".encode() + source
    return _other_hex(prefix + b" [A:OBC:INFO]: " + text)


def _decode(t_ms: float, data_hex: str, **overrides: Any) -> dict[str, Any]:
    return {
        "observation_id": 1,
        "decoder": "askew_demod_from_file",
        "received_at": _START + timedelta(milliseconds=t_ms),
        "time_in_file_ms": float(t_ms),
        "data_hex": data_hex,
        "data_length_bytes": len(data_hex) // 2,
        "rssi_db": -1.0,
        "rs_corrected_error_count": 0,
        "rs_correctable": True,
        "csp_crc_valid": True,
        **overrides,
    }


def _observations() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"id": i, "start": _START, "end": _START + timedelta(minutes=10)}
            for i in (1, 2, 3)
        ]
    )


def _sat_ms(*decodes: dict[str, Any]) -> dict[str, int | None]:
    """Each content's result, as milliseconds after `_SAT_START`."""
    result = compute_distinct_packets(pl.DataFrame(list(decodes)), _observations())
    return {
        data_hex: None if at is None else round((at - _SAT_START).total_seconds() * 1e3)
        for data_hex, at in result.select(
            "data_hex", "transmitted_at_sat_clock"
        ).iter_rows()
    }


def test_interpolates_and_extends_from_the_nearest_beacon() -> None:
    first, middle, last = (_other_hex(m) for m in (b"first", b"middle", b"last"))
    b1, b2 = _beacon_hex(1_000), _beacon_hex(21_100, _V5)

    result = _sat_ms(
        _decode(5_000, first),
        _decode(10_000, b1),
        _decode(20_000, middle),
        _decode(30_000, b2),
        _decode(40_000, last),
    )

    assert result == {
        first: -4_000,
        b1: 1_000,
        middle: 11_050,
        b2: 21_100,
        last: 31_100,
    }


def test_ignores_received_at() -> None:
    packet = _other_hex()
    beacon = _beacon_hex(1_000)
    skewed = _START + timedelta(seconds=47)

    result = _sat_ms(
        _decode(10_000, beacon),
        _decode(20_000, packet, received_at=skewed),
    )

    assert result[packet] == 11_000


def test_decodes_are_merged_by_when_the_satellite_sent_them() -> None:
    """A repeat of the same content 25 s later (a file downlinked again) is
    its own packet, while one transmission caught by a station whose clock
    is 90 s off is still one -- whatever `received_at` says."""
    packet, beacon = _other_hex(b"chunk"), _beacon_hex(1_000)

    def late_by_90_s(t_ms: float, data_hex: str) -> dict[str, Any]:
        return _decode(
            t_ms,
            data_hex,
            observation_id=2,
            received_at=_START + timedelta(milliseconds=t_ms + 90_000),
        )

    result = compute_distinct_packets(
        pl.DataFrame(
            [
                _decode(0, beacon),
                _decode(10_000, packet),
                _decode(35_000, packet),
                late_by_90_s(0, beacon),
                late_by_90_s(10_000, packet),
            ]
        ),
        _observations(),
    ).filter(pl.col("data_hex") == packet)

    assert sorted(
        (round((at - _SAT_START).total_seconds() * 1e3), count)
        for at, count in result.select(
            "transmitted_at_sat_clock", "packet_count"
        ).iter_rows()
    ) == [(11_000, 2), (36_000, 1)]


def test_field_offsets_match_the_full_decoder() -> None:
    beacon = _beacon_hex(12_345, _V5)
    decoded = decode_packet_safe(beacon)
    assert decoded is not None

    result = compute_distinct_packets(
        pl.DataFrame([_decode(1_000, beacon)]), _observations()
    )

    assert result["transmitted_at_sat_clock"][0] == datetime.fromisoformat(
        decoded["utc_time"]
    ) + timedelta(milliseconds=430)


@pytest.mark.parametrize(
    "kind",
    [
        _BASIC,
        (0x20, b" X2\0", 198),
        (0x20, b" X4\0", 198),
        _V5,
        (0x21, b" X6\0", 198),
        (0x21, b" X9\0", 198),
        (0x21, b"X10\0", 198),
        (0x21, b" X10", 198),
        (0x21, b"X99\0", 198),
    ],
)
def test_live_beacons_are_used(kind: tuple[int, bytes, int]) -> None:
    packet = _other_hex()

    result = _sat_ms(_decode(0, _beacon_hex(500, kind)), _decode(1_000, packet))

    assert result[packet] == 1_500


@pytest.mark.parametrize(
    "kind",
    [
        (0x21, b" S5\0", 198),
        (0x21, b" S6\0", 198),
        (0x21, b"S10\0", 198),
        (0x21, b" S10", 198),
        (0x21, b" X4\0", 198),
        (0x21, b"END\0", 198),
        (0x21, b"\0\0\0\0", 198),
        (0x20, b" S4\0", 198),
        (0x01, b" S5\0", 130),
        (0x03, b"END\0", 130),
    ],
)
def test_other_beacons_are_not_used(kind: tuple[int, bytes, int]) -> None:
    packet = _other_hex()

    result = _sat_ms(_decode(0, _beacon_hex(500, kind)), _decode(1_000, packet))

    assert result[packet] is None


@pytest.mark.parametrize("length", [5, 20, 30, 133])
def test_truncated_beacons_are_not_used(length: int) -> None:
    """A historical beacon cut short before its " S5" has a valid CRC once
    step 2 has computed one for it."""
    truncated = _with_crc(bytes.fromhex(_beacon_hex(500, _V5))[:length])
    packet = _other_hex()

    result = _sat_ms(_decode(0, truncated), _decode(1_000, packet))

    assert result[packet] is None


def test_historical_beacon_gets_its_replay_time() -> None:
    live = _beacon_hex(1_000, _V5)
    stored = _beacon_hex(-9_000_000, (0x21, b" S5\0", 198))

    result = _sat_ms(_decode(10_000, live), _decode(12_000, stored))

    assert result[stored] == 3_000


@pytest.mark.parametrize(
    "overrides",
    [
        {"csp_crc_valid": False},
        {"decoder": "gr_satellites_kiss"},
        {"decoder": "satnogs_client_live_data"},
        {"time_in_file_ms": None},
        {"observation_id": 2},
    ],
)
def test_untrusted_or_unrelated_beacons_are_not_used(overrides: dict[str, Any]) -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _beacon_hex(500), **overrides),
        _decode(1_000, packet),
    )

    assert result[packet] is None


@pytest.mark.parametrize("epoch_ms", [0, 2**63 + 5])
def test_invalid_timestamps_are_not_used(epoch_ms: int) -> None:
    raw = bytearray.fromhex(_beacon_hex(500))[:-4]
    struct.pack_into("<Q", raw, 19, epoch_ms)
    packet = _other_hex()

    result = _sat_ms(_decode(0, _with_crc(bytes(raw))), _decode(1_000, packet))

    assert result[packet] is None


@pytest.mark.parametrize("late_index", [0, 1, 2])
def test_beacon_that_waited_to_be_sent_is_not_used(late_index: int) -> None:
    times = [0, 20_000, 40_000]
    stamps = [t - 2_000 if i == late_index else t for i, t in enumerate(times)]
    beacons = [_beacon_hex(stamp) for stamp in stamps]

    result = _sat_ms(*map(_decode, times, beacons))

    # Including the late one itself: it was sent later than it was stamped.
    assert [result[beacon] for beacon in beacons] == times


def test_dropped_audio_is_not_mistaken_for_late_beacons() -> None:
    times = [0, 20_000, 40_000, 60_000]
    stamps = [0, 20_000, 45_000, 65_000]  # 5s of audio missing
    beacons = [_beacon_hex(stamp) for stamp in stamps]

    result = _sat_ms(*map(_decode, times, beacons))

    assert [result[beacon] for beacon in beacons] == stamps


def test_does_not_interpolate_across_a_clock_step() -> None:
    near_first, near_second = _other_hex(b"a"), _other_hex(b"b")

    result = _sat_ms(
        _decode(0, _beacon_hex(0, uptime_ms=100_000)),
        _decode(4_000, near_first),
        _decode(16_000, near_second),
        # Set forward by 100s: same uptime progression, different clock.
        _decode(20_000, _beacon_hex(120_000, uptime_ms=120_000)),
    )

    assert result[near_first] == 4_000
    assert result[near_second] == 116_000


def test_reads_a_real_log_prefix() -> None:
    log = _other_hex(b"1780208589000+0002833553_E [A:OBC:INFO]: Heartbeat")

    result = compute_distinct_packets(pl.DataFrame([_decode(0, log)]), _observations())

    # What the same message's own text says: "Datetime: 2026-05-31T071022.553Z".
    assert result["transmitted_at_sat_clock"][0] == datetime(
        2026, 5, 31, 7, 10, 22, 553_000, tzinfo=UTC
    )


@pytest.mark.parametrize("source", [b"E", b"G", b"T", b"A", b"Z"])
def test_logs_are_anchors(source: bytes) -> None:
    packet = _other_hex()
    log = _log_hex(2_000, source=source, text="🚀 not only ascii".encode())

    result = _sat_ms(_decode(0, log), _decode(1_000, packet))

    assert result == {log: 2_000, packet: 3_000}


@pytest.mark.parametrize(
    "message",
    [
        b"1780208589000+0002833553_N unsynced",
        b"0000000000000+0002833553_E synced at zero",
        b"9999999999999+0002833553_E too far in the future",
        b"1780208589000+K\xbc* garbled",
        b"1780208589000+0002833553\xab\xdd garbled",
        b"178020858900+0002833553_E too few digits",
        b"text 1780208589000+0002833553_E not at the start",
        b"1780208589000+00",
    ],
)
def test_other_logs_are_not_anchors(message: bytes) -> None:
    packet = _other_hex()

    result = _sat_ms(_decode(0, _other_hex(message)), _decode(1_000, packet))

    assert result[packet] is None


def test_log_with_bad_crc_is_not_an_anchor() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _log_hex(0), csp_crc_valid=False), _decode(1_000, packet)
    )

    assert result[packet] is None


def test_interpolates_between_a_beacon_and_a_log() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _beacon_hex(1_000)),
        _decode(10_000, packet),
        _decode(20_000, _log_hex(21_100)),
    )

    assert result[packet] == 11_050


def test_log_that_waited_to_be_sent_is_not_used() -> None:
    log = _log_hex(19_700)

    result = _sat_ms(
        _decode(0, _beacon_hex(0)),
        _decode(20_000, log),
        _decode(40_000, _beacon_hex(40_000)),
    )

    assert result[log] == 20_000


def test_does_not_interpolate_across_a_step_only_logs_show() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _log_hex(0)),
        _decode(4_000, packet),
        _decode(20_000, _log_hex(21_000, synced_ms=15_000)),
    )

    assert result[packet] == 4_000


def test_log_between_beacons_showing_a_step_joins_the_closer_side() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _beacon_hex(0, uptime_ms=100_000)),
        _decode(8_000, packet),
        # The clock was set forward 2s before this log.
        _decode(10_000, _log_hex(12_000)),
        _decode(20_000, _beacon_hex(22_000, uptime_ms=120_000)),
    )

    # Not 9_600, as interpolating across the step would give.
    assert result[packet] == 10_000


def test_big_step_between_the_only_beacon_and_the_only_log() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _beacon_hex(0)),
        _decode(4_000, packet),
        _decode(20_000, _log_hex(120_000)),
    )

    assert result[packet] == 4_000


def test_stamp_to_transmit_covers_every_beacon_type() -> None:
    assert set(sat_clock.BEACON_STAMP_TO_TRANSMIT_MS) == set(
        sat_clock.LIVE_END_MESSAGES
    )


def test_median_across_recordings() -> None:
    packet = _other_hex()

    result = _sat_ms(
        *(
            decode
            for obs, sat_ms in ((1, 1_000), (2, 1_100), (3, 9_000))
            for decode in (
                _decode(0, _beacon_hex(sat_ms), observation_id=obs),
                _decode(5_000, packet, observation_id=obs),
            )
        )
    )

    assert result[packet] == 6_100


def test_each_recording_counts_once_however_many_decoders() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _beacon_hex(1_000)),
        _decode(5_000, packet),
        _decode(5_000, packet, decoder="sso_rx_replay"),
        _decode(0, _beacon_hex(900), observation_id=2),
        _decode(5_000, packet, observation_id=2),
        _decode(0, _beacon_hex(800), observation_id=3),
        _decode(5_000, packet, observation_id=3),
    )

    assert result[packet] == 5_900


def test_repeats_of_a_packet_never_average_to_a_time_nothing_was_sent() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(0, _beacon_hex(1_000)),
        _decode(5_000, packet),
        _decode(55_000, packet),
    )

    assert result[packet] == 56_000


def test_late_anchor_just_after_a_clock_step_still_marks_the_step() -> None:
    sync_log = _log_hex(40_000, synced_ms=39_975)

    result = _sat_ms(
        _decode(0, _log_hex(0)),
        _decode(20_000, _log_hex(20_000)),
        # Set forward 19s, then this waited 150ms to be sent.
        _decode(21_150, sync_log),
        _decode(60_000, _log_hex(79_000, synced_ms=39_975)),
    )

    assert result[sync_log] == 40_000


def test_recording_with_only_the_other_baseline_decoder_counts() -> None:
    packet = _other_hex()

    result = _sat_ms(
        _decode(5_000, packet),
        _decode(0, _beacon_hex(700), observation_id=2, decoder="sso_rx_replay"),
        _decode(5_000, packet, observation_id=2, decoder="sso_rx_replay"),
    )

    assert result[packet] == 5_700


def test_carried_into_everything_decoded() -> None:
    distinct = compute_distinct_packets(
        pl.DataFrame([_decode(0, _beacon_hex(0)), _decode(1_000, _other_hex())]),
        _observations(),
    )

    decoded = compute_decoded_packets(distinct)

    assert_frame_equal(
        decoded.select("packet_id", "transmitted_at_sat_clock"),
        distinct.select("packet_id", "transmitted_at_sat_clock"),
    )


# The recordings here run over midnight into September at 300_000ms.
@pytest.mark.parametrize(
    ("before", "added"),
    [
        pytest.param(
            [_decode(290_000, _other_hex())],
            [_decode(310_000, _beacon_hex(5_000))],
            id="new_beacon_changes_previous_month",
        ),
        pytest.param(
            [
                _decode(280_000, _beacon_hex(1_000)),
                _decode(290_000, _other_hex()),
            ],
            [_decode(310_000, _other_hex(), observation_id=2)],
            id="recomputing_needs_previous_months_beacons",
        ),
        pytest.param(
            [
                _decode(290_000, _other_hex()),
                _decode(280_000, _beacon_hex(1_000), observation_id=2),
                _decode(291_000, _other_hex(), observation_id=2),
            ],
            [_decode(310_000, _beacon_hex(9_000))],
            id="recomputing_needs_other_recordings_beacons",
        ),
        pytest.param(
            [_decode(290_000, _other_hex())],
            [_decode(310_000, _log_hex(5_000))],
            id="new_log_changes_previous_month",
        ),
        pytest.param(
            [_decode(280_000, _log_hex(1_000)), _decode(290_000, _other_hex())],
            [_decode(310_000, _other_hex(), observation_id=2)],
            id="recomputing_needs_previous_months_logs",
        ),
    ],
)
def test_incremental_run_matches_a_full_one(
    tmp_path: Path, before: list[dict[str, Any]], added: list[dict[str, Any]]
) -> None:
    everything = pl.DataFrame(before + added).sort("received_at")
    incremental_dir, full_dir = tmp_path / "incremental", tmp_path / "full"
    for directory in (incremental_dir, full_dir):
        directory.mkdir()
        _observations().write_parquet(directory / "raw_observations.parquet")

    raw_path = incremental_dir / "raw_packets.parquet"
    partitions.write_frame(pl.DataFrame(before).sort("received_at"), raw_path)
    step_2_pipeline.run(data_dir=incremental_dir)
    partitions.write_frame(
        everything.filter(partitions.month_expr() == "2026-09"),
        raw_path,
        months={"2026-09"},
    )
    step_2_pipeline.run(data_dir=incremental_dir, incremental=True)
    partitions.write_frame(everything, full_dir / "raw_packets.parquet")
    step_2_pipeline.run(data_dir=full_dir)

    expected = pl.read_parquet(full_dir / step_2_pipeline.OUTPUT_FILENAME)
    assert expected["transmitted_at_sat_clock"].is_not_null().any()
    assert_frame_equal(
        pl.read_parquet(incremental_dir / step_2_pipeline.OUTPUT_FILENAME), expected
    )
