from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_12_detect_tx_outages.pipeline import (
    compute_tx_outages,
)

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _h(hours: float) -> datetime:
    return _BASE + timedelta(hours=hours)


def _history(
    passes: Sequence[tuple[int, float, bool]],
    uncorrectable_at: Sequence[float] = (),
    no_audio_at: Sequence[float] = (),
) -> tuple[pl.DataFrame, pl.DataFrame, pl.LazyFrame, pl.LazyFrame]:
    """Three reliable stations (1, 2, 3) each heard 60 earlier passes, plus
    `passes`: (station, start hour, heard?), each 10 min long. Each heard
    pass got one beacon, 5 min in, its OBC uptime the pass's start hour.
    Passes starting at `no_audio_at` hours only ran satnogs_client_live_data
    (the audio download failed)."""
    all_passes = [(s, -100.0 + i, True) for i in range(60) for s in (1, 2, 3)]
    all_passes += list(passes)
    observations = pl.DataFrame(
        {
            "id": list(range(len(all_passes))),
            "ground_station": [s for s, _, _ in all_passes],
            "station_name": [f"GS{s}" for s, _, _ in all_passes],
            "start": [_h(t) for _, t, _ in all_passes],
            "end": [_h(t) + timedelta(minutes=10) for _, t, _ in all_passes],
        }
    )
    decoder_runs = pl.DataFrame(
        {
            "observation_id": [*observations["id"]] * 2,
            "decoder": ["askew_demod_from_file"] * observations.height
            + ["satnogs_client_live_data"] * observations.height,
            "error": [
                "audio download: HTTPError: 503" if t in no_audio_at else None
                for _, t, _ in all_passes
            ]
            + [None] * observations.height,
        },
        schema={"observation_id": pl.Int64, "decoder": pl.String, "error": pl.String},
    )
    heard = [(i, t) for i, (_, t, hit) in enumerate(all_passes) if hit]
    packets = pl.DataFrame(
        {
            "observation_id": [i for i, _ in heard] + [0] * len(uncorrectable_at),
            "received_at": [_h(t) + timedelta(minutes=5) for _, t in heard]
            + [_h(t) for t in uncorrectable_at],
            "rs_correctable": [True] * len(heard) + [False] * len(uncorrectable_at),
            "csp_crc_valid": [True] * len(heard) + [False] * len(uncorrectable_at),
        }
    )
    beacons = pl.DataFrame(
        {
            "received_at": [_h(t) + timedelta(minutes=5) for _, t in heard],
            "packet_type": ["BEACON_BASIC"] * len(heard),
            "csp_crc_valid": [True] * len(heard),
            "uptime_sec": [t for _, t in heard],
            "eps_uptime_sec": [7] * len(heard),
            "reboot_reason": ["BROWNOUT_RESET"] * len(heard),
            "eps_reset_cause": ["COMMANDED"] * len(heard),
        }
    )
    return observations, decoder_runs, packets.lazy(), beacons.lazy()


def test_outage_needs_two_reliable_stations_missing() -> None:
    passes = [
        (1, 0, True),
        # 0-10h: a single station misses -- not an outage.
        (1, 4, False),
        (2, 10, True),
        # 10-20h: two stations miss -- an outage.
        (1, 13, False),
        (2, 15, False),
        (3, 20, True),
    ]
    outages = compute_tx_outages(*_history(passes, uncorrectable_at=[11, 19.9]))
    assert outages.height == 1
    row = outages.row(0, named=True)
    assert row["last_good_packet_at"] == _h(10) + timedelta(minutes=5)
    assert row["first_good_packet_after_at"] == _h(20) + timedelta(minutes=5)
    # The one at 19.9h is after the first miss (the recovery pass), so not it.
    assert row["last_uncorrectable_packet_at"] == _h(11)
    assert not row["is_ongoing"]
    assert row["missed_stations"] == ["GS1 (#1)", "GS2 (#2)"]
    assert 0 < row["probability_by_chance"] < 0.01
    # The first beacon after recovery: the 20h pass's.
    assert row["recovery_beacon_at"] == _h(20) + timedelta(minutes=5)
    assert row["recovery_obc_uptime_sec"] == 20
    assert row["recovery_eps_uptime_sec"] == 7
    assert row["recovery_obc_reboot_reason"] == "BROWNOUT_RESET"
    assert row["recovery_eps_reboot_reason"] == "COMMANDED"


def test_ongoing_outage() -> None:
    # The misses are within 3h of the last good packet, but a later
    # (non-reliable, also silent) pass shows the silence lasting past 3h.
    passes = [(1, 0, True), (2, 1, False), (3, 2, False), (9, 4, False)]
    outages = compute_tx_outages(*_history(passes))
    assert outages["is_ongoing"].to_list() == [True]
    assert outages["first_good_packet_after_at"].to_list() == [None]
    assert outages["recovery_beacon_at"].to_list() == [None]


def test_ongoing_gap_under_3h_is_not_an_outage() -> None:
    passes = [(1, 0, True), (2, 1, False), (3, 2, False)]
    assert compute_tx_outages(*_history(passes)).is_empty()


def test_pass_without_audio_is_not_a_miss() -> None:
    # Two misses, but one's audio never downloaded: only one station missed.
    passes = [(1, 0, True), (2, 4, False), (3, 6, False), (1, 10, True)]
    assert compute_tx_outages(*_history(passes, no_audio_at=[6])).is_empty()
    assert compute_tx_outages(*_history(passes)).height == 1


def test_short_gap_is_not_an_outage() -> None:
    passes = [(1, 0, True), (2, 0.5, False), (3, 1, False), (1, 2, True)]
    assert compute_tx_outages(*_history(passes)).is_empty()
