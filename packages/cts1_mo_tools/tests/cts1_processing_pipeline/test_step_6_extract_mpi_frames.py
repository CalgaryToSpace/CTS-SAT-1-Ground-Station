import json
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames.pipeline import (
    MPI_BUFFER_LEN,
    MPI_COLLECTION_MAX_DURATION,
    MPI_SYNC,
    assign_mpi_collections,
    compute_mpi_frames,
    compute_mpi_frames_with_outcomes,
    mpi_crc16,
)

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_RECORDED_AT = datetime(2026, 8, 9, 6, 33, 10, tzinfo=UTC)
_DOWNLINKED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
_FRAME_LEN = 152  # Pixels 48..112, as flown.
_PACKET_DATA_LEN = 195
_BLOCK_MS = 4200
_POWER_ON_MS = 5000

# A real frame off the satellite, CRC and all.
_REAL_FRAME_HEX = (
    "0cffff0c086306681103ff3fff1b0f503070010407d307d207d107d007ef07b907ee07ff"
    "080907f2086f07e307f107ee07c60803083107e407bc07c70796084e074a086307c5080e"
    "079807b907f307c107c307c9077f081307b407d607f00835080a07b707fe083d07e107fd"
    "080207d80806079d07f60751082407b4081f0843080d078b07940818080e07ac083d07ce"
    "081c0812084b8043"
)


def _frame(counter: int, rng: random.Random) -> bytes:
    body = (
        MPI_SYNC
        + (counter % 0x10000).to_bytes(2, "big")
        + (1324).to_bytes(2, "big")  # 10.34 C
        + bytes([17])
        + bytes.fromhex("03ff3fff1b0f50")
        + bytes([48, 112])
        + (260).to_bytes(2, "big")
        + b"".join(rng.randrange(1800, 2200).to_bytes(2, "big") for _ in range(65))
    )
    return body + mpi_crc16(body).to_bytes(2, "big")


def _timestamp_json(uptime_ms: int, epoch_ms: int, source: str = "E") -> bytes:
    sync_epoch_ms = epoch_ms - uptime_ms + 1000
    # The firmware's "+<ms since sync>" part is computed from the truncated
    # epoch time instead of the uptime -- reproduce that bug.
    buggy_since_sync = (epoch_ms % 2**32) - (sync_epoch_ms - epoch_ms + uptime_ms)
    return (
        f'{{"uptime_ms":{uptime_ms},"timestamp":"{sync_epoch_ms}+'
        f'{buggy_since_sync:010d}_{source}","datetime":"2026-08-09T063310.114Z_'
        f'{source}","timestamp_ms":{epoch_ms}}}'
    ).encode()


def _header_json(uptime_ms: int, epoch_ms: int) -> bytes:
    return b'{"mpi_start":1,' + _timestamp_json(uptime_ms, epoch_ms)[1:]


def _epoch_ms(at: datetime) -> int:
    return round((at - _EPOCH).total_seconds() * 1000)


class _Recording:
    """A synthetic MPI science data file, laid out like the firmware does."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        blocks: int = 4,
        recorded_at: datetime = _RECORDED_AT,
        uptime_ms: int = 251_818_680,
        seed: int = 0,
        source: str = "E",
        stream_skew: int = 2,
        stale_frames: int = 0,
    ) -> None:
        rng = random.Random(seed)  # noqa: S311
        frame_count = (blocks * MPI_BUFFER_LEN) // _FRAME_LEN + 2
        # Left in the buffer by the previous recording (a firmware bug).
        stale = b"".join(_frame(1640 + i, rng) for i in range(stale_frames))
        # The MPI stream doesn't start on a frame boundary.
        self.stream = (
            stale
            + bytes(stream_skew)
            + b"".join(_frame(i, rng) for i in range(frame_count))
        )
        self.stream_skew = len(stale) + stream_skew
        header_epoch = _epoch_ms(recorded_at)
        self.timestamps = [
            (
                uptime_ms + _POWER_ON_MS + (k + 1) * _BLOCK_MS,
                header_epoch + _POWER_ON_MS + (k + 1) * _BLOCK_MS,
            )
            for k in range(blocks)
        ]
        data = bytearray(_header_json(uptime_ms, header_epoch))
        self.timestamp_offsets: list[int] = []
        for k, (up, ep) in enumerate(self.timestamps):
            data += self.stream[k * MPI_BUFFER_LEN : (k + 1) * MPI_BUFFER_LEN]
            self.timestamp_offsets.append(len(data))
            data += _timestamp_json(up, ep, source)
        data += (
            b'{"data_lost_bytes": 0, "time_taken_ms": 1, "reason_for_stopping": "X" }'
        )
        self.data = bytes(data)
        self.blocks = blocks

    def pixels_by_counter(self) -> dict[int, list[int]]:
        frames = self.stream[self.stream_skew :]
        return {
            i: [
                int.from_bytes(frames[s + 20 + 2 * p : s + 22 + 2 * p], "big")
                for p in range(65)
            ]
            for i, s in enumerate(range(0, len(frames), _FRAME_LEN))
        }

    def expected_frames(self, *, first_block: bool = True) -> dict[int, datetime]:
        """frame counter -> expected collected_at, for every frame that
        should come out (with or without those in the first block).
        """
        expected: dict[int, datetime] = {}
        counter = 0
        start = self.stream_skew
        while start + _FRAME_LEN <= self.blocks * MPI_BUFFER_LEN:
            end = start + _FRAME_LEN
            block = (end - 1) // MPI_BUFFER_LEN
            if block >= 1 or first_block:
                ep_b = self.timestamps[block][1]
                fraction = (MPI_BUFFER_LEN - end % MPI_BUFFER_LEN) % MPI_BUFFER_LEN
                expected[counter] = _EPOCH + timedelta(
                    milliseconds=ep_b - fraction / MPI_BUFFER_LEN * _BLOCK_MS
                )
            counter += 1
            start = end
        return expected

    def frames_in_first_block(self) -> int:
        return (MPI_BUFFER_LEN - self.stream_skew) // _FRAME_LEN


_packet_counter = 0


def _downlink(
    recording: _Recording,
    *,
    at: datetime = _DOWNLINKED_AT,
    observation_id: int = 1,
    skip_offsets: frozenset[int] = frozenset(),
) -> list[dict[str, Any]]:
    """BULK_FILE_DOWNLINK rows for the whole file, ~250ms apart."""
    global _packet_counter  # noqa: PLW0603
    rows: list[dict[str, Any]] = []
    for i, offset in enumerate(range(0, len(recording.data), _PACKET_DATA_LEN)):
        if offset in skip_offsets:
            continue
        _packet_counter += 1
        chunk = recording.data[offset : offset + _PACKET_DATA_LEN]
        rows.append(
            {
                "packet_id": f"pkt{_packet_counter:06d}",
                "packet_type": "BULK_FILE_DOWNLINK",
                "csp_crc_valid": True,
                "received_at": at + timedelta(seconds=0.25 * i),
                "received_at_source": "askew_demod_from_file",
                "decoders": '["askew_demod_from_file"]',
                "observation_ids": f"[{observation_id}]",
                "bulk_file_offset": offset,
                "bulk_data_hex": chunk.hex(),
            }
        )
    return rows


def _chunk_offset_containing(file_offset: int) -> int:
    return file_offset - file_offset % _PACKET_DATA_LEN


def _assert_times_match(result: pl.DataFrame, expected: dict[int, datetime]) -> None:
    for counter, collected_at in zip(
        result["frame_counter"], result["collected_at"], strict=True
    ):
        assert abs(collected_at - expected[counter]) < timedelta(milliseconds=1)


def test_crc_matches_real_frame() -> None:
    frame = bytes.fromhex(_REAL_FRAME_HEX)
    assert mpi_crc16(frame[:-2]) == int.from_bytes(frame[-2:], "big")


def test_empty_input_returns_empty_with_expected_schema() -> None:
    result = compute_mpi_frames(pl.DataFrame())
    assert result.height == 0
    assert result.schema["collected_at"] == pl.Datetime("us", "UTC")
    assert result.schema["pixels"] == pl.String


def _outcomes(table: pl.DataFrame) -> dict[str, int]:
    return dict(zip(table["outcome"], table["frames"], strict=True))


def test_full_downlink_recovers_and_times_every_frame() -> None:
    recording = _Recording()
    result, outcomes = compute_mpi_frames_with_outcomes(
        pl.DataFrame(_downlink(recording))
    )
    expected = recording.expected_frames()

    assert sorted(result["frame_counter"]) == sorted(expected)
    assert result.schema == compute_mpi_frames(pl.DataFrame()).schema
    _assert_times_match(result, expected)
    # The first block has no timestamp before it, so is extrapolated
    # backwards from the one after it.
    extrapolated = result.filter(pl.col("timing_method") == "extrapolated")
    assert extrapolated.height == recording.frames_in_first_block()
    assert extrapolated["anchor_span_blocks"].is_null().all()
    assert extrapolated["anchor_before_at"].is_null().all()
    assert set(
        result.filter(pl.col("timing_method") == "interpolated")["anchor_span_blocks"]
    ) == {1}
    assert _outcomes(outcomes) == {
        "kept: interpolated between two timestamps": result.height
        - extrapolated.height,
        "kept: extrapolated from one timestamp": extrapolated.height,
    }
    assert set(result["time_sync_source"]) == {"E"}
    # The synthetic timestamps were synced 1s into uptime, so time since
    # sync always trails uptime by exactly that.
    assert (result["obc_uptime_ms"] - result["ms_since_time_sync"] == 1000).all()
    row = result.row(0, named=True)
    assert row["firmware_version"] == 17
    assert row["board_temperature_c"] == 1324 / 128  # Exact in binary.
    assert len(json.loads(row["pixels"])) == 65
    assert row["first_received_at"] >= _DOWNLINKED_AT
    assert set(result["times_received"]) == {1}


def test_frames_split_by_a_timestamp_are_carved_back_together() -> None:
    recording = _Recording()
    result = compute_mpi_frames(pl.DataFrame(_downlink(recording)))
    # A frame spanning each timestamp JSON (apart from the stale first
    # block's) came out, with its bytes on both sides of the JSON.
    for json_offset in recording.timestamp_offsets[:-1]:
        straddling = result.filter(
            (pl.col("file_offset") < json_offset)
            & (pl.col("file_offset") > json_offset - _FRAME_LEN)
        )
        assert straddling.height == 1


def test_unsynced_timestamps_yield_nothing() -> None:
    recording = _Recording(source="N")
    result, outcomes = compute_mpi_frames_with_outcomes(
        pl.DataFrame(_downlink(recording))
    )
    assert result.height == 0
    assert set(outcomes["outcome"]) == {
        "lost: clock unsynced, or resynced, at its timestamp(s)"
    }
    assert outcomes["percent"].to_list() == [100.0]


def test_missing_middle_timestamp_interpolates_across_two_blocks() -> None:
    recording = _Recording()
    json_offset = recording.timestamp_offsets[1]
    skip = frozenset(
        {
            _chunk_offset_containing(json_offset),
            _chunk_offset_containing(json_offset + 140),
        }
    )
    result = compute_mpi_frames(pl.DataFrame(_downlink(recording, skip_offsets=skip)))

    two_block = result.filter(pl.col("anchor_span_blocks") == 2)
    assert two_block.height > 0
    # Linear interpolation over two equal blocks lands on the same times as
    # one block at a time, given the synthetic timestamps are evenly spaced.
    _assert_times_match(result, recording.expected_frames())


def test_two_missing_timestamps_interpolate_across_three_blocks() -> None:
    recording = _Recording(blocks=5)
    skip = frozenset(
        _chunk_offset_containing(json_offset + delta)
        for json_offset in recording.timestamp_offsets[1:3]
        for delta in (0, 140)
    )
    result = compute_mpi_frames(pl.DataFrame(_downlink(recording, skip_offsets=skip)))

    three_block = result.filter(pl.col("anchor_span_blocks") == 3)
    assert three_block.height > 2 * MPI_BUFFER_LEN // _FRAME_LEN
    _assert_times_match(result, recording.expected_frames())


def _gap_offsets(recording: _Recording) -> tuple[int, int]:
    """[start, end) chunk offsets of a run of chunks too long to bridge, in
    the middle of the third block.
    """
    start = _chunk_offset_containing(recording.timestamp_offsets[1] + 8_000)
    return start, start + 10 * _PACKET_DATA_LEN


def test_gap_too_long_to_bridge_is_joined_by_frame_counter() -> None:
    recording = _Recording()
    gap_start, gap_end = _gap_offsets(recording)
    skip = frozenset(range(gap_start, gap_end, _PACKET_DATA_LEN))
    result, outcomes = compute_mpi_frames_with_outcomes(
        pl.DataFrame(_downlink(recording, skip_offsets=skip))
    )

    # Interpolated straight across the gap, between the third block's own
    # timestamps, which each piece holds one of.
    third_block = result.filter(
        pl.col("file_offset").is_between(
            recording.timestamp_offsets[1],
            recording.timestamp_offsets[2] - _FRAME_LEN,
        )
    )
    assert set(third_block["timing_method"]) == {"interpolated"}
    assert third_block["anchor_before_at"].n_unique() == 1
    expected = recording.expected_frames()
    lost = set(expected) - set(result["frame_counter"])
    assert 0 < len(lost) <= (gap_end - gap_start) // _FRAME_LEN + 2
    _assert_times_match(result, expected)
    assert all(outcome.startswith("kept") for outcome in outcomes["outcome"])


def test_frames_beyond_a_downlink_gap_are_extrapolated() -> None:
    recording = _Recording()
    # The two sides of the gap downlinked on different passes, so they
    # can't be joined: the third block's data is split between two
    # separately-reassembled pieces, each holding only one of its
    # timestamps.
    gap_start, gap_end = _gap_offsets(recording)
    every_offset = frozenset(range(0, len(recording.data), _PACKET_DATA_LEN))
    rows = _downlink(
        recording, skip_offsets=frozenset(o for o in every_offset if o >= gap_start)
    ) + _downlink(
        recording,
        at=_DOWNLINKED_AT + timedelta(hours=2),
        observation_id=2,
        skip_offsets=frozenset(o for o in every_offset if o < gap_end),
    )
    result, outcomes = compute_mpi_frames_with_outcomes(pl.DataFrame(rows))

    third_block = result.filter(
        pl.col("file_offset").is_between(
            recording.timestamp_offsets[1],
            recording.timestamp_offsets[2] - _FRAME_LEN,
        )
    )
    assert set(third_block["timing_method"]) == {"extrapolated"}
    # Both sides of the gap: after the earlier timestamp and before the later.
    assert third_block["anchor_before_at"].null_count() > 0
    assert third_block["anchor_after_at"].null_count() > 0
    _assert_times_match(result, recording.expected_frames())
    assert all(outcome.startswith("kept") for outcome in outcomes["outcome"])


def test_stale_first_block_is_dropped() -> None:
    recording = _Recording(stale_frames=20)
    result, outcomes = compute_mpi_frames_with_outcomes(
        pl.DataFrame(_downlink(recording))
    )

    expected = recording.expected_frames(first_block=False)
    assert sorted(result["frame_counter"]) == sorted(expected)
    _assert_times_match(result, expected)
    assert (
        _outcomes(outcomes)["lost: stale first block of a recording (firmware bug)"]
        == 20 + recording.frames_in_first_block()
    )


def test_missing_chunks_lose_only_their_own_frames() -> None:
    recording = _Recording()
    skipped = 30_000 - 30_000 % _PACKET_DATA_LEN
    result = compute_mpi_frames(
        pl.DataFrame(_downlink(recording, skip_offsets=frozenset({skipped})))
    )
    expected = recording.expected_frames()
    lost = set(expected) - set(result["frame_counter"])
    assert 1 <= len(lost) <= 3
    _assert_times_match(result, expected)


def test_corrupt_frame_drops_its_whole_span() -> None:
    recording = _Recording()
    rows = _downlink(recording)
    # Flip a pixel byte in the middle of the third block.
    target = recording.timestamp_offsets[1] + 10_000
    for row in rows:
        if row["bulk_file_offset"] == _chunk_offset_containing(target):
            data = bytearray.fromhex(row["bulk_data_hex"])
            data[target - row["bulk_file_offset"]] ^= 0x01
            row["bulk_data_hex"] = data.hex()
    result, outcomes = compute_mpi_frames_with_outcomes(pl.DataFrame(rows))
    interpolated = result.filter(pl.col("timing_method") == "interpolated")
    assert interpolated["anchor_before_at"].n_unique() == recording.blocks - 2
    # Nor is it extrapolated into, its timestamps having been received.
    assert _outcomes(outcomes)["lost: CRC/frame counter mismatch in its block(s)"] > 0


def test_repeated_downlinks_dont_duplicate_frames() -> None:
    recording = _Recording()
    rows = _downlink(recording) + _downlink(
        recording, at=_DOWNLINKED_AT + timedelta(hours=2), observation_id=2
    )
    result = compute_mpi_frames(pl.DataFrame(rows))
    assert result.height == len(recording.expected_frames())
    assert result["frame_counter"].is_unique().all()
    assert set(result["times_received"]) == {2}


def test_different_recordings_at_same_offsets_stay_apart() -> None:
    first = _Recording(seed=1)
    second = _Recording(
        seed=2, recorded_at=_RECORDED_AT + timedelta(days=3), uptime_ms=12_345_678
    )
    # Each downlink misses chunks the other has, so a careless merge by
    # offset alone would fill one's holes with the other's bytes.
    rows = _downlink(first, skip_offsets=frozenset({195 * 150, 195 * 300})) + _downlink(
        second,
        at=_DOWNLINKED_AT + timedelta(hours=2),
        observation_id=2,
        skip_offsets=frozenset({195 * 200, 195 * 350}),
    )
    result = compute_mpi_frames(pl.DataFrame(rows))

    cutoff = _RECORDED_AT + timedelta(days=1)
    for recording, part in (
        (first, result.filter(pl.col("collected_at") < cutoff)),
        (second, result.filter(pl.col("collected_at") >= cutoff)),
    ):
        expected = recording.expected_frames()
        assert part.height > 0
        _assert_times_match(part, expected)
        # Every frame recovered is this recording's own, not the other's
        # frame with the same counter at the same offset.
        own_pixels = recording.pixels_by_counter()
        for row in part.iter_rows(named=True):
            assert json.loads(row["pixels"]) == own_pixels[row["frame_counter"]]


@pytest.mark.parametrize(
    ("uptime_ms", "downlinked_later"),
    [
        # Laid out byte for byte like the first, so their frame counters
        # line up across the gap just as one recording's would -- only
        # being downlinked on a different pass gives it away.
        (312_345_678, timedelta(hours=2)),
        # Downlinked on the same pass, but a digit shorter uptime makes
        # every JSON marker, so every frame after the header, a byte off.
        (12_345_678, timedelta(minutes=5)),
    ],
)
def test_different_recordings_either_side_of_a_gap_stay_apart(
    uptime_ms: int, downlinked_later: timedelta
) -> None:
    first = _Recording(seed=1)
    second = _Recording(
        seed=2, recorded_at=_RECORDED_AT + timedelta(days=3), uptime_ms=uptime_ms
    )
    # The start of one, and a stretch of the other after a gap. The stretch
    # holds no timestamp of its own, so joined, its frames would be
    # extrapolated from the first's.
    gap_start, gap_end = _gap_offsets(first)
    stretch_end = _chunk_offset_containing(first.timestamp_offsets[2])
    every_offset = frozenset(range(0, len(first.data), _PACKET_DATA_LEN))
    rows = _downlink(
        first, skip_offsets=frozenset(o for o in every_offset if o >= gap_start)
    ) + _downlink(
        second,
        at=_DOWNLINKED_AT + downlinked_later,
        observation_id=2,
        skip_offsets=frozenset(
            o for o in every_offset if not gap_end <= o < stretch_end
        ),
    )
    result = compute_mpi_frames(pl.DataFrame(rows))

    # Just the first's frames -- including those it holds of the block the
    # gap is in, which a bad join would drop or replace.
    third_block = result.filter(pl.col("file_offset") >= first.timestamp_offsets[1])
    assert third_block.height > 0
    assert set(third_block["timing_method"]) == {"extrapolated"}
    assert (result["file_offset"] < gap_start).all()
    _assert_times_match(result, first.expected_frames())
    own_pixels = first.pixels_by_counter()
    for row in result.iter_rows(named=True):
        assert json.loads(row["pixels"]) == own_pixels[row["frame_counter"]]


def _frames(*runs: tuple[datetime, list[int]]) -> pl.DataFrame:
    """Frames 31ms apart from each (start time, counters) run."""
    times: list[datetime] = []
    counters: list[int] = []
    for start, run_counters in runs:
        times += [
            start + timedelta(milliseconds=31 * i) for i in range(len(run_counters))
        ]
        counters += run_counters
    return pl.DataFrame(
        {"collected_at": times, "frame_counter": counters},
        schema={"collected_at": pl.Datetime("us", "UTC"), "frame_counter": pl.Int64},
    )


def _collection_starts(frames: pl.DataFrame) -> list[tuple[datetime, int]]:
    return (
        assign_mpi_collections(frames)
        .select("collection_started_at", "collection_first_frame_counter")
        .unique(maintain_order=True)
        .rows()
    )


def test_counter_reset_starts_a_new_collection() -> None:
    second_start = _RECORDED_AT + timedelta(seconds=60)
    frames = _frames(
        (_RECORDED_AT, list(range(100, 200))), (second_start, list(range(5, 50)))
    )
    assert _collection_starts(frames) == [(_RECORDED_AT, 100), (second_start, 5)]


def test_counter_wraparound_starts_a_new_collection() -> None:
    frames = _frames((_RECORDED_AT, [65534, 65535, 0, 1]))
    assert _collection_starts(frames) == [
        (_RECORDED_AT, 65534),
        (_RECORDED_AT + timedelta(milliseconds=62), 0),
    ]


def test_slightly_out_of_order_frames_stay_in_one_collection() -> None:
    frames = _frames((_RECORDED_AT, [*range(100), *range(90, 200)]))
    assert _collection_starts(frames) == [(_RECORDED_AT, 0)]


def test_small_counter_drop_long_after_starts_a_new_collection() -> None:
    later = _RECORDED_AT + timedelta(seconds=30)
    frames = _frames((_RECORDED_AT, [20, 21]), (later, [3, 4]))
    assert _collection_starts(frames) == [(_RECORDED_AT, 20), (later, 3)]


def test_collection_ends_after_twenty_minutes() -> None:
    later = _RECORDED_AT + MPI_COLLECTION_MAX_DURATION
    frames = _frames((_RECORDED_AT, [0, 1]), (later, [40_000, 40_001]))
    assert _collection_starts(frames) == [(_RECORDED_AT, 0), (later, 40_000)]


def test_full_downlink_is_one_collection() -> None:
    recording = _Recording()
    result = compute_mpi_frames(pl.DataFrame(_downlink(recording)))
    assert result["collection_started_at"].unique().to_list() == [
        result["collected_at"].min()
    ]
    assert set(result["collection_first_frame_counter"]) == {0}
    assert result["frame_counter"].n_unique() == result.height
