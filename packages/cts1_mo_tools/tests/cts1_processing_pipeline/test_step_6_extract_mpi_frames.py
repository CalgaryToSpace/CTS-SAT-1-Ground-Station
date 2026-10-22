import json
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from cts1_mo_tools.cts1_processing_pipeline.step_6_extract_mpi_frames.pipeline import (
    MPI_BUFFER_LEN,
    MPI_SYNC,
    compute_mpi_frames,
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
    ) -> None:
        rng = random.Random(seed)  # noqa: S311
        frame_count = (blocks * MPI_BUFFER_LEN) // _FRAME_LEN + 2
        # The MPI stream doesn't start on a frame boundary.
        self.stream = bytes(stream_skew) + b"".join(
            _frame(i, rng) for i in range(frame_count)
        )
        self.stream_skew = stream_skew
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

    def expected_frames(self) -> dict[int, datetime]:
        """frame counter -> expected collected_at, for every frame that
        should come out (all those ending after the first block).
        """
        expected: dict[int, datetime] = {}
        counter = 0
        start = self.stream_skew
        while start + _FRAME_LEN <= self.blocks * MPI_BUFFER_LEN:
            end = start + _FRAME_LEN
            block = (end - 1) // MPI_BUFFER_LEN
            if block >= 1:
                ep_a = self.timestamps[block - 1][1]
                ep_b = self.timestamps[block][1]
                fraction = (end - block * MPI_BUFFER_LEN) / MPI_BUFFER_LEN
                expected[counter] = _EPOCH + timedelta(
                    milliseconds=ep_a + fraction * (ep_b - ep_a)
                )
            counter += 1
            start = end
        return expected


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


def test_full_downlink_recovers_and_times_every_frame_after_first_block() -> None:
    recording = _Recording()
    result = compute_mpi_frames(pl.DataFrame(_downlink(recording)))
    expected = recording.expected_frames()

    assert sorted(result["frame_counter"]) == sorted(expected)
    assert result.schema == compute_mpi_frames(pl.DataFrame()).schema
    _assert_times_match(result, expected)
    assert set(result["anchor_span_blocks"]) == {1}
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
    assert compute_mpi_frames(pl.DataFrame(_downlink(recording))).height == 0


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
    result = compute_mpi_frames(pl.DataFrame(rows))
    spans = set(result["anchor_before_at"])
    assert len(spans) == recording.blocks - 2


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
