"""When each decode was transmitted, according to the satellite's own clock.

Only two things are trusted: the onboard timestamp in live beacons, and how
far into a recording each frame starts (`time_in_file_ms`, from the baseline
decoders only, which agree with each other to under a millisecond). Neither
`received_at` nor any ground station's clock is used.

Within one recording, every live beacon gives the offset between the
satellite's clock and time-in-file. That offset is linearly interpolated to
every other frame of the recording, or taken from the nearest beacon for a
frame with a beacon on one side only.

  - Historical beacons (replayed from onboard storage) are excluded by only
    accepting the exact end markers of live ones (`LIVE_END_MESSAGES`): a
    truncated one has lost its " S5", so not seeing that isn't enough.
  - A change in `unix_epoch_time_ms - uptime_ms` is the clock being set (or
    the OBC rebooting). Nothing is interpolated across one: a frame between
    two such beacons goes with the nearer one.
  - A beacon is stamped when it's built, so one that then waits to be sent
    can only make the offset look too low. One in such a dip isn't used.
    A lasting change of level (audio dropped from the recording) isn't a
    dip, and is kept.

The fields are sliced straight out of the bytes, as running the packet
decoder here would be far too slow.
"""

__all__ = [
    "BEACON_TYPE_HEX",
    "estimate_transmit_times",
]

import struct

import polars as pl

from cts1_mo_tools.cts1_decode_satnogs_packets import (
    BEACON_FIXED_SIZE,
    CSP_HEADER_SIZE,
    END_MESSAGE_SIZE,
    FRIENDLY_MESSAGE_SIZE,
    MAX_VALID_EPOCH_MS,
    PACKET_TYPE_MAP_INV,
)

LIVE_END_MESSAGES = {
    "BEACON_BASIC": [b"END\0"],
    "BEACON_EXTENDED": [f" X{v}\0".encode() for v in range(2, 10)],
    # The field is 4 bytes, so from v10 on either the space or the null has to go.
    "BEACON_EXTENDED_V5PLUS": [
        *(f" X{v}\0".encode() for v in range(5, 10)),
        *(f"X{v}\0".encode() for v in range(10, 100)),
        *(f" X{v}".encode() for v in range(10, 100)),
    ],
}
BEACON_TYPE_HEX = [f"{PACKET_TYPE_MAP_INV[name]:02x}" for name in LIVE_END_MESSAGES]

UPTIME_OFFSET = CSP_HEADER_SIZE + struct.calcsize("<B4sBB")
EPOCH_OFFSET = CSP_HEADER_SIZE + struct.calcsize("<B4sBBII")
END_MESSAGE_OFFSET = CSP_HEADER_SIZE + BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE

# The two clocks are read a millisecond or so apart, so can differ by that much.
CLOCK_STEP_TOLERANCE_MS = 5
# Above the normal jitter, and the ~40ms by which the beacon types differ in
# how long they take from being stamped to being sent.
LATE_BEACON_TOLERANCE_MS = 250


def _live_beacons(decodes: pl.LazyFrame) -> pl.LazyFrame:
    """One row per live beacon per recording, in time order."""
    data = pl.col("_data")
    packet_type = data.bin.slice(CSP_HEADER_SIZE, 1)
    end_message = data.bin.slice(END_MESSAGE_OFFSET, END_MESSAGE_SIZE)
    is_live = pl.any_horizontal(
        (packet_type == pl.lit(bytes([PACKET_TYPE_MAP_INV[name]])))
        & end_message.is_in(markers)
        for name, markers in LIVE_END_MESSAGES.items()
    )
    return (
        decodes.filter(is_live & pl.col("csp_crc_valid"))
        .with_columns(
            _clock=data.bin.slice(EPOCH_OFFSET, 8)
            .bin.reinterpret(dtype=pl.UInt64, endianness="little")
            .cast(pl.Int64, strict=False),  # null (so dropped below) if absurd
            _uptime=data.bin.slice(UPTIME_OFFSET, 4)
            .bin.reinterpret(dtype=pl.UInt32, endianness="little")
            .cast(pl.Int64),
        )
        .filter(pl.col("_clock").is_between(1, MAX_VALID_EPOCH_MS))
        .drop("_data")
        # Both baseline decoders usually report the same beacon.
        .sort("observation_id", "_content_id", "_trust_rank", "_t")
        .unique(["observation_id", "_content_id"], keep="first", maintain_order=True)
        .sort("observation_id", "_t", "_content_id")
    )


def _clock_anchors(beacons: pl.LazyFrame) -> pl.LazyFrame:
    """The beacons worth interpolating between: `_offset` is satellite clock
    minus time-in-file, and `_segment` numbers the stretches between clock
    steps within each recording."""
    segment = ["observation_id", "_segment"]
    offset = pl.col("_offset")
    highest_before = offset.cum_max().over(segment)
    highest_after = offset.cum_max(reverse=True).over(segment)
    in_dip = (
        pl.min_horizontal(highest_before, highest_after) - offset
        > LATE_BEACON_TOLERANCE_MS
    )
    # A dip needs a beacon on each side, so the first and last are instead
    # compared with their one neighbour.
    previous = offset.shift(1).over(segment)
    following = offset.shift(-1).over(segment)
    late_at_either_end = (
        (previous.is_null() & (following - offset > LATE_BEACON_TOLERANCE_MS))
        | (following.is_null() & (previous - offset > LATE_BEACON_TOLERANCE_MS))
    ).fill_null(value=False)

    clock_stepped = (
        (pl.col("_clock") - pl.col("_uptime")).diff().abs() > CLOCK_STEP_TOLERANCE_MS
    ).fill_null(value=False)
    return (
        beacons.with_columns(
            _offset=pl.col("_clock") - pl.col("_t"),
            _segment=clock_stepped.cum_sum().over("observation_id"),
        )
        .filter(~in_dip)
        .filter(~late_at_either_end)
        .select("observation_id", "_t", "_offset", "_segment")
    )


def estimate_transmit_times(rows: pl.LazyFrame) -> pl.LazyFrame:
    """`(_row_id, observation_id, _transmitted_at)` for every baseline decode
    in `rows` (as prepared by step 2) that has a `time_in_file_ms`. The time
    is null where its recording has no usable beacon.
    """
    decodes = (
        rows.filter(pl.col("_trust_rank").is_not_null())
        .select(
            "_row_id",
            "observation_id",
            "_content_id",
            "_trust_rank",
            "_data",
            "csp_crc_valid",
            _t=pl.col("time_in_file_ms").cast(pl.Float64),
        )
        .filter(pl.col("_t").is_not_null())
    )
    anchors = _clock_anchors(_live_beacons(decodes))
    before = anchors.select(
        "observation_id",
        "_t",
        _t0="_t",
        _offset0="_offset",
        _segment0="_segment",
    )
    after = anchors.select(
        "observation_id",
        "_t",
        _t1="_t",
        _offset1="_offset",
        _segment1="_segment",
    )

    t, t0, t1 = pl.col("_t"), pl.col("_t0"), pl.col("_t1")
    offset0, offset1 = pl.col("_offset0"), pl.col("_offset1")
    offset = (
        pl.when((pl.col("_segment0") == pl.col("_segment1")) & (t1 > t0))
        .then(offset0 + (offset1 - offset0) * (t - t0) / (t1 - t0))
        .when(t1.is_null() | (t - t0 <= t1 - t))
        .then(offset0)
        .otherwise(offset1)
    )
    # `check_sortedness` can't see sorting within `by` groups, and only warns.
    return (
        decodes.select("_row_id", "observation_id", "_t")
        .sort("observation_id", "_t")
        .join_asof(before, on="_t", by="observation_id", check_sortedness=False)
        .join_asof(
            after,
            on="_t",
            by="observation_id",
            strategy="forward",
            check_sortedness=False,
        )
        .select(
            "_row_id",
            "observation_id",
            # Whole milliseconds, like the beacons' own timestamps.
            _transmitted_at=(t + offset)
            .round()
            .cast(pl.Int64)
            .cast(pl.Datetime("ms", "UTC"))
            .dt.cast_time_unit("us"),
        )
    )
