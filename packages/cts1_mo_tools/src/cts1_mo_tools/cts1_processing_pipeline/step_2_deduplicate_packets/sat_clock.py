"""When each decode was transmitted, according to the satellite's own clock.

Only two things are trusted: the onboard timestamps in log messages and live
beacons, and how far into a recording each frame starts (`time_in_file_ms`,
from the baseline decoders only, which agree with each other to under a
millisecond). Neither `received_at` nor any ground station's clock is used.

Within one recording, each of those timestamps (an "anchor") gives the offset
between the satellite's clock and time-in-file. That offset is linearly
interpolated to every other frame of the recording, or taken from the nearest
anchor for a frame with an anchor on one side only.

  - A log message starts `<time synced at>+<ms since sync>_<sync source>`,
    which sum to when it was transmitted.
  - A beacon is stamped a fixed time before it's transmitted, which is added
    on (`BEACON_STAMP_TO_TRANSMIT_MS`).
  - Historical beacons (replayed from onboard storage) are excluded by only
    accepting the exact end markers of live ones (`LIVE_END_MESSAGES`): a
    truncated one has lost its " S5", so not seeing that isn't enough.
  - Nothing is interpolated across the clock being set (or the OBC
    rebooting): a frame between two such anchors goes with the nearer one.
    See `_number_segments`.
  - Waiting to be sent can only make an anchor's offset look too low, so one
    in such a dip isn't used. A lasting change of level (audio dropped from
    the recording) isn't a dip, and is kept.

The fields are sliced straight out of the bytes, as running the packet
decoder here would be far too slow.
"""

__all__ = [
    "ANCHOR_TYPE_HEX",
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
# How long after being stamped each type of beacon starts transmitting: the
# median against heartbeat log messages (sent as soon as they're stamped) in
# the same recordings. Monthly medians were 387-408ms for basic (May-October
# 2026) and 423-442ms for extended (July-October). V5+ had too few pairs to
# measure (101, all in a congested November; they peak at 420-430ms), so is
# assumed to match extended. Re-measure if the firmware's beacon code changes.
BEACON_STAMP_TO_TRANSMIT_MS = {
    "BEACON_BASIC": 390,
    "BEACON_EXTENDED": 430,
    "BEACON_EXTENDED_V5PLUS": 430,
}
LOG_TYPE = bytes([PACKET_TYPE_MAP_INV["LOG_MESSAGE"]])
# `1780208589000+0002833553_E`, matched on hex as the rest of a log needn't be
# valid UTF-8. Any source letter but N (never synced, so not a UTC time).
LOG_PREFIX_HEX = r"^((?:3[0-9]){13})2b((?:3[0-9]){10})5f(?:4[1-9a-df]|5[0-9a])$"
LOG_PREFIX_SIZE = 13 + 1 + 10 + 2

ANCHOR_TYPE_HEX = [
    f"{PACKET_TYPE_MAP_INV[name]:02x}" for name in (*LIVE_END_MESSAGES, "LOG_MESSAGE")
]

UPTIME_OFFSET = CSP_HEADER_SIZE + struct.calcsize("<B4sBB")
EPOCH_OFFSET = CSP_HEADER_SIZE + struct.calcsize("<B4sBBII")
END_MESSAGE_OFFSET = CSP_HEADER_SIZE + BEACON_FIXED_SIZE + FRIENDLY_MESSAGE_SIZE

# The two clocks are read a millisecond or so apart, so can differ by that much.
CLOCK_STEP_TOLERANCE_MS = 5
# An abrupt change this big is a step even if no two anchors of one kind show
# it (e.g. the only beacon is before the clock was set, the only log after).
OFFSET_STEP_MS = 5_000
# Above the normal jitter.
LATE_ANCHOR_TOLERANCE_MS = 100


_ANCHOR_KEYS = ("observation_id", "_content_id", "_trust_rank", "_t")


def _live_beacons(decodes: pl.LazyFrame) -> pl.LazyFrame:
    """`_clock` (when it was transmitted) for each live beacon decode.
    `_sync` only changes when the clock is set or the OBC reboots."""
    data = pl.col("_data")
    packet_type = data.bin.slice(CSP_HEADER_SIZE, 1)
    end_message = data.bin.slice(END_MESSAGE_OFFSET, END_MESSAGE_SIZE)
    is_type = {
        name: packet_type == pl.lit(bytes([PACKET_TYPE_MAP_INV[name]]))
        for name in LIVE_END_MESSAGES
    }
    is_live = pl.any_horizontal(
        is_type[name] & end_message.is_in(markers)
        for name, markers in LIVE_END_MESSAGES.items()
    )
    lead = pl.coalesce(
        pl.when(is_type[name]).then(pl.lit(ms, pl.Int64))
        for name, ms in BEACON_STAMP_TO_TRANSMIT_MS.items()
    )
    uptime = (
        data.bin.slice(UPTIME_OFFSET, 4)
        .bin.reinterpret(dtype=pl.UInt32, endianness="little")
        .cast(pl.Int64)
    )
    return (
        decodes.filter(is_live & pl.col("csp_crc_valid"))
        .with_columns(
            _stamp=data.bin.slice(EPOCH_OFFSET, 8)
            .bin.reinterpret(dtype=pl.UInt64, endianness="little")
            .cast(pl.Int64, strict=False),  # null (so dropped below) if absurd
        )
        .filter(pl.col("_stamp").is_between(1, MAX_VALID_EPOCH_MS))
        .select(
            *_ANCHOR_KEYS,
            _is_log=pl.lit(value=False),
            _clock=pl.col("_stamp") + lead,
            _sync=pl.col("_stamp") - uptime,
        )
    )


def _timestamped_logs(decodes: pl.LazyFrame) -> pl.LazyFrame:
    """`_clock` (when it was transmitted) for each log message decode with a
    timestamp prefix. `_sync` is when the clock was last set."""
    data = pl.col("_data")
    prefix = data.bin.slice(CSP_HEADER_SIZE + 1, LOG_PREFIX_SIZE).bin.encode("hex")

    def number(group: int) -> pl.Expr:
        digits = prefix.str.extract(LOG_PREFIX_HEX, group).str.decode("hex")
        return digits.cast(pl.String).cast(pl.Int64)

    is_log = data.bin.slice(CSP_HEADER_SIZE, 1) == pl.lit(LOG_TYPE)
    return (
        decodes.filter(is_log & pl.col("csp_crc_valid"))
        .with_columns(_sync=number(1), _since_sync=number(2))
        .select(
            *_ANCHOR_KEYS,
            _is_log=pl.lit(value=True),
            _clock=pl.col("_sync") + pl.col("_since_sync"),
            _sync="_sync",
        )
        .filter(
            (pl.col("_sync") > 0) & pl.col("_clock").is_between(1, MAX_VALID_EPOCH_MS)
        )
    )


def _number_segments(anchors: pl.LazyFrame) -> pl.LazyFrame:
    """Add `_segment`, which differs either side of a clock step.

    Beacons and logs each show a step exactly, as a change in their own
    `_sync`, but the two aren't comparable. So an anchor of the other kind
    that's between the two showing a step joins whichever its offset is
    closer to.
    """
    recording = "observation_id"
    offset = pl.col("_offset")

    def steps_seen_by(kind: pl.Expr) -> pl.Expr:
        sync = pl.when(kind).then("_sync")
        last_sync = sync.forward_fill().shift(1).over(recording)
        stepped = (
            ((sync - last_sync).abs() > CLOCK_STEP_TOLERANCE_MS)
            .fill_null(value=False)
            .cum_sum()
            .over(recording)
        )
        count = pl.when(kind).then(stepped)
        own_offset = pl.when(kind).then(offset)
        to_previous = (offset - own_offset.forward_fill().over(recording)).abs()
        to_next = (offset - own_offset.backward_fill().over(recording)).abs()
        return (
            pl.when(to_next.is_null() | (to_previous <= to_next))
            .then(count.forward_fill().over(recording))
            .otherwise(count.backward_fill().over(recording))
            .fill_null(0)
        )

    jumps = (
        (offset.diff().abs() > OFFSET_STEP_MS)
        .fill_null(value=False)
        .cum_sum()
        .over(recording)
    )
    return anchors.with_columns(
        _segment=pl.struct(
            logs=steps_seen_by(pl.col("_is_log")),
            beacons=steps_seen_by(~pl.col("_is_log")),
            jumps=jumps,
        ).rank("dense")
    )


def _clock_anchors(decodes: pl.LazyFrame) -> pl.LazyFrame:
    """The timestamps worth interpolating between, in time order: `_offset`
    is satellite clock minus time-in-file."""
    anchors = (
        pl.concat([_live_beacons(decodes), _timestamped_logs(decodes)])
        # Both baseline decoders usually report the same packet.
        .sort("observation_id", "_content_id", "_trust_rank", "_t")
        .unique(["observation_id", "_content_id"], keep="first", maintain_order=True)
        .sort("observation_id", "_t", "_content_id")
        .with_columns(_offset=pl.col("_clock") - pl.col("_t"))
    )
    segment = ["observation_id", "_segment"]
    offset = pl.col("_offset")
    highest_before = offset.cum_max().over(segment)
    highest_after = offset.cum_max(reverse=True).over(segment)
    in_dip = (
        pl.min_horizontal(highest_before, highest_after) - offset
        > LATE_ANCHOR_TOLERANCE_MS
    )
    # A dip needs an anchor on each side, so the first and last are instead
    # compared with their one neighbour. Only at the ends of the recording:
    # next to a clock step, they're what says where the step was.
    previous = offset.shift(1).over(segment)
    following = offset.shift(-1).over(segment)
    late_at_either_end = (
        (pl.col("_is_first") & (following - offset > LATE_ANCHOR_TOLERANCE_MS))
        | (pl.col("_is_last") & (previous - offset > LATE_ANCHOR_TOLERANCE_MS))
    ).fill_null(value=False)
    return (
        _number_segments(anchors)
        .with_columns(
            _is_first=offset.shift(1).over("observation_id").is_null(),
            _is_last=offset.shift(-1).over("observation_id").is_null(),
        )
        .filter(~in_dip)
        .filter(~late_at_either_end)
        .select("observation_id", "_t", "_offset", "_segment")
    )


def estimate_transmit_times(rows: pl.LazyFrame) -> pl.LazyFrame:
    """`(_row_id, observation_id, _transmitted_at)` for every baseline decode
    in `rows` (as prepared by step 2) that has a `time_in_file_ms`. The time
    is null where its recording has no usable anchor.
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
    anchors = _clock_anchors(decodes)
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
            # Whole milliseconds, like the onboard timestamps.
            _transmitted_at=(t + offset)
            .round()
            .cast(pl.Int64)
            .cast(pl.Datetime("ms", "UTC"))
            .dt.cast_time_unit("us"),
        )
    )
