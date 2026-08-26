"""Vectorized (Polars-native) CRC-32C over a hex-string column.

`cts1_decode_satnogs_packets.crc32c()` is a golden, bit-by-bit reference
implementation -- clear, but a per-row Python call away from being usable at
the row counts step 2 deals with. `crc32c_hex_series()` here computes the
same CRC-32C (Castagnoli), table-driven, entirely as chained Polars
expressions: the only Python-level loop is over byte *position* (bounded by
the longest payload in the column, not by row count), with each iteration a
single vectorized pass over the whole column. See
`tests/cts1_processing_pipeline/test_crc32c_vectorized.py` for a
row-for-row comparison against the golden implementation.
"""

from __future__ import annotations

__all__ = ["crc32c_hex_series"]

import polars as pl

_CRC32C_POLY = 0x82F63B78  # reflected form of 0x1EDC6F41 (Castagnoli), per libcsp


def _make_crc32c_table() -> dict[int, int]:
    table: dict[int, int] = {}
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ (_CRC32C_POLY if crc & 1 else 0)
        table[i] = crc & 0xFFFFFFFF
    return table


_CRC32C_TABLE = _make_crc32c_table()
_HEX_BYTE_TABLE = {i: f"{i:02x}" for i in range(256)}


def crc32c_hex_series(df: pl.DataFrame, hex_col: str) -> pl.Series:
    """CRC-32C (Castagnoli) over a hex-string column's decoded bytes, as an
    8-hex-char big-endian string -- entirely as vectorized Polars
    expressions (table-driven, byte-position by byte-position across the
    whole column at once), so it stays fast at millions of rows instead of
    paying a per-row Python call.

    Matches `cts1_decode_satnogs_packets.crc32c()` bit-for-bit -- see this
    module's docstring for the reference implementation this is validated
    against.
    """
    n_bytes = pl.col(hex_col).str.len_chars() // 2
    max_len = df.select(n_bytes.max()).item() or 0

    work = df.select(hex_col).with_columns(
        _crc32c=pl.lit(0xFFFFFFFF, dtype=pl.UInt32), _crc32c_n_bytes=n_bytes
    )
    for i in range(max_len):
        byte_val = (
            pl.col(hex_col)
            .str.slice(i * 2, 2)
            .str.to_integer(base=16, dtype=pl.UInt32, strict=False)
        )
        table_idx = (pl.col("_crc32c") ^ byte_val) & 0xFF
        table_val = table_idx.replace_strict(_CRC32C_TABLE, return_dtype=pl.UInt32)
        updated = (pl.col("_crc32c") // 256) ^ table_val
        work = work.with_columns(
            _crc32c=pl.when(i < pl.col("_crc32c_n_bytes"))
            .then(updated)
            .otherwise(pl.col("_crc32c"))
        )

    work = work.with_columns(_crc32c=pl.col("_crc32c") ^ 0xFFFFFFFF)
    crc = pl.col("_crc32c")
    hex_str = work.select(
        pl.concat_str(
            [
                ((crc // 16_777_216) & 0xFF).replace_strict(
                    _HEX_BYTE_TABLE, return_dtype=pl.String
                ),
                ((crc // 65_536) & 0xFF).replace_strict(
                    _HEX_BYTE_TABLE, return_dtype=pl.String
                ),
                ((crc // 256) & 0xFF).replace_strict(
                    _HEX_BYTE_TABLE, return_dtype=pl.String
                ),
                (crc & 0xFF).replace_strict(_HEX_BYTE_TABLE, return_dtype=pl.String),
            ]
        ).alias("crc_hex")
    )
    return hex_str.to_series()
