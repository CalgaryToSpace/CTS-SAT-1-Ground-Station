"""Test CRC-32C calculator(s)."""

import polars as pl
import polars_hash
from cts1_mo_tools.cts1_decode_satnogs_packets import crc32c


def test_crc32c_calc_python() -> None:
    # 0xE3069283 is the standard check value for the CRC-32C variant.
    assert crc32c(b"123456789") == 0xE3069283
    assert crc32c(b"") == 0x00


def test_crc32c_polars_hash() -> None:
    df = pl.DataFrame(
        {"data": ["123456789", ""]},
        infer_schema_length=None,
    )

    df = df.with_columns(
        crc32c_output=(
            polars_hash.col("data")
            .nchash.crc32c(return_binary=True, byte_order="big")
            .bin.encode("hex")
        )
    )

    assert df["crc32c_output"].to_list() == ["e3069283", "00000000"]
