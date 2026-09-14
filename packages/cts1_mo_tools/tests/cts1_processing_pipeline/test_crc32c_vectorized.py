import random

import polars as pl
from cts1_mo_tools.cts1_decode_satnogs_packets import crc32c
from cts1_mo_tools.cts1_processing_pipeline.step_2_deduplicate_packets import (
    crc32c_vectorized as target,
)


def _golden_crc32c_hex(data_hex: str) -> str:
    """The golden, per-row reference: bit-by-bit `crc32c()` + `.hex()`."""
    return crc32c(bytes.fromhex(data_hex)).to_bytes(4, "big").hex()


def _assert_matches_golden(hex_values: list[str]) -> None:
    df = pl.DataFrame({"data_hex": hex_values})
    actual = target.crc32c_hex_series(df, "data_hex").to_list()
    expected = [_golden_crc32c_hex(h) for h in hex_values]
    assert actual == expected


def test_empty_payload() -> None:
    _assert_matches_golden([""])


def test_single_byte_payload() -> None:
    _assert_matches_golden(["ab"])


def test_known_vector() -> None:
    # CRC-32C("123456789") == 0xE3069283, the standard check value for this variant.
    _assert_matches_golden(["313233343536373839"])


def test_varying_lengths_in_one_column() -> None:
    """Rows of very different lengths side by side -- exercises the
    per-row `i < n_bytes` masking that lets shorter rows stop updating
    while longer rows in the same batch keep going.
    """
    _assert_matches_golden(["", "aa", "aabbcc", "c2a28a00" + "11" * 40])


def test_random_payloads_match_golden_implementation() -> None:
    rng = random.Random(0)  # noqa: S311 -- not cryptographic, just test data
    hex_values = [
        bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 64))).hex()
        for _ in range(500)
    ]
    _assert_matches_golden(hex_values)


def test_all_same_length_payloads_match_golden_implementation() -> None:
    """A batch where every row has the same length -- no masking kicks in
    at all, so this exercises the plain byte-position loop on its own.
    """
    rng = random.Random(1)  # noqa: S311 -- not cryptographic, just test data
    hex_values = [
        bytes(rng.randint(0, 255) for _ in range(20)).hex() for _ in range(200)
    ]
    _assert_matches_golden(hex_values)
