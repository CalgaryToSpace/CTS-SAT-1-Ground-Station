from cts1_mo_tools.cts1_processing_pipeline.decode_gr_satellites import (
    parse_hexdump_stdout,
)
from cts1_mo_tools.cts1_processing_pipeline.decode_sso_rx_replay import (
    parse_forensics_line,
)

# ---------------------------------------------------------------------------
# sso_rx_replay --forensics-report line parsing
# ---------------------------------------------------------------------------


def test_parse_forensics_line_frame() -> None:
    line = (
        '{"filename":"sample.ogg","time_in_file_ms":45802.229,'
        '"rssi":-2.1,"rs":-2,"data_base64":"wiKKABCR"}'
    )
    row = parse_forensics_line(line)
    assert row == {
        "sso_filename": "sample.ogg",
        "sso_time_in_file_ms": 45802.229,
        "sso_rssi": -2.1,
        "sso_rs": -2,
        "sso_data_base64": "wiKKABCR",
        "sso_error": None,
    }


def test_parse_forensics_line_no_decode() -> None:
    row = parse_forensics_line('{"filename":"sample.ogg"}')
    assert row is not None
    assert row["sso_filename"] == "sample.ogg"
    assert row["sso_data_base64"] is None
    assert row["sso_error"] is None


def test_parse_forensics_line_error() -> None:
    row = parse_forensics_line(
        '{"filename":"bad.ogg","error":"could not decode audio file via libsndfile"}'
    )
    assert row is not None
    assert row["sso_error"] == "could not decode audio file via libsndfile"
    assert row["sso_data_base64"] is None


def test_parse_forensics_line_blank_and_malformed() -> None:
    assert parse_forensics_line("") is None
    assert parse_forensics_line("   ") is None
    assert parse_forensics_line("not json") is None


# ---------------------------------------------------------------------------
# gr_satellites --hexdump stdout parsing
# ---------------------------------------------------------------------------

_HEXDUMP_STDOUT = """\
***** VERBOSE PDU DEBUG PRINT ******
((transmitter . 9k6 FSK downlink))
pdu length =        208 bytes
pdu vector contents =
0000: c2 a2 8a 00 10 d0 6c 0e 00 08 22 08 5e 07 fa 07
0010: b8 08 08 07 8e 07 dd 07 cf 07 fc 07 ea 07 d6 08
0020: 5c 08 7c 07 fc 07 c5 07 ed 07 c9 07 fe 07 af 07
0030: c0 43 cb 0c ff ff 0c 18 24 08 4c 11 03 ff 2f ff
0040: 1c 0f 50 30 70 01 04 07 f0 07 ed 07 e5 07 d2 07
0050: b6 07 5b 07 e2 07 6d 07 d0 07 de 07 67 07 77 07
0060: 40 07 57 07 de 07 bc 07 bf 07 aa 07 d0 07 bc 07
0070: fb 07 a8 07 dd 07 74 07 d1 07 d5 07 84 07 cb 07
0080: f7 08 1e 07 cc 07 a7 07 eb 07 f0 07 f9 07 a9 07
0090: 56 07 73 07 c7 07 9b 07 df 07 a2 07 88 08 15 07
00a0: b4 08 20 08 08 07 ab 07 9b 08 07 07 69 07 85 07
00b0: b4 08 05 08 18 07 a7 08 05 08 03 07 c5 07 b3 07
00c0: d5 07 b4 07 b3 07 54 07 a6 04 22 0c 9b d4 db 07
************************************
"""  # noqa: W291


def test_parse_hexdump_stdout_extracts_pdu() -> None:
    rows = parse_hexdump_stdout(_HEXDUMP_STDOUT)
    assert len(rows) == 1
    row = rows[0]
    assert row["gr_transmitter"] == "9k6 FSK downlink"
    assert row["gr_pdu_length_bytes"] == 208  # noqa: PLR2004
    assert len(row["gr_pdu_hex"]) == 208 * 2
    assert row["gr_pdu_hex"].startswith("c2a28a0010d06c0e000822085e07fa07")


def test_parse_hexdump_stdout_no_pdus() -> None:
    assert parse_hexdump_stdout("nothing decoded here\n") == []
