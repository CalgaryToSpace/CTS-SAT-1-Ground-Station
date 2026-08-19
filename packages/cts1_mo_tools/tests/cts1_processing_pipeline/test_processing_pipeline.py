import pytest
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate.decode_gr_satellites import (  # noqa: E501
    parse_hexdump_stdout,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate.decode_gr_satellites_kiss import (  # noqa: E501
    parse_kiss_file,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate.decode_satnogs_data_demod import (  # noqa: E501
    parse_demod_filename_time,
)
from cts1_mo_tools.cts1_processing_pipeline.step_1_download_and_demodulate.decode_sso_rx_replay import (  # noqa: E501
    parse_forensics_line,
)

FEND, FESC, TFEND, TFESC = 0xC0, 0xDB, 0xDC, 0xDD


def _kiss_escape(payload: bytes) -> bytes:
    out = bytearray()
    for b in payload:
        if b == FEND:
            out += bytes([FESC, TFEND])
        elif b == FESC:
            out += bytes([FESC, TFESC])
        else:
            out.append(b)
    return bytes(out)


def _kiss_frame(control: int, payload: bytes) -> bytes:
    return bytes([FEND]) + _kiss_escape(bytes([control]) + payload) + bytes([FEND])


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
        "time_in_file_ms": 45802.229,
        "rssi": -2.1,
        "rs_corrected_error_count": None,
        "rs_correctable": False,
        "data_hex": "c2228a001091",
        "data_length_bytes": 6,
    }


def test_parse_forensics_line_frame_rs_corrected() -> None:
    line = (
        '{"filename":"sample.ogg","time_in_file_ms":45802.229,'
        '"rssi":-2.1,"rs":3,"data_base64":"wiKKABCR"}'
    )
    row = parse_forensics_line(line)
    assert row == {
        "time_in_file_ms": 45802.229,
        "rssi": -2.1,
        "rs_corrected_error_count": 3,
        "rs_correctable": True,
        "data_hex": "c2228a001091",
        "data_length_bytes": 6,
    }


def test_parse_forensics_line_no_decode() -> None:
    # SSO quirk: reports just the filename when the file decoded cleanly but
    # had no frames. Nothing to land in raw_packets, so it's skipped.
    assert parse_forensics_line('{"filename":"sample.ogg"}') is None


def test_parse_forensics_line_error() -> None:
    # An {"filename", "error"} line carries no frame fields, so indexing them
    # raises rather than silently fabricating a row.
    with pytest.raises(KeyError):
        parse_forensics_line(
            '{"filename":"bad.ogg","error":"could not decode audio file via libsndfile"}'  # noqa: E501
        )


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
"""


def test_parse_hexdump_stdout_extracts_pdu() -> None:
    rows = parse_hexdump_stdout(_HEXDUMP_STDOUT)
    assert len(rows) == 1
    row = rows[0]
    # Excluded - assert row["transmitter_name"] == "9k6 FSK downlink"
    assert row["data_length_bytes"] == 208
    assert len(row["data_hex"]) == 208 * 2
    assert row["data_hex"].startswith("c2a28a0010d06c0e000822085e07fa07")


def test_parse_hexdump_stdout_no_pdus() -> None:
    assert parse_hexdump_stdout("nothing decoded here\n") == []


# ---------------------------------------------------------------------------
# gr_satellites --kiss_out binary parsing
# ---------------------------------------------------------------------------


def test_parse_kiss_file_timestamp_then_data() -> None:
    launch_ms = 1_700_000_000_000
    ts_ms = launch_ms + 12_345
    data = _kiss_frame(0x09, ts_ms.to_bytes(8, "big")) + _kiss_frame(
        0x00, b"\xaa\xbb\xcc"
    )

    rows = parse_kiss_file(data, launch_time_ms=launch_ms)

    assert rows == [
        {
            "data_length_bytes": 3,
            "data_hex": "aabbcc",
            "time_in_file_ms": 12_345,
        }
    ]


def test_parse_kiss_file_unescapes_control_bytes() -> None:
    # Payload deliberately contains FEND (0xc0) and FESC (0xdb), which must
    # come back through KISS-escaped and be correctly restored.
    launch_ms = 1_700_000_000_000
    payload = bytes([0xC0, 0xDB, 0x01])
    data = _kiss_frame(0x00, payload)

    rows = parse_kiss_file(data, launch_time_ms=launch_ms)

    assert len(rows) == 1
    assert rows[0]["data_hex"] == payload.hex()
    assert rows[0]["time_in_file_ms"] is None  # no timestamp frame preceded it


def test_parse_kiss_file_multiple_pdus_each_use_own_timestamp() -> None:
    launch_ms = 1_700_000_000_000
    data = (
        _kiss_frame(0x09, (launch_ms + 100).to_bytes(8, "big"))
        + _kiss_frame(0x00, b"\x01")
        + _kiss_frame(0x09, (launch_ms + 200).to_bytes(8, "big"))
        + _kiss_frame(0x00, b"\x02")
    )

    rows = parse_kiss_file(data, launch_time_ms=launch_ms)

    assert [r["time_in_file_ms"] for r in rows] == [100, 200]
    assert [r["data_hex"] for r in rows] == ["01", "02"]


def test_parse_kiss_file_empty() -> None:
    assert parse_kiss_file(b"", launch_time_ms=0) == []


# ---------------------------------------------------------------------------
# satnogs_data_demod filename timestamp parsing
# ---------------------------------------------------------------------------

_DEMOD_URL_BASE = (
    "https://s3.eu-central-1.wasabisys.com/satnogs-network/data_obs/"
    "2026/8/12/19/14759295/data_14759295_2026-08-12T19-36-50"
)


def test_parse_demod_filename_time_no_suffix() -> None:
    dt = parse_demod_filename_time(_DEMOD_URL_BASE)
    assert dt is not None
    assert dt.isoformat() == "2026-08-12T19:36:50+00:00"


def test_parse_demod_filename_time_with_seq_suffix() -> None:
    dt = parse_demod_filename_time(_DEMOD_URL_BASE + "_1")
    assert dt is not None
    assert dt.isoformat() == "2026-08-12T19:36:50.100000+00:00"


def test_parse_demod_filename_time_with_g_suffix() -> None:
    dt = parse_demod_filename_time(_DEMOD_URL_BASE + "_g1")
    assert dt is not None
    assert dt.isoformat() == "2026-08-12T19:36:50.150000+00:00"


def test_parse_demod_filename_time_unrecognized_filename() -> None:
    assert parse_demod_filename_time("https://example.com/not-a-demod-file") is None
