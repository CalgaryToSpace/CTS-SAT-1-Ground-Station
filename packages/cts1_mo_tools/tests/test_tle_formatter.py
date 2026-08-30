from unittest.mock import Mock, patch

import pytest
from cts1_mo_tools.cts1_tle_formatter import (
    TLE,
    convert_to_telecommand,
    extract_orbit_parameters,
    fetch_tle,
    parse_tle,
)


def test_full_decode_and_convert_case_1() -> None:
    """Full test from TLE to telecommand.

    This is the main proof that this tool works.
    """
    input_tle = """
FRONTIERSAT
1 69015U 26100AM  26183.24927377  .00010390  00000+0  45830-3 0  9996
2 69015  97.3985  80.8620 0007979   5.6312 354.5013 15.21937754  9113
""".strip()

    expected_output = """
CTS1+adcs_set_sgp4_orbit_params(97.3985,0.0007979,80.8620,5.6312,0.0004583,15.21937754,354.5013,26183.24927377)!
""".strip()

    assert convert_to_telecommand(input_tle) == expected_output


# ---------------------------------------------------------------------
# fetch_tle() tests
# ---------------------------------------------------------------------
"""Tests that "No GP data found" in tle_data is handled"""


@patch("cts1_mo_tools.cts1_tle_formatter.requests.get")
def test_fetch_tle_no_data(mock_get: Mock) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "No GP data found"

    mock_get.return_value = mock_response

    assert fetch_tle() is None


"""Tests that empty tle_data is handled"""


@patch("cts1_mo_tools.cts1_tle_formatter.requests.get")
def test_fetch_tle_empty_response(mock_get: Mock) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = ""

    mock_get.return_value = mock_response

    assert fetch_tle() is None


"""Tests Connection Error"""


@patch("cts1_mo_tools.cts1_tle_formatter.requests.get")
def test_fetch_tle_http_error(mock_get: Mock) -> None:
    mock_response = Mock()
    mock_response.status_code = 404

    mock_get.return_value = mock_response

    assert fetch_tle() is None


# ---------------------------------------------------------------------
# parse_tle(text: str) tests
# ---------------------------------------------------------------------
"""Test that CelesTrak returns 3 lines"""


def test_parse_tle_raises_error_if_not_three_lines() -> None:
    invalid_tle = "FRONTIERSAT\n1 69015U 24001A"

    with pytest.raises(ValueError, match="Expected a 3-line TLE"):
        parse_tle(invalid_tle)


# ---------------------------------------------------------------------
# extract_orbit_parameters(tle: TLE) tests
# ---------------------------------------------------------------------
"""Test that eccentricity is converted to the correct format"""


def test_extract_orbit_parameters_converts_eccentricity() -> None:
    tle = TLE(
        name="TEST",
        line1="1 69015U 24001A 25035.12345678 .00000000 00000-0 36581-5 0 9991",
        line2="2 69015 97.5000 120.0000 0001000 180.0000 180.0000 15.00000000",
    )

    result = extract_orbit_parameters(tle)

    assert result.eccentricity == "0.0001000"


"""Test that drag term is converted to the correct format"""


@pytest.mark.parametrize(
    ("drag_value", "expected"),
    [
        ("36581-5", "0.0000036581"),
        ("36581-6", "3.6581E-7"),
        ("36581-1", "0.036581"),
    ],
)
def test_extract_orbit_parameters_drag_term_exponents(
    drag_value: str,
    expected: str,
) -> None:
    tle = TLE(
        name="TEST",
        line1=f"1 69015U 24001A 25035.12345678 .00000000 00000-0 {drag_value} 0 9991",
        line2="2 69015 97.5000 120.0000 0001000 180.0000 180.0000 15.00000000",
    )

    result = extract_orbit_parameters(tle)

    assert result.drag_term == expected
