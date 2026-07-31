from unittest.mock import patch, Mock
from cts1_mo_tools.cts1_tle_formatter import fetch_tle


# ---------------------------------------------------------------------
# fetch_tle() tests
# ---------------------------------------------------------------------
''' ("No GP data found" in tle_data) test'''
@patch("cts1_mo_tools.cts1_tle_formatter.requests.get")
def test_fetch_tle_no_data(mock_get: Mock):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "No GP data found"

    mock_get.return_value = mock_response

    assert fetch_tle() is None
    
'''Empty tle_data test'''
@patch("cts1_mo_tools.cts1_tle_formatter.requests.get")
def test_fetch_tle_empty_response(mock_get: Mock):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = ""

    mock_get.return_value = mock_response

    assert fetch_tle() is None

'''Connection Error test'''
@patch("cts1_mo_tools.cts1_tle_formatter.requests.get")
def test_fetch_tle_http_error(mock_get: Mock):
    mock_response = Mock()
    mock_response.status_code = 404

    mock_get.return_value = mock_response

    assert fetch_tle() is None