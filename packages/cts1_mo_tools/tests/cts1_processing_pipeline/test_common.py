from cts1_mo_tools.cts1_processing_pipeline.common import duration_str


def test_duration_str() -> None:
    assert duration_str(42) == "42s"
    assert duration_str(16 * 60 + 50) == "16m 50s"
    assert duration_str(3600 + 23 * 60 + 5) == "1h 23m"
    assert duration_str(2 * 86400 + 5 * 3600) == "2d 5h"
