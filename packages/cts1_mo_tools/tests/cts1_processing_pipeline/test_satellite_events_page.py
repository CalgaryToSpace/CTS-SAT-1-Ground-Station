from cts1_mo_tools.cts1_processing_pipeline.web_ui.satellite_events_page import (
    format_chance,
)


def test_format_chance() -> None:
    assert format_chance(0.0755301) == "7.553%"
    assert format_chance(0.00001) == "0.001%"
    assert format_chance(1.3193e-22) == "10⁻²²"
    assert format_chance(3e-6) == "10⁻⁶"
    assert format_chance(0.0) == "0.000%"
