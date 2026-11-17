from cts1_mo_tools.cts1_processing_pipeline import daemon
from cts1_mo_tools.cts1_processing_pipeline.web_ui.export_page import (
    PIPELINE_FLOWCHART,
)
from cts1_mo_tools.cts1_processing_pipeline.web_ui.export_tables import TABLE_SPECS


def test_flowchart_has_every_table() -> None:
    # The flowchart is kept by hand -- see `PIPELINE_FLOWCHART`.
    missing = [s.key for s in TABLE_SPECS if f'[("{s.key}")]' not in PIPELINE_FLOWCHART]
    assert not missing


def test_flowchart_has_every_step() -> None:
    missing = [
        n for n in daemon.STEP_NAMES if f'"Step {n}\\n' not in PIPELINE_FLOWCHART
    ]
    assert not missing
