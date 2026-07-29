from pathlib import Path

from cts1_mo_tools.cts1_spreadsheet_to_agenda import spreadsheet_file_to_agenda_file

SAMPLE_CSV = (
    Path(__file__).parent
    / "cts1_spreadsheet_to_agenda_templates"
    / "agenda_spreadsheet_template_example.csv"
)
SAMPLE_EXCEL_FILE = (
    Path(__file__).parent
    / "cts1_spreadsheet_to_agenda_templates"
    / "agenda_spreadsheet_template_example.xlsx"
)


def test_agenda_file_from_csv(tmp_path: Path) -> None:
    assert SAMPLE_CSV.is_file()

    output_file = tmp_path / "output.txt"

    spreadsheet_file_to_agenda_file(
        input_file=SAMPLE_CSV,
        output_file=output_file,
        seed=0,
    )

    assert output_file.stat().st_size > 1000


def test_agenda_file_from_xlsx(tmp_path: Path) -> None:
    assert SAMPLE_CSV.is_file()

    output_file = tmp_path / "output.txt"

    spreadsheet_file_to_agenda_file(
        input_file=SAMPLE_CSV,
        output_file=output_file,
        seed=0,
    )

    assert output_file.stat().st_size > 1000
