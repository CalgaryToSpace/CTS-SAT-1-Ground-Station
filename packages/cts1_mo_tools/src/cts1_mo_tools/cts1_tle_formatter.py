"""
Script to fetch a TLE from CelesTrak (or read one from a file), and format it
into the CTS1+adcs_set_sgp4_orbit_params telecommand.
"""

import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import requests
import tyro
from loguru import logger

DEFAULT_NORAD_ID = 69015
"""NORAD ID of CTS-SAT-1 (FrontierSat)."""


@dataclass
class Tle:
    name: str
    line1: str
    line2: str


@dataclass
class TelecommandParams:
    inclination: str
    eccentricity: str
    right_ascension: str
    argument_of_perigee: str
    drag_term: str
    mean_motion: str
    mean_anomaly: str
    epoch: str

    def to_args_list(self) -> list[str]:
        """Return list of telecommand arguments in order of the telecommand."""
        return [
            self.inclination,
            self.eccentricity,
            self.right_ascension,
            self.argument_of_perigee,
            self.drag_term,
            self.mean_motion,
            self.mean_anomaly,
            self.epoch,
        ]


def fetch_tle(norad_id: int) -> str | None:
    """Fetch the latest TLE for the given NORAD ID from CelesTrak."""

    url = "https://celestrak.org/NORAD/elements/gp.php"
    params = {
        "CATNR": str(norad_id),
        "FORMAT": "tle",
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        success_status_code = 200

        if response.status_code == success_status_code:
            tle_data = response.text.strip()

            if "No GP data found" in tle_data or not tle_data:
                logger.error(f"Error: No orbital data found for NORAD ID {norad_id}.")
                return None

            return tle_data

        logger.error(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
        return None  # noqa: TRY300

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while connecting to CelesTrak: {e}")
        return None


def convert_3le_to_telecommand(tle_text: str) -> str:
    """Convert raw 3-line TLE text into the CTS telecommand."""

    tle = parse_tle(tle_text)
    params = extract_orbit_parameters(tle)
    return format_telecommand(params)


def parse_tle(text: str) -> Tle:
    """Parse raw TLE text into a TLE object."""

    lines = text.strip().splitlines()

    expected_num_lines = 3

    if len(lines) != expected_num_lines:
        error_message = "Expected a 3-line TLE"
        raise ValueError(error_message)

    return Tle(
        name=lines[0],
        line1=lines[1],
        line2=lines[2],
    )


def extract_orbit_parameters(tle: Tle) -> TelecommandParams:
    """Extract the orbital parameters required by the CTS telecommand."""

    line1 = tle.line1.split()
    line2 = tle.line2.split()

    # Convert TLE drag term (e.g. 36581-3 -> 0.00036581)
    initial_drag_term = Decimal("0." + line1[6][:5]) * (
        Decimal(10) ** int(line1[6][5:])
    )

    return TelecommandParams(
        inclination=line2[2],
        eccentricity="0." + line2[4],
        right_ascension=line2[3],
        argument_of_perigee=line2[5],
        drag_term=str(initial_drag_term.normalize()),
        mean_motion=line2[7],
        mean_anomaly=line2[6],
        epoch=line1[3],
    )


def format_telecommand(params: TelecommandParams) -> str:
    """Format the final CTS telecommand."""
    args_str = ",".join(params.to_args_list())

    return f"CTS1+adcs_set_sgp4_orbit_params({args_str})"


def run(
    *,
    norad_id: int | None = None,
    tle_file: Path | None = None,
) -> None:
    """Format a TLE into the CTS1+adcs_set_sgp4_orbit_params telecommand.

    At most one of --norad-id or --tle-file may be given. If neither is given,
    the TLE for CTS-SAT-1 (NORAD ID 69015) is fetched from CelesTrak.

    Args:
        norad_id: NORAD ID to fetch the TLE for from CelesTrak. Defaults to 69015.
        tle_file: Path to a file containing a 3-line TLE (name line + 2 lines).
    """
    if norad_id is not None and tle_file is not None:
        logger.error("Specify only one of --norad-id or --tle-file, not both.")
        sys.exit(2)

    if tle_file is None:
        if norad_id is None:
            norad_id = DEFAULT_NORAD_ID
        tle_text = fetch_tle(norad_id)
        if tle_text is None:
            sys.exit(1)
        logger.info(f"\n--- TLE RETURNED FROM CELESTRAK ---\n{tle_text}\n")
    else:
        tle_text = tle_file.read_text().strip()
        logger.info(f"\n--- TLE READ FROM {tle_file} ---\n{tle_text}\n")

    telecommand = convert_3le_to_telecommand(tle_text)
    logger.success(f"\n--- FINAL FORMATTED TELECOMMAND ---\n{telecommand}\n")


def main() -> None:
    """Entry point."""
    tyro.cli(run)


if __name__ == "__main__":
    main()
