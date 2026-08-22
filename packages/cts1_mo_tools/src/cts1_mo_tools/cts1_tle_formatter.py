from dataclasses import dataclass
from decimal import Decimal

import requests

"""
Script to fetch FrontierSat TLE from CelesTrak, and format it into the
CTS1+adcs_set_sgp4_orbit_params telecommand.

To run from this file:
1) Click the play button on the top right to run the python file

To run from CLI:
1) In the terminal, navigate to the
.../CTS-SAT-1-Ground-Station/packages/cts1_mo_tools/src/cts1_mo_tools folder
2) Run 'python cts1_tle_formatter.py'

Debugging:
If unable to run script due to 'missing requests module', then run 'uv sync'
"""


@dataclass
class TLE:
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


def fetch_tle() -> str | None:
    """Fetch the latest FrontierSat TLE from CelesTrak."""

    url = "https://celestrak.org/NORAD/elements/gp.php"
    params = {
        "CATNR": "69015",
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
                print("Error: No orbital data found for FrontierSat.")  # noqa: T201
                return None

            return tle_data

        print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")  # noqa: T201
        return None  # noqa: TRY300

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while connecting to CelesTrak: {e}")  # noqa: T201
        return None


def convert_to_telecommand(tle_text: str) -> str:
    """Convert raw TLE text into the CTS telecommand."""

    tle = parse_tle(tle_text)
    params = extract_orbit_parameters(tle)
    return format_telecommand(params)


def parse_tle(text: str) -> TLE:
    """Parse raw TLE text into a TLE object."""

    lines = text.strip().splitlines()

    expected_num_lines = 3

    if len(lines) != expected_num_lines:
        error_message = "Expected a 3-line TLE"
        raise ValueError(error_message)

    return TLE(
        name=lines[0],
        line1=lines[1],
        line2=lines[2],
    )


def extract_orbit_parameters(tle: TLE) -> TelecommandParams:
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

    return (
        "CTS1+adcs_set_sgp4_orbit_params("
        f"{params.inclination},"
        f"{params.eccentricity},"
        f"{params.right_ascension},"
        f"{params.argument_of_perigee},"
        f"{params.drag_term},"
        f"{params.mean_motion},"
        f"{params.mean_anomaly},"
        f"{params.epoch}"
        ")!"
    )


def main() -> None:
    tle_text = fetch_tle()

    if tle_text is None:
        return

    print("\n--- TLE RETURNED FROM CELESTRAK ---")  # noqa: T201
    print(tle_text)  # noqa: T201

    telecommand = convert_to_telecommand(tle_text)

    print("\n--- FINAL FORMATTED TELECOMMAND ---")  # noqa: T201
    print(f"{telecommand}\n")  # noqa: T201


if __name__ == "__main__":
    main()
