"""Test TLE from GNSS data matches NORAD TLE."""

from pathlib import Path

import numpy as np
import satkit as sk
from cts1_mo_tools.cts1_gnss_to_tle_parser import fit_tle, load_gnss_file, tle_position

# real GNSS samples from CTS-SAT-1, Aug 10 2026
GNSS_FILE = Path(__file__).parent / "data" / "GNSSdata-aug10.txt"

# official CTS-SAT-1 TLE from Space-Track
NORAD_TLE = [
    "1 69015U 26100AM  26222.69061375  .00004346  00000-0  19055-3 0  9991",
    "2 69015  97.3931 119.6599 0007141 198.2375 161.8606 15.22486218 15118",
]


def test_tle_matches_norad() -> None:
    times, positions, velocities = load_gnss_file(GNSS_FILE)

    our_tle, _ = fit_tle(times, positions, velocities)
    norad_tle = sk.TLE.from_lines(NORAD_TLE)[0]

    for t in times:
        distance = np.linalg.norm(tle_position(our_tle, t) - tle_position(norad_tle, t))
        assert distance < 1.0
