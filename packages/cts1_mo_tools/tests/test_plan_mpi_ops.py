from datetime import UTC, datetime, timedelta

import numpy as np
from cts1_mo_tools import cts1_plan_mpi_ops as p

T0 = datetime(2026, 11, 2, 5, 0, 0, tzinfo=UTC)


def _t(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


def _sec(t: datetime) -> float:
    return (t - T0).total_seconds()


def _eph(qd: list[float], mlt: list[float]) -> p.Ephemeris:
    n = len(qd)
    z = np.zeros(n)
    return p.Ephemeris(z, z, z, np.array(qd), np.array(mlt), z)


def test_in_region_mask_wraps_mlt_and_splits_hemispheres() -> None:
    aurora = p.MagRegion("aurora", (62.0, 72.0), (18.0, 6.0))
    eph = _eph([65, 65, 65, -65, 75, 65], [23.0, 3.0, 12.0, 1.0, 1.0, 18.0])
    assert p.in_region_mask(aurora, "north", eph).tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
    ]
    assert p.in_region_mask(aurora, "south", eph).tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_classify_lighting() -> None:
    assert p.classify_lighting(np.array([0.0, 0.5, 1.0])).tolist() == [
        "umbra",
        "penumbra",
        "sunlit",
    ]


def _plan_camera(crossings: list[p.Crossing], n: int = 1) -> p.CameraPlan:
    return p.plan_camera_events(
        crossings,
        photos_per_crossing=n,
        setup_lead=timedelta(seconds=30),
        capture_duration=timedelta(seconds=25),
        lighting_mode_at=lambda _t: "n",
        filename_format="camera/%Y-%m-%d_%H%M%SZ.img",
    )


def test_camera_setup_then_capture_mid_crossing() -> None:
    plan = _plan_camera([p.Crossing("aurora-N", _t(100), _t(200))])
    assert [(_sec(e.timestamp), e.command) for e in plan.events] == [
        (120, "CTS1+camera_setup()"),
        (150, "CTS1+camera_capture(camera/2026-11-02_050230Z.img,n)"),
    ]
    assert [(_sec(a), _sec(b)) for a, b in plan.blackouts] == [(120, 175)]


def test_camera_skips_overlapping_photos() -> None:
    plan = _plan_camera(
        [p.Crossing("a", _t(0), _t(100)), p.Crossing("b", _t(60), _t(120))]
    )
    # Photo for "b" would set up at 60, while "a"'s capture (at 50) runs to 75.
    assert len(plan.events) == 2


def test_mpi_avoids_camera_and_overlaps() -> None:
    events = p.plan_mpi_events(
        [p.Crossing("a", _t(0), _t(200)), p.Crossing("b", _t(150), _t(260))],
        collection=timedelta(seconds=60),
        break_duration=timedelta(seconds=5),
        min_collection=timedelta(seconds=10),
        filename_format="mpi_data/%Y-%m/%Y-%m-%d_%H%M%SZ.mpi",
        blackouts=[(_t(120), _t(175))],
    )
    pairs = [
        (_sec(events[i].timestamp), _sec(events[i + 1].timestamp))
        for i in range(0, len(events), 2)
    ]
    # Crossing "a": 0-60, 65-115 (stops 5 s before camera setup at 120).
    # Resumes 5 s after the camera (180) until "a" exits, then "b" takes over.
    assert pairs == [(0, 60), (65, 115), (180, 200), (205, 260)]
    assert events[0].command == (
        "CTS1+mpi_enable_active_mode(mpi_data/2026-11/2026-11-02_050000Z.mpi)"
    )


def test_make_directory_events() -> None:
    events = [
        p.Event(_t(100), "CTS1+mpi_enable_active_mode(mpi_data/2026-11/x.mpi)", "a"),
        p.Event(_t(200), "CTS1+camera_capture(camera/2026-12/y.img,n)", "a"),
    ]
    mkdirs = p.plan_make_directory_events(events)
    assert [(_sec(e.timestamp), e.command) for e in mkdirs] == [
        (87, "CTS1+fs_make_directory(mpi_data)"),
        (88, "CTS1+fs_make_directory(mpi_data/2026-11)"),
        (89, "CTS1+fs_make_directory(camera)"),
        (90, "CTS1+fs_make_directory(camera/2026-12)"),
    ]
