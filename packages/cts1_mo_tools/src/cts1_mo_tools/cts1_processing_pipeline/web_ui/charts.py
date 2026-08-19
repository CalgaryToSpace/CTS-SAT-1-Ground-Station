"""Field groupings + ECharts option builders for beacon telemetry.

Every chart is a line series with symbols shown, which reads as both a line
graph (trend over time) and a scatter graph (each received beacon is a
visible dot) at once -- one widget satisfies both without duplicating charts.
Fields whose decoded value is a category label (an enum string, or a bool)
get a category y-axis instead of a numeric one, since they aren't really
"a number over time".
"""

from __future__ import annotations

__all__ = ["BEACON_CHART_GROUPS", "OTHER_CHART_GROUPS", "ChartSpec", "chart_option"]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

# Palette shared across every multi-series chart on the page.
_SERIES_COLORS = [
    "#5B8FF9",
    "#5AD8A6",
    "#F6BD16",
    "#E8684A",
    "#6DC8EC",
    "#9270CA",
    "#FF9D4D",
    "#269A99",
]


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """One chart: one or more columns plotted against `received_at`."""

    title: str
    columns: list[str]
    unit: str = ""
    categorical: bool = False
    labels: list[str] = field(default_factory=list[str])

    def series_label(self, column: str, index: int) -> str:
        if self.labels:
            return self.labels[index]
        return column


# -- Beacon telemetry (BEACON_BASIC / BEACON_EXTENDED rows only) -------------

BEACON_CHART_GROUPS: list[tuple[str, list[ChartSpec]]] = [
    (
        "Identity & Timing",
        [
            ChartSpec("Uptime", ["uptime_sec"], unit="s"),
            ChartSpec(
                "Time Since Last Uplink", ["duration_since_last_uplink_ms"], unit="ms"
            ),
            ChartSpec(
                "Beacon Count Since Boot", ["total_beacon_count_since_boot"], unit=""
            ),
            ChartSpec(
                "Telecommand Queue",
                ["total_tcmd_queued_count", "pending_queued_tcmd_count"],
                unit="",
                labels=["total queued", "pending"],
            ),
            ChartSpec("Time Sync Source", ["last_time_sync_source"], categorical=True),
        ],
    ),
    (
        "OBC",
        [
            ChartSpec("OBC Temperature", ["obc_temperature_C"], unit="°C"),
            ChartSpec(
                "OBC Active Oscillator (extended)",
                ["obc_active_oscillator_MHz"],
                unit="MHz",
            ),
            ChartSpec(
                "OBC ADC Battery Voltage (extended)",
                ["obc_adc_battery_voltage_V"],
                unit="V",
            ),
            ChartSpec("Reboot Reason", ["reboot_reason"], categorical=True),
            ChartSpec("Filesystem Mounted", ["is_fs_mounted"], categorical=True),
        ],
    ),
    (
        "EPS Battery",
        [
            ChartSpec("Battery Voltage", ["eps_battery_voltage_V"], unit="V"),
            ChartSpec("Battery Percent", ["eps_battery_percent"], unit="%"),
            ChartSpec(
                "Battery Temperatures",
                ["eps_battery_temperature_0_C", "eps_battery_temperature_1_C"],
                unit="°C",
                labels=["sensor 0", "sensor 1"],
            ),
            ChartSpec("EPS Uptime", ["eps_uptime_sec"], unit="s"),
            ChartSpec("EPS Fault Count", ["eps_total_fault_count"], unit=""),
            ChartSpec("EPS Error Code", ["eps_error_code"], unit=""),
            ChartSpec("EPS Mode", ["eps_mode"], categorical=True),
            ChartSpec("EPS Reset Cause", ["eps_reset_cause"], categorical=True),
        ],
    ),
    (
        "EPS Power Flow",
        [
            ChartSpec(
                "PCU Power, Instantaneous",
                ["eps_total_pcu_power_input_W", "eps_total_pcu_power_output_W"],
                unit="W",
                labels=["input", "output"],
            ),
            ChartSpec(
                "PCU Power, Average",
                [
                    "eps_total_avg_pcu_power_input_W",
                    "eps_total_avg_pcu_power_output_W",
                ],
                unit="W",
                labels=["input", "output"],
            ),
            ChartSpec(
                "Net Battery / Distributed Power, Average (extended)",
                [
                    "eps_total_avg_net_battery_power_W",
                    "eps_total_avg_power_distributed_W",
                ],
                unit="W",
                labels=["net battery", "distributed"],
            ),
        ],
    ),
    (
        "EPS Solar Channels (extended)",
        [
            ChartSpec(
                "MPPT Input Voltage",
                [
                    "eps_pcu_ch0_volt_in_mppt_V",
                    "eps_pcu_ch1_volt_in_mppt_V",
                    "eps_pcu_ch2_volt_in_mppt_V",
                    "eps_pcu_ch3_volt_in_mppt_V",
                ],
                unit="V",
                labels=["ch0", "ch1", "ch2", "ch3"],
            ),
            ChartSpec(
                "MPPT Input Current",
                [
                    "eps_pcu_ch0_curr_in_mppt_A",
                    "eps_pcu_ch1_curr_in_mppt_A",
                    "eps_pcu_ch2_curr_in_mppt_A",
                    "eps_pcu_ch3_curr_in_mppt_A",
                ],
                unit="A",
                labels=["ch0", "ch1", "ch2", "ch3"],
            ),
            ChartSpec(
                "MPPT Output Current",
                [
                    "eps_pcu_ch0_curr_ou_mppt_A",
                    "eps_pcu_ch1_curr_ou_mppt_A",
                    "eps_pcu_ch2_curr_ou_mppt_A",
                    "eps_pcu_ch3_curr_ou_mppt_A",
                ],
                unit="A",
                labels=["ch0", "ch1", "ch2", "ch3"],
            ),
        ],
    ),
    (
        "CTS1 State & RF Switch",
        [
            ChartSpec(
                "CTS1 Operation State", ["cts1_operation_state"], categorical=True
            ),
            ChartSpec("RBF Pin State", ["rbf_pin_state"], categorical=True),
            ChartSpec(
                "Active RF Switch Antenna", ["active_rf_switch_antenna"], unit=""
            ),
            ChartSpec(
                "RF Switch Control Mode",
                ["active_rf_switch_control_mode"],
                categorical=True,
            ),
        ],
    ),
    (
        "MPI",
        [
            ChartSpec("MPI RX Mode", ["mpi_rx_mode"], categorical=True),
            ChartSpec(
                "MPI Transceiver State", ["mpi_transceiver_state"], categorical=True
            ),
            ChartSpec(
                "MPI Last Stop Reason",
                ["mpi_last_reason_for_stopping"],
                categorical=True,
            ),
            ChartSpec(
                "MPI Last Temperature (extended)",
                ["mpi_last_temperature_C"],
                unit="°C",
            ),
        ],
    ),
    (
        "GNSS",
        [
            ChartSpec(
                "GNSS UART Interrupt Enabled",
                ["gnss_uart_interrupt_enabled"],
                categorical=True,
            ),
            ChartSpec("GNSS RX Mode", ["gnss_rx_mode"], categorical=True),
        ],
    ),
    (
        "ADCS Attitude (extended)",
        [
            ChartSpec(
                "Estimated Attitude",
                [
                    "adcs_estimated_roll_angle_deg",
                    "adcs_estimated_pitch_angle_deg",
                    "adcs_estimated_yaw_angle_deg",
                ],
                unit="deg",
                labels=["roll", "pitch", "yaw"],
            ),
            ChartSpec(
                "Estimated Angular Rate",
                [
                    "adcs_estimated_rate_x_deg_per_sec",
                    "adcs_estimated_rate_y_deg_per_sec",
                    "adcs_estimated_rate_z_deg_per_sec",
                ],
                unit="deg/s",
                labels=["x", "y", "z"],
            ),
            ChartSpec(
                "Angular Rate Norm",
                ["adcs_angular_rate_norm_deg_per_sec"],
                unit="deg/s",
            ),
            ChartSpec(
                "Attitude Estimation Mode",
                ["adcs_attitude_estimation_mode"],
                categorical=True,
            ),
            ChartSpec("Control Mode", ["adcs_control_mode"], categorical=True),
            ChartSpec("Run Mode", ["adcs_run_mode"], categorical=True),
            ChartSpec("ASGP4 Mode", ["adcs_asgp4_mode"], categorical=True),
            ChartSpec(
                "Sun Above Local Horizon",
                ["adcs_sun_above_local_horizon"],
                categorical=True,
            ),
        ],
    ),
    (
        "ADCS Sensors (extended)",
        [
            ChartSpec(
                "Magnetic Field",
                [
                    "adcs_magnetic_field_x_uT",
                    "adcs_magnetic_field_y_uT",
                    "adcs_magnetic_field_z_uT",
                ],
                unit="µT",
                labels=["x", "y", "z"],
            ),
            ChartSpec(
                "Raw Coarse Sun Sensors",
                [
                    "adcs_raw_css_1",
                    "adcs_raw_css_2",
                    "adcs_raw_css_3",
                    "adcs_raw_css_4",
                    "adcs_raw_css_5",
                    "adcs_raw_css_6",
                    "adcs_raw_css_7",
                    "adcs_raw_css_9",
                ],
                unit="raw",
                labels=["css1", "css2", "css3", "css4", "css5", "css6", "css7", "css9"],
            ),
        ],
    ),
]

# -- Non-beacon-specific fields, computed over every decoded packet ----------

OTHER_CHART_GROUPS: list[tuple[str, list[ChartSpec]]] = [
    (
        "Link Quality (all packets)",
        [
            ChartSpec("RSSI", ["rssi_db"], unit="dB"),
            ChartSpec(
                "Reed-Solomon Corrected Errors", ["rs_corrected_error_count"], unit=""
            ),
            ChartSpec("Received Packet Length", ["data_length_bytes"], unit="bytes"),
        ],
    ),
]


def _numeric_series(df: pl.DataFrame, column: str) -> list[list[Any]]:
    sub = df.select("received_at", column).drop_nulls()
    return [
        [ts.isoformat(), value]
        for ts, value in zip(sub["received_at"], sub[column], strict=True)
    ]


def _categorical_series(df: pl.DataFrame, column: str) -> list[list[Any]]:
    sub = df.select("received_at", column).drop_nulls()
    return [
        [ts.isoformat(), str(value)]
        for ts, value in zip(sub["received_at"], sub[column], strict=True)
    ]


def chart_option(spec: ChartSpec, df: pl.DataFrame) -> dict[str, Any] | None:
    """Build an ECharts `option` dict for `spec` from `df`, or None if empty.

    `df` must contain `received_at` plus every column in `spec.columns`.
    """
    available = [c for c in spec.columns if c in df.columns]
    if not available:
        return None

    # Perf notes (this page can have 40+ of these mounted/rebuilt at once, so
    # every bit of per-chart render cost multiplies):
    #  - animation off: skips the mount-in / update transition entirely.
    #  - no `dataZoom` slider: the slider is its own mini chart + DOM + drag
    #    handlers per instance; `inside` (wheel/pinch/drag-pan on the plot
    #    itself) covers zooming for a fraction of the cost.
    #  - `large`/`largeThreshold`: switches ECharts to its batched, symbol-
    #    skipping fast path once a series has enough points to matter.
    if spec.categorical:
        column = available[0]
        data = _categorical_series(df, column)
        if not data:
            return None
        categories = sorted({row[1] for row in data})
        return {
            "animation": False,
            "title": {"text": spec.title, "textStyle": {"fontSize": 14}},
            "grid": {"left": 110, "right": 20, "top": 40, "bottom": 30},
            "tooltip": {"trigger": "item"},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "category", "data": categories},
            "dataZoom": [{"type": "inside"}],
            "series": [
                {
                    "type": "scatter",
                    "symbolSize": 8,
                    "data": data,
                    "color": _SERIES_COLORS[0],
                    "large": True,
                    "largeThreshold": 200,
                }
            ],
        }

    series: list[dict[str, Any]] = []
    any_data = False
    for i, column in enumerate(available):
        data = _numeric_series(df, column)
        if data:
            any_data = True
        series.append(
            {
                "name": spec.series_label(column, i),
                "type": "line",
                "showSymbol": True,
                "symbolSize": 5,
                "connectNulls": False,
                "data": data,
                "color": _SERIES_COLORS[i % len(_SERIES_COLORS)],
                "sampling": "lttb",
                "large": True,
                "largeThreshold": 200,
            }
        )
    if not any_data:
        return None

    y_label = spec.unit
    return {
        "animation": False,
        "title": {"text": spec.title, "textStyle": {"fontSize": 14}},
        "grid": {"left": 60, "right": 20, "top": 40, "bottom": 30},
        "tooltip": {"trigger": "axis"},
        "legend": {"show": len(available) > 1, "top": 26},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": y_label, "scale": True},
        "dataZoom": [{"type": "inside"}],
        "series": series,
    }
