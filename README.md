# CTS-SAT-1-Ground-Station
Ground Station documentation and software for the CTS-SAT-1 (FrontierSat) 3U CubeSat mission.
This software intends to receive, store, and display telemetry from the satellite, and is used to send
telecommands during live passes. 

## Goals

1. Receive and store telemetry from FrontierSat during ground station passes.
2. Provide a real-time dashboard for monitoring satellite health and status.
3. Enable commanding of the satellite over the ground station link.
4. Provide additional tools, documentation and parsing capabilities.

## Features

* Python-based, structured as a `uv` workspace with multiple cooperating packages.
* **`cts1_gs_forwarder`** — receives frames from the radio/modem and forwards them to the rest of the stack.
* **`cts1_gs_tool_lib`** — shared library of parsing and protocol utilities.
* **`cts1_gs_database`** — stores telemetry and command history in a local database.
* **`cts1_gs_dashboard`** — web-based dashboard for real-time telemetry display and commanding.