# Repository Organization

This repo uses a `uv` workspace for Python packages (other than the GNU Radio aspects).

## Main Projects

- `gnu_radio_usrp_b210`: GNU Radio flowgraph with the USRP B210 radio. Not managed as a proper Python package.
- `src/cts1_gs_parent`: Basically empty and unused package. uv workspaces require a parent package.
- `packages/*`: Python packages managed by the uv workspace, which provide
- `packages/cts1_gs_forwarder`: CLI program, which interfaces with GNU Radio via TCP, and runs a server (TBD, maybe HTTP or MQTT) for multiple generic tools to connect, uplink, and downlink packets easily. Handles uplink packet auth (HMAC), maybe validation, and logs all operations to a file.
- `packages/cts1_gs_tool_lib`: Library for use in any tool(s) for connecting to the `cts1_gs_forwarder`.
- `packages/cts1_gs_database`: CLI program, which connects to the `cts1_gs_forwarder`, and writes all downlinked data to a database for later querying. Imports the `cts1_gs_tool_lib`.
- `packages/cts1_gs_dashboard`: GUI program which presents a dashboard of the current satellite state from the latest beacon package.
