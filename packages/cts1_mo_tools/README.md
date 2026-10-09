# `cts1_mo_tools` - Mission Operations Tools for CTS-SAT-1

Mission Operations tools that are mostly independent of the ground station capabilities (e.g., independent of uplink and downlink). 

These tools can run on the ground station computer, but are mostly meant to run on local personal computers for planning and analysis.

## Tools

### Pass Planning

* `cts1_make_bulk_uplink_agenda`
* `cts1_spreadsheet_to_agenda`
* `cts1_satnogs_interval`
* `cts1_tle_formatter`
* `cts1_plan_mpi_ops`

### Data Analysis and Assembly

* `cts1_decode_satnogs_packets`
* `cts1_picam_to_jpg`

### Web Dashboard

View the web dashboard at https://frontiersat.mooo.com/.

Alternatively, run the server backend and processing pipeline locally:

* `cts1_processing_pipeline`
* `cts1_serve_dashboard`
* `cts1_bootstrap_dashboard`
