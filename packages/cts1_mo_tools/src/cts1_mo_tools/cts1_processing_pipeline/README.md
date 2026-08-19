# `cts1_processing_pipeline`

## Steps

### Step 1: Download and demodulate.

* Download and demodulate each audio file.
* Download the raw data files produced by the flowgraphs (one file per frame).
* Output: DuckDB database (`cts1_processing_pipeline.duckdb`), which gets checkpointed into parquet files.
* Output Tables:
    * `decoder_runs`
    * `raw_observations`
    * `raw_packets`

### Step 2: De-duplicate packets over time

* Read the `raw_observations` and `raw_packets` tables (from parquets) from step 1.
* Reprocess into a table with one row per packet.
* Logic: De-duplicate across decoding tools. Deduplicate within time windows across observations.
* Output: `distinct_packets_over_time.parquet`

### Step 3: Decode packets

* Read the `distinct_packets_over_time.parquet` table from step 2.
* Run the logic of the `cts1_decode_satnogs_packets` script to produce a super-wide table of all the packets.
* Output: `everything_decoded.parquet`


### Web UI

* Read any/all of the above parquet files/tables.
* Serve a multi-user, web-based UI for exploring packets, exporting files, etc.
