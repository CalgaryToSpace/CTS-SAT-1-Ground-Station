# `cts1_processing_pipeline`

Every step below defaults its data-file paths (the DuckDB database, every
parquet file) from the `CTS1_DATA_DIR` environment variable (`./output` if
unset) -- see [DEPLOY.md](../../../DEPLOY.md) for running the daemon and
web UI as two containers sharing that directory.

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


### Daemon

* Runs steps 1-3 continuously instead of one-off: an initial backfill of `--start` (default: 24h), then every `--interval` minutes (default: 15), requeries step 1 for observations starting in the trailing `interval + 30` minutes and reruns steps 2 and 3.
* The 30-minute overlap on every requery catches a SatNOGS observation that was still uploading/being vetted during the previous poll; it doesn't waste decode time since step 1 already skips any observation/decoder pair already recorded in `decoder_runs`.
* Runs until interrupted (Ctrl+C).

### Web UI

* Read any/all of the above parquet files/tables.
* Serve a multi-user, web-based UI for exploring packets, exporting files, etc.
