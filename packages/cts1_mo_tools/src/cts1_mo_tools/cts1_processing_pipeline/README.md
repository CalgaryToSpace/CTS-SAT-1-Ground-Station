# `cts1_processing_pipeline`

Every step below defaults its data-file paths (the DuckDB database, every
parquet file) from the `CTS1_DATA_DIR` environment variable (`./output` if
unset) -- see [DEPLOY.md](../../../DEPLOY.md) for running the daemon and
web UI as two containers sharing that directory.

## Steps

### Step 0: List observations.

* List every SatNOGS observation for the satellite (all statuses) into DuckDB, without downloading anything.
* Logic: query the SatNOGS API in 12-hour listing windows aligned to 00:00/12:00 UTC, each one filtered to observations starting within the window plus 25 minutes either side (so an observation right on a boundary lands in both neighbours; `raw_observations` is upserted by `id`, so the overlap costs nothing). The current window's query stops at the current time.
* Every listing attempt is logged, and a window is only listed again if it has never been listed successfully, or its latest listing started before it settled (within 2 hours of the end of its query range, when observations in it may still be in progress or uploading), or its `needs_refetch` was set to true by hand, or `--refetch-all` is passed. Re-listing a still-settling window is incremental: only observations starting from 45 minutes before its previous listing are queried, with one final full listing once it settles. So repeated runs (the daemon) only query about the last hour of observations, and an interrupted backfill picks up where it left off.
* Output: DuckDB database (`cts1_processing_pipeline.duckdb`), exported to parquet files at the end of each run.
* Output Tables:
    * `raw_observations` -- one row per SatNOGS observation.
    * `observation_listing_windows` -- one row per listing window listed so far: its latest listing (`last_listed_at`, `last_query_start_gt`/`last_query_start_lt`, `last_observation_count`), how many times it's been listed, and `needs_refetch`.
    * `observation_listing_history` -- one row per listing attempt, append-only, including failed ones (`succeeded`, `error`).

### Step 1: Download and demodulate.

* Read the observations step 0 listed from `raw_observations` (filtered to the NORAD ID and, optionally, `--start`), skipping any observation/decoder pair already recorded in `decoder_runs`.
* Download and demodulate each audio file.
* Download the raw data files produced by the flowgraphs (one file per frame).
* Output: the same DuckDB database as step 0, which gets checkpointed into parquet files.
* Output Tables:
    * `decoder_runs`
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

### Step 4: Detect satellite events from beacons

* Read the `everything_decoded.parquet` table from step 3, filtered to `BEACON_BASIC`/`BEACON_EXTENDED` rows (both considered together, one timeline sorted by `received_at`).
* Logic: find the first beacon where an onboard counter that only ever counts up (`uptime_sec`, `eps_uptime_sec`, `duration_since_last_uplink_ms`) is lower than the previous beacon's -- that beacon is the first one received after an OBC reboot / EPS reboot / uplinked-commands event, respectively. The event's own UTC time is estimated as that beacon's `received_at` minus the counter's value.
* Output: `satellite_events_from_beacons.parquet` -- one row per detected event (unpivoted across the three event types), with `event_type`, `detected_at`, `estimated_event_at`, `time_since_event_when_detected_ms`, `obc_reboot_reason`, and `eps_reboot_reason`.

### Step 5: Reassemble telecommand responses

* Read the `everything_decoded.parquet` table from step 3, filtered to `TCMD_RESPONSE` rows. Independent of step 4 (both only depend on step 3).
* Logic: a response too long for one downlink frame is split across several packets sharing one `tcmd_ts_sent`, numbered `tcmd_response_seq_num` 1..`tcmd_response_max_seq_num`. Group packets by `tcmd_ts_sent` (plus the response code/duration/part count every part shares), order by sequence number, and join their `tcmd_response_text` back together. Any never-received part is filled with a full frame's worth (186) of `?` characters. Single-packet responses pass through as groups of one.
* Output: `reassembled_tcmd_responses.parquet` -- one row per telecommand response, with `tcmd_ts_sent`/`tcmd_sent_at`, `part_count`, `received_part_count`, `missing_seq_nums`, `is_complete`, `packet_ids`, and the reassembled `tcmd_response_text`.

### Step 6: Extract MPI frames

* Read the `everything_decoded.parquet` table from step 3, filtered to CRC-valid `BULK_FILE_DOWNLINK` rows that look like MPI science data. Independent of steps 4 and 5 (all only depend on step 3).
* Logic: an MPI data file is a `{"mpi_start":...}` JSON header, then repeated 20480-byte blocks of raw MPI UART stream each followed by a `{"uptime_ms":...,"timestamp_ms":...}` JSON timestamp, then a `{"data_lost_bytes":...}` footer. Bulk packets carry only a file offset (no file name), so distinct chunks are linked into same-file components from their content: a frame straddling two chunks passing its CRC (only when the earlier chunk contributes pixel bytes, since frame counters restart at 0 every recording), or the two chunks being received seconds apart in the same observation (also bridging up to 3 never-received chunks), or non-aligned chunks agreeing on their overlap. Within a component, consecutive JSON timestamps exactly one (or two, when the middle timestamp was never received) 20480-byte blocks apart form an anchor span; the JSON is carved out, and every fully-received frame ending in the span is CRC-checked, its counter checked against its position, and its time linearly interpolated between the span's two timestamps (by its last byte's position). Anything not certain is dropped: the stale first block after a `mpi_start` header, unsynced (`_N`) timestamps, a time resync mid-span, a span with any CRC failure or counter/position mismatch, and a span whose duration is off from its component's median by more than 10%.
* Output: `mpi_frames.parquet` -- one row per frame, with `collected_at` (interpolated), `obc_uptime_ms` and `ms_since_time_sync` (both interpolated likewise), `time_sync_source`, `anchor_span_blocks`, the decoded housekeeping fields, `pixels` (JSON array), `file_offset`, `times_received` (how many times the whole frame came down, counted in `everything_decoded` packets), and `packet_ids`/`first_received_at`/`last_received_at` lineage.

### Daemon

* Runs steps 0-6 continuously instead of one-off: an initial backfill of `--start` (default: 24h), then a rerun of steps 0 through 6 every `--interval` minutes (default: 15).
* `--start` is resolved to an absolute time once, at startup, and every run covers the same span: step 0 only re-lists the tail of the listing windows that haven't settled yet (about the last hour of observations), and step 1 only decodes observations not already recorded in `decoder_runs`. So a SatNOGS observation that was still uploading during one run is picked up by the next.
* With `--idle-backfill` (or `CTS1_IDLE_BACKFILL=1`), the time between runs is spent backfilling steps 0 and 1 for the history before `--start`, back to the satellite's first observations: one 12h listing window at a time, newest first, picking only windows step 0 hasn't listed or with observations step 1 hasn't decoded. A trigger request or the next scheduled run is honoured between windows. Steps 2-6 pick up the backfilled packets on the next run.
* Runs until interrupted (Ctrl+C).

### Web UI

* Read any/all of the above parquet files/tables.
* Serve a multi-user, web-based UI for exploring packets, exporting files, etc.
