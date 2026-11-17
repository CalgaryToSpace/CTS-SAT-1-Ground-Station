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

* Read the `everything_decoded.parquet` table from step 3, filtered to `BEACON_BASIC`/`BEACON_EXTENDED`/`BEACON_EXTENDED_V5PLUS` rows (all considered together, one timeline sorted by `received_at`).
* Logic: find the first beacon where an onboard counter that only ever counts up (`uptime_sec`, `eps_uptime_sec`, `duration_since_last_uplink_ms`) is lower than the previous beacon's -- that beacon is the first one received after an OBC reboot / EPS reboot / uplinked-commands event, respectively. The event's own UTC time is estimated as that beacon's `received_at` minus the counter's value.
* Output: `satellite_events_from_beacons.parquet` -- one row per detected event (unpivoted across the three event types), with `event_type`, `detected_at`, `estimated_event_at`, `time_since_event_when_detected_ms`, `obc_reboot_reason`, and `eps_reboot_reason`.

### Step 5: Reassemble telecommand responses

* Read the `everything_decoded.parquet` table from step 3, filtered to `TCMD_RESPONSE` rows. Independent of step 4 (both only depend on step 3).
* Logic: a response too long for one downlink frame is split across several packets sharing one `tcmd_ts_sent`, numbered `tcmd_response_seq_num` 1..`tcmd_response_max_seq_num`. Group packets by `tcmd_ts_sent` (plus the response code/duration/part count every part shares), order by sequence number, and join their `tcmd_response_text` back together. Any never-received part is filled with a full frame's worth (186) of `?` characters. Single-packet responses pass through as groups of one.
* Output: `reassembled_tcmd_responses.parquet` -- one row per telecommand response, with `tcmd_ts_sent`/`tcmd_sent_at`, `part_count`, `received_part_count`, `missing_seq_nums`, `is_complete`, `packet_ids`, and the reassembled `tcmd_response_text`.

### Step 6: Extract MPI frames

* Read the `everything_decoded.parquet` table from step 3, filtered to CRC-valid `BULK_FILE_DOWNLINK` rows that look like MPI science data. Independent of steps 4 and 5 (all only depend on step 3).
* Logic: an MPI data file is a `{"mpi_start":...}` JSON header, then repeated 20480-byte blocks of raw MPI UART stream each followed by a `{"uptime_ms":...,"timestamp_ms":...}` JSON timestamp, then a `{"data_lost_bytes":...}` footer. Bulk packets carry only a file offset (no file name), so distinct chunks are linked into same-file components from their content: a frame straddling two chunks passing its CRC (only when the earlier chunk contributes pixel bytes, since frame counters restart at 0 every recording), or the two chunks being received seconds apart in the same observation (also bridging up to 3 never-received chunks), or non-aligned chunks agreeing on their overlap. Within a component, consecutive JSON timestamps one to three 20480-byte blocks apart (with the missing timestamps in between placed by their length and the frame counters) form an anchor span; the JSON is carved out, and every fully-received frame ending in the span is CRC-checked, its counter checked against its position, and its time linearly interpolated between the span's two timestamps (by its last byte's position). A block with a timestamp on only one side (e.g. where the received data runs out, or the first block of a recording) is instead extrapolated from that timestamp at the MPI data rate measured over all anchor spans (median per frame length/integration period). Anything not certain is dropped: unsynced (`_N`) timestamps, a time resync mid-span, a block with any CRC failure or counter/position mismatch (including a recording's first block starting with stale data from the previous one), a span whose duration is off from its component's median by more than 10%, and any block next to such a rejected span.
* Output: `mpi_frames.parquet` -- one row per frame, with `collected_at`, `timing_method` (`interpolated` or `extrapolated`), `obc_uptime_ms` and `ms_since_time_sync` (both interpolated likewise), `time_sync_source`, `anchor_span_blocks`, the decoded housekeeping fields, `pixels` (JSON array), `file_offset`, `times_received` (how many times the whole frame came down, counted in `everything_decoded` packets), `collection_started_at`/`collection_first_frame_counter` (its MPI collection, below), and `packet_ids`/`first_received_at`/`last_received_at` lineage.
* MPI collections: frames (in `collected_at` order) are grouped into collections, each identified by its first frame (time and counter). A new one starts 20 minutes after the current one's first frame, or when the frame counter drops (a new recording, or the counter wrapping around) -- unless it's by at most 64 frames between frames within 2s of each other, which is just two frames' times landing out of order.
* Logs a table of what became of every distinct CRC-valid frame received: how many were kept (interpolated/extrapolated), and how many were lost, for each reason.

### Step 7: Fetch TLE history

* Download every TLE Space-Track has published for the satellite from the [`juliensimon/space-track-tle-history`](https://huggingface.co/datasets/juliensimon/space-track-tle-history) Hugging Face dataset (DuckDB reading its parquet files over `hf://`). Doesn't depend on any other step.
* Refetched at most every 6 hours (the dataset updates about daily, and the query takes tens of seconds). A failed fetch keeps the previous file and only logs a warning.
* Output: `tle_history.parquet` -- one row per TLE: `norad_id`, `epoch` (UTC), and the orbital elements (`inclination`, `raan`, `eccentricity`, `arg_perigee`, `mean_anomaly`, `mean_motion`, `mean_motion_dot`, `bstar`), plus `intl_designator` and `altitude_km`.

### Step 8: Summarize MPI collections

* Read `mpi_frames.parquet` from step 6, and (if present) `reassembled_tcmd_responses.parquet` from step 5 and `tle_history.parquet` from step 7. Depends on all three, so it runs last.
* Collection summary: each collection's start/end, frames present vs expected from the counters, largest gap, integration period(s), time sync source(s) and age, and MPI board temperature range, from its frames (in `mpi_frames.parquet`).
* Orbit: each collection's start and end are SGP4-propagated (satkit) from the TLE published right before each time (a backward `join_asof` on the TLE epochs -- see `orbit.py`) for the sub-satellite latitude/longitude, and its lighting is sampled every second in between: sunlit `yes` (full sunlight throughout), `no` (Earth's umbra throughout), or `partly`.
* MPI analyses: the satellite's own summary of an MPI data file, from a complete, successful `analyze_mpi_data_*` telecommand response (step 5's output, parsed as JSON) -- each file's primary analysis is the latest version (the `v<N>` in its `action`), then the largest, then the earliest received. A frame collection overlapping an analysis's `earliest`..`latest` (give or take 60s) is merged into it, and the merged collection's start/end become the analysis's `earliest`/`latest`. An analysis with no frames downlinked is still a collection.
* Strong signals: for each strong signal in an analysis, how many frames starting in its `bytes` range were downlinked (and recovered) during the analysis's `earliest`..`latest`.
* Output: `mpi_analyses.parquet` -- every distinct successful `analyze_mpi_data_*` response (identical re-downlinks kept once), one row each: the `analysis_*` fields below plus the raw `analysis_response_text`, and `analysis_strong_signal_frames_downlinked` (JSON array: for each strong signal, how many frames starting in its `bytes` range were downlinked and recovered during the analysis's `earliest`..`latest`).
* Output: `mpi_collections.parquet` -- one row per data collection event

### Step 9: Clean MPI frames

* Read `mpi_frames.parquet` from step 6. Only depends on step 6, but runs after step 8.
* Logic: the MPI cleaning from simple_sat_ops' MPI viewer (`utils/mpi_viewer.c`), in three steps, each building on the one before -- see `step_9_clean_mpi_frames/cleaning.py`. Step 1 undoes the instrument's own background subtraction (which uses a 0 V frame kept every 256 frames, landing frames on a ~2000 DN offset), leaving raw counts. Step 2 detrends each frame by taking off a least-squares line through its first two and last two pixels. Step 3 takes off a local background: the pixel-by-pixel median of the 0 V first frames of the 5 sweeps centred on each sweep. Steps 2 and 3 put the 2000 DN offset back on. Each frame collection is cleaned on its own, and the whole file is recomputed every run, since a frame's cleaned values depend on its neighbours.
* Output: `mpi_frames_cleaned.parquet` -- `mpi_frames.parquet`'s rows and columns, plus `pixels_clean_step_1`, `pixels_clean_step_2` and `pixels_clean_step_3` (JSON arrays on the same pixel indices as `pixels`; null where a frame has no value at that step). The web UI's MPI Vis page reads this file.

### Step 10: List satellite files

* Read `reassembled_tcmd_responses.parquet` from step 5.
* Logic: every response naming a file on the satellite -- directory listings (`fs_list_directory_json`: `{"/dir/name": size, "/dir/sub/": null, ...}`), bulk downlink headers (`bulk_downlink_start*`), ADCS SD file copies (`adcs_get_latest_sd_file_blob*`), file maps (`get_file_map*`), MPI analyses (`analyze_mpi_data*`), and any other response with an `action` and a `file` -- is one mention of that file. Paths are normalized to the satellite's usual form (`camera/x.img`: no leading `/`, no `//` or `./`), parsed tolerantly so a response with a lost frame (`?`-filled by step 5) still gives up everything around the gap.
* Listing quirks (firmware `LFS_list_directory_json_dict`): it takes an `offset`/`count`, with nothing in the response saying a listing is only a page -- seen in flight as listings of just the first 28 of 61 files. Since littlefs lists in name order, a listing only counts as covering the names up to its last entry (and from its first, when it has no `./` entry, i.e. a later page). Its keys echo the directory argument as typed (hence the normalization), and it isn't guaranteed to be valid JSON, so entries are matched one by one and completeness comes from step 5.
* Output: `satellite_file_mentions.parquet` -- one row per (response, file): `source`, `action`, `file_path` and `raw_path` (as given), size, SHA-256, a header's `downlink_offset`/`downlink_length`, and a listing's directory and covered name range.
* Output: `satellite_files.parquet` -- one row per file: latest size and SHA-256 (and every one seen), sources, first/last mentioned, `path_sha256_prefix(es)` (first 3 bytes of SHA-256 of the path as a downlink command gave it -- what the v5+ beacons report), and `listing_status`: `listed`, `not_in_latest_listing` (likely deleted since), `newer_than_latest_listing` (likely created since) or `never_listed`, against the latest complete listing covering its name.

### Step 11: Reassemble downlinked files

* Read `everything_decoded.parquet` (CRC-valid `BULK_FILE_DOWNLINK` chunks and live `BEACON_EXTENDED_V5PLUS` beacons) from step 3, and step 10's outputs.
* Attribution: a bulk chunk carries an offset but no file name. Each chunk is tied to a file by whichever came last: the latest downlink header (only for chunks in its `offset`..`offset+length`, until its downlink should have finished at 500 B/s plus 2 minutes), or the latest live beacon within 2 minutes (its `bulk_downlink_file_path_sha256_prefix`, looked up in step 10's files; a beacon with no prefix means nothing is downlinking). Failing those, the next beacon naming a file within 2 minutes. Otherwise the chunk is unattributed -- typically a downlink from a time-tagged command, before the v5 beacons, whose header went out before any station could hear it.
* Downlinks: consecutive chunks of one file, with no gap over 2 minutes and no new header between them.
* Consistency: before the `bulk_downlink_start_blob` fix (firmware issue #653), a new downlink didn't always close the previous file, so chunks after a header could be another file's (seen in flight). A file's downlinks are clustered by whether they agree on every overlapping byte; the cluster matching a SHA-256 the satellite reported is kept, else one not carrying another verified file's bytes, else the largest. The rest are rejected, with the reason.
* Gap filling: unattributed chunks received within 5 minutes of one of a file's kept downlinks (and near no other file's) fill its gaps (`time_window` attribution) -- taken per unattributed downlink, nearest first, only if every chunk fits the file's size, agrees with every byte it already has, and fills at least one missing byte. Ones from across a header, or outside the byte range the file's own header was sending, are from a different downlink, so they're only taken if they complete the file with a SHA-256 the satellite reported. On the data so far: 10 more files verified, none with the wrong hash.
* Reassembly: each file's kept chunks, over every downlink of it, with `file_reassembly` (per-byte good/missing/conflicting), padded with missing bytes to the file's known size, and its SHA-256 checked against the satellite's. Chunks of an unknown path's prefix are reassembled per prefix, and each unattributed downlink on its own.
* Output: `file_downlinks.parquet` -- one row per downlink: file, attribution(s), start/end, chunks, offsets, bytes received, its header's fields, and `rejected_reason`.
* Output: `downlinked_files.parquet` -- one row per file: size, good/missing/conflicting byte counts, `is_complete`, `sha256` vs `expected_sha256s` (`sha256_verified`), downlink counts, `segments` (JSON), `content_kind`, an image preview where one is detected (a PiCAM image converted to JPG), and the reassembled bytes (`data`). The web UI's Downlinked Files page reads this file.

### Parquet steps

* `cts1_processing_pipeline parquet_steps` runs every parquet-in-parquet-out step (2-6 and 8-11) once, in order -- e.g. to regenerate their outputs after changing one of them.
* Unlike the daemon, it doesn't list, download or fetch anything (steps 0, 1 and 7), and runs once rather than continuously. Step 8 uses whatever `tle_history.parquet` is already in `--data-dir`.

### Partitioned outputs and incremental runs

* `raw_packets.parquet`, `distinct_packets_over_time.parquet` and `everything_decoded.parquet` are directories holding one file per UTC month of `received_at` (e.g. `everything_decoded.parquet/2026-11.parquet`). Polars reads such a directory just like a single file (`pl.scan_parquet(path)`). Never put any other file in one: polars then refuses to read it. See `partitions.py`.
* Every month file of a directory has the same schema. For `everything_decoded`, that's `step_3_decode_packets/schema.py`'s `DECODED_SCHEMA`: when the decoder grows a field, add it there (a test checks this).
* Step 1 only re-exports months whose row count in DuckDB changed. Steps 2 and 3 run either in full or incrementally (`incremental=True`), reprocessing only what's affected by the months of their input that changed since their last run, as recorded in `<output>.state.json` next to their output. An incremental run's output is the same as a full run's.
* The daemon's first run after it starts is always in full (so a new deploy reprocesses everything); every run after that is incremental, and picks up history landed by idle backfill too. Deleting an output directory (or its `.state.json`) forces a full rebuild of it on the next run.

### Daemon

* Runs steps 0-11 continuously instead of one-off: an initial backfill of `--start` (default: 24h), then a rerun of steps 0 through 11 every `--interval` minutes (default: 15).
* `--start` is resolved to an absolute time once, at startup, and every run covers the same span: step 0 only re-lists the tail of the listing windows that haven't settled yet (about the last hour of observations), and step 1 only decodes observations not already recorded in `decoder_runs`. So a SatNOGS observation that was still uploading during one run is picked up by the next.
* With `--idle-backfill` (or `CTS1_IDLE_BACKFILL=1`), the time between runs is spent backfilling steps 0 and 1 for the history before `--start`, back to the satellite's first observations: one 12h listing window at a time, newest first, picking only windows step 0 hasn't listed or with observations step 1 hasn't decoded. A trigger request or the next scheduled run is honoured between windows. Steps 2-6 pick up the backfilled packets on the next run.
* Runs until interrupted (Ctrl+C).

### Web UI

* Read any/all of the above parquet files/tables.
* Serve a multi-user, web-based UI for exploring packets, exporting files, etc.
