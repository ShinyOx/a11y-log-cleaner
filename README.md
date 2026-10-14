# a11y-log-cleaner

A Python/Pandas pipeline that cleans and summarizes interaction logs from an AI-assisted 3D modeling tool for blind and low-vision (BLV) users, modeled on the kind of system described in the A11yShape paper (OpenSCAD plus an LLM assistant).

> **All data in this repo is synthetic.** I don't have access to any real study data. The log formats are my own guess at what such a tool might record, built to practice research-grade data cleaning. Column names would need to be adapted to a real logger.

## What it does

It takes two raw logs:

| Log | One row per | Key columns |
|---|---|---|
| OpenSCAD edit log | editor event (code edit, render, undo, hierarchy selection, description request, version save) | `session_id`, `timestamp`, `event_type`, `input_modality`, `render_ms` |
| LLM request log | request sent to the AI assistant | `request_id`, `sent_at`, `received_at`, `request_type`, `status` |

And produces:

| Output | Contents |
|---|---|
| `clean_edit_log.csv` | cleaned editor events, sorted by session and time |
| `clean_llm_log.csv` | cleaned LLM requests with computed `latency_ms` and `n_attempts` |
| `session_summary.csv` | one row per session: duration, edits, render error rate, screen-reader share, LLM failure rate, median LLM latency |
| `excluded_rows.csv` | every row removed, with the reason |
| `qc_report.json` | counts of every row removed or flagged, by reason |

## Cleaning rules

- **No invented data.** Impossible values (negative render time, received-before-sent clock skew) are set to missing and flagged in a `qc_flag` column. They are not replaced with a median or guess.
- **No silent drops.** Every removed row goes to `excluded_rows.csv` with a reason: exact duplicate, missing session ID, unparseable timestamp, system event, unknown event type, or superseded by a retry.
- **Keyboard and screen-reader input is real user input.** BLV users don't click and drag, so only system heartbeats are excluded from activity counts.
- **Retries are collapsed.** Requests that share a `request_id` keep the final attempt, and `n_attempts` records how many tries it took.
- **Raw files are never overwritten.** Outputs go to a separate folder.

## Run it

Requires Python 3.10+ with pandas and numpy.

```bash
pip install pandas numpy

# Generate synthetic logs with realistic problems injected, then clean them
python data_pipeline.py --demo

# Or clean your own CSVs that use the same columns
python data_pipeline.py --edits edits.csv --llm llm.csv --out output
```

## Example QC result (from `--demo`)

From 312 raw edit rows and 48 raw LLM rows, the pipeline:

- removed 5 duplicate edit rows, 3 with a missing session ID, 2 with corrupt timestamps, and 29 system heartbeats
- flagged 3 impossible render times without deleting the events
- collapsed 2 retried LLM requests and removed 2 duplicate request rows
- flagged 2 requests with clock skew (received before sent) and left their latency blank

## What I'd add next

- Validate against a real logger's schema once one is available
- Link each LLM request to the edit event that triggered it
- Unit tests for each cleaning rule

## About

Built by Siddharth Datla, a sophomore at Coppell High School.
