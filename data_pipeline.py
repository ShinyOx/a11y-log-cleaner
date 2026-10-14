"""
a11y-log-cleaner: clean and summarize interaction logs from an
AI-assisted 3D modeling tool for blind and low-vision (BLV) users.

IMPORTANT: every dataset this script touches is SYNTHETIC. I have no access
to any real study data. The log formats below are my own guess at what a
tool like A11yShape (OpenSCAD + an LLM assistant) might record, built so I
could practice research-grade cleaning. Column names would need to be
adapted to any real logger.

Two logs are handled:
  1. OpenSCAD edit log  - one row per user or system event in the editor
  2. LLM request log    - one row per request sent to the AI assistant

Design rules (what makes this "research-grade" rather than a quick script):
  * Never invent data. Bad values are set to missing and FLAGGED, not
    replaced with a median or guess.
  * Never silently drop rows. Every removed row is written to
    excluded_rows.csv with the reason, and counted in qc_report.json.
  * Input from the user is keyboard and screen-reader driven (BLV users),
    so those are treated as real human input. Only system heartbeats are
    excluded from user-activity counts.
  * Raw files are never overwritten. Outputs go to a separate folder.

Usage:
  python data_pipeline.py --demo
      Generate synthetic logs in data/synthetic/ and clean them into output/.

  python data_pipeline.py --edits EDITS.csv --llm LLM.csv --out OUTDIR
      Clean your own CSVs that follow the same column layout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EDIT_COLUMNS = [
    "session_id", "participant_id", "timestamp", "event_type",
    "input_modality", "chars_changed", "render_ms",
]
LLM_COLUMNS = [
    "request_id", "session_id", "sent_at", "received_at", "request_type",
    "model", "prompt_tokens", "completion_tokens", "status",
]
USER_EVENTS = {
    "code_edit", "render", "render_error", "undo",
    "hierarchy_select", "description_request", "version_save",
}
SYSTEM_EVENTS = {"heartbeat"}
FAILED_STATUSES = {"timeout", "rate_limited", "server_error"}


# --------------------------------------------------------------------------
# 1. Synthetic data generator (clearly separate from the cleaning code)
# --------------------------------------------------------------------------
def generate_synthetic_logs(out_dir: Path, n_sessions: int = 6, seed: int = 42):
    """Write two SYNTHETIC raw logs with realistic problems injected."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2026-01-15 10:00:00")

    edit_rows, llm_rows = [], []
    req_counter = 0
    for s in range(1, n_sessions + 1):
        session_id = f"S{s:02d}"
        participant_id = f"P{(s - 1) // 2 + 1}"  # 2 sessions per participant
        t = start + pd.Timedelta(days=s)

        for _ in range(int(rng.integers(40, 70))):
            t += pd.Timedelta(seconds=float(rng.exponential(25)))
            event = rng.choice(
                ["code_edit", "render", "render_error", "undo",
                 "hierarchy_select", "description_request", "version_save",
                 "heartbeat"],
                p=[0.38, 0.17, 0.05, 0.06, 0.12, 0.08, 0.04, 0.10],
            )
            modality = ("system" if event == "heartbeat"
                        else rng.choice(["keyboard", "screen_reader"], p=[0.6, 0.4]))
            chars = int(rng.integers(1, 120)) if event == "code_edit" else 0
            render_ms = (float(rng.normal(900, 250))
                         if event in {"render", "render_error"} else np.nan)
            edit_rows.append([session_id, participant_id, t.isoformat(), event,
                              modality, chars, render_ms])

            # Description requests and some edits trigger an LLM call.
            if event == "description_request" or (event == "code_edit" and rng.random() < 0.15):
                req_counter += 1
                sent = t + pd.Timedelta(milliseconds=50)
                latency = pd.Timedelta(milliseconds=float(rng.gamma(4, 900)))
                status = rng.choice(["ok", "timeout", "rate_limited", "server_error"],
                                    p=[0.88, 0.05, 0.04, 0.03])
                llm_rows.append([
                    f"R{req_counter:04d}", session_id, sent.isoformat(),
                    (sent + latency).isoformat(),
                    "describe_model" if event == "description_request"
                    else rng.choice(["debug_code", "verify_change"]),
                    "gpt-4o", int(rng.integers(800, 4000)),
                    int(rng.integers(50, 600)) if status == "ok" else 0, status,
                ])

    edits = pd.DataFrame(edit_rows, columns=EDIT_COLUMNS)
    llm = pd.DataFrame(llm_rows, columns=LLM_COLUMNS)

    # ---- Inject the kinds of problems real loggers produce ----
    edits = pd.concat([edits, edits.sample(5, random_state=seed)])        # duplicate rows
    edits.loc[edits.sample(3, random_state=1).index, "session_id"] = np.nan  # lost session id
    edits.loc[edits.sample(2, random_state=2).index, "timestamp"] = "not-a-time"  # corrupt time
    render_idx = edits[edits["event_type"] == "render"].sample(3, random_state=3).index
    edits.loc[render_idx, "render_ms"] = -1.0                             # impossible duration

    retries = llm[llm["status"] != "ok"].head(2).copy()                  # retried requests
    retries["status"] = "ok"
    retries["completion_tokens"] = 300
    for col in ["sent_at", "received_at"]:                                # retry is 5 s later
        retries[col] = (pd.to_datetime(retries[col]) + pd.Timedelta(seconds=5)).map(
            lambda x: x.isoformat())
    llm = pd.concat([llm, retries, llm.sample(2, random_state=seed)])     # + exact dupes
    bad = llm.sample(2, random_state=4).index
    llm.loc[bad, "received_at"] = llm.loc[bad, "sent_at"]                 # clock skew below
    llm.loc[bad, "sent_at"] = (pd.to_datetime(llm.loc[bad, "sent_at"])
                               + pd.Timedelta(seconds=2)).map(lambda x: x.isoformat())

    edits_path = out_dir / "synthetic_openscad_edit_log.csv"
    llm_path = out_dir / "synthetic_llm_request_log.csv"
    edits.to_csv(edits_path, index=False)
    llm.to_csv(llm_path, index=False)
    print(f"Wrote SYNTHETIC raw logs:\n  {edits_path}\n  {llm_path}")
    return edits_path, llm_path


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def _exclude(df, mask, reason, source, excluded):
    """Move rows matching mask into the excluded list with a reason."""
    if mask.any():
        dropped = df[mask].copy()
        dropped["source_log"] = source
        dropped["exclusion_reason"] = reason
        excluded.append(dropped)
    return df[~mask].copy()


def clean_edit_log(raw: pd.DataFrame, excluded: list, qc: dict) -> pd.DataFrame:
    df = raw.copy()
    qc["edit_rows_raw"] = len(df)

    dupes = df.duplicated(keep="first")
    df = _exclude(df, dupes, "exact_duplicate", "edit_log", excluded)

    df = _exclude(df, df["session_id"].isna(), "missing_session_id", "edit_log", excluded)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    df = _exclude(df, df["timestamp"].isna(), "unparseable_timestamp", "edit_log", excluded)

    # System heartbeats are not user actions; keep them out of activity counts.
    df = _exclude(df, df["event_type"].isin(SYSTEM_EVENTS), "system_event", "edit_log", excluded)

    unknown = ~df["event_type"].isin(USER_EVENTS)
    df = _exclude(df, unknown, "unknown_event_type", "edit_log", excluded)

    # Impossible durations: flag and blank, do NOT impute.
    df["qc_flag"] = ""
    neg = df["render_ms"] < 0
    df.loc[neg, "render_ms"] = np.nan
    df.loc[neg, "qc_flag"] = "invalid_render_ms"
    qc["edit_rows_flagged_invalid_render_ms"] = int(neg.sum())

    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    qc["edit_rows_clean"] = len(df)
    return df


def clean_llm_log(raw: pd.DataFrame, excluded: list, qc: dict) -> pd.DataFrame:
    df = raw.copy()
    qc["llm_rows_raw"] = len(df)

    dupes = df.duplicated(keep="first")
    df = _exclude(df, dupes, "exact_duplicate", "llm_log", excluded)

    df = _exclude(df, df["session_id"].isna(), "missing_session_id", "llm_log", excluded)

    df["sent_at"] = pd.to_datetime(df["sent_at"], format="ISO8601", errors="coerce")
    df["received_at"] = pd.to_datetime(df["received_at"], format="ISO8601", errors="coerce")
    df = _exclude(df, df["sent_at"].isna(), "unparseable_timestamp", "llm_log", excluded)

    # Retries share a request_id. Keep the final attempt, record how many tries.
    df = df.sort_values(["request_id", "sent_at"], kind="stable")
    df["n_attempts"] = df.groupby("request_id")["request_id"].transform("size")
    earlier_attempt = df.duplicated(subset="request_id", keep="last")
    df = _exclude(df, earlier_attempt, "superseded_by_retry", "llm_log", excluded)

    df["latency_ms"] = (df["received_at"] - df["sent_at"]).dt.total_seconds() * 1000
    df["qc_flag"] = ""
    skew = df["latency_ms"] < 0
    df.loc[skew, "latency_ms"] = np.nan
    df.loc[skew, "qc_flag"] = "negative_latency_clock_skew"
    qc["llm_rows_flagged_clock_skew"] = int(skew.sum())
    qc["llm_requests_that_needed_retry"] = int((df["n_attempts"] > 1).sum())

    df = df.sort_values(["session_id", "sent_at"]).reset_index(drop=True)
    qc["llm_rows_clean"] = len(df)
    return df


# --------------------------------------------------------------------------
# 3. Per-session summary (what a researcher would actually analyze)
# --------------------------------------------------------------------------
def summarize_sessions(edits: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    g = edits.groupby("session_id")
    summary = pd.DataFrame({
        "participant_id": g["participant_id"].first(),
        "session_minutes": (g["timestamp"].max() - g["timestamp"].min()).dt.total_seconds() / 60,
        "user_events": g.size(),
        "code_edits": g["event_type"].apply(lambda s: (s == "code_edit").sum()),
        "renders": g["event_type"].apply(lambda s: s.isin(["render", "render_error"]).sum()),
        "render_errors": g["event_type"].apply(lambda s: (s == "render_error").sum()),
        "description_requests": g["event_type"].apply(lambda s: (s == "description_request").sum()),
        "screen_reader_share": g["input_modality"].apply(lambda s: (s == "screen_reader").mean()),
        "median_render_ms": g["render_ms"].median(),
    })
    summary["render_error_rate"] = summary["render_errors"] / summary["renders"].replace(0, np.nan)

    lg = llm.groupby("session_id")
    llm_summary = pd.DataFrame({
        "llm_requests": lg.size(),
        "llm_failure_rate": lg["status"].apply(lambda s: s.isin(FAILED_STATUSES).mean()),
        "median_llm_latency_ms": lg["latency_ms"].median(),
    })
    out = summary.join(llm_summary, how="left")
    out["llm_requests"] = out["llm_requests"].fillna(0).astype(int)
    return out.round(3).reset_index()


# --------------------------------------------------------------------------
# 4. Runner
# --------------------------------------------------------------------------
def run(edits_path: Path, llm_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_edits = pd.read_csv(edits_path)
    raw_llm = pd.read_csv(llm_path)

    for name, df, cols in [("edit log", raw_edits, EDIT_COLUMNS),
                           ("LLM log", raw_llm, LLM_COLUMNS)]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    excluded, qc = [], {}
    edits = clean_edit_log(raw_edits, excluded, qc)
    llm = clean_llm_log(raw_llm, excluded, qc)
    summary = summarize_sessions(edits, llm)

    excluded_df = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    qc["excluded_by_reason"] = (
        excluded_df.groupby(["source_log", "exclusion_reason"]).size()
        .rename("rows").reset_index().to_dict(orient="records")
        if not excluded_df.empty else []
    )

    edits.to_csv(out_dir / "clean_edit_log.csv", index=False)
    llm.to_csv(out_dir / "clean_llm_log.csv", index=False)
    summary.to_csv(out_dir / "session_summary.csv", index=False)
    excluded_df.to_csv(out_dir / "excluded_rows.csv", index=False)
    (out_dir / "qc_report.json").write_text(json.dumps(qc, indent=2, default=str))

    print("\nQC report")
    print(json.dumps(qc, indent=2, default=str))
    print(f"\nSession summary ({len(summary)} sessions):")
    print(summary.to_string(index=False))
    print(f"\nAll outputs written to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true",
                        help="generate synthetic logs, then clean them")
    parser.add_argument("--edits", type=Path, help="raw OpenSCAD edit log CSV")
    parser.add_argument("--llm", type=Path, help="raw LLM request log CSV")
    parser.add_argument("--out", type=Path, default=Path("output"), help="output folder")
    args = parser.parse_args()

    if args.demo:
        edits_path, llm_path = generate_synthetic_logs(Path("data/synthetic"))
        run(edits_path, llm_path, args.out)
    elif args.edits and args.llm:
        run(args.edits, args.llm, args.out)
    else:
        parser.error("use --demo, or pass both --edits and --llm")


if __name__ == "__main__":
    main()
