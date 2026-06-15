"""Refresh service_language_code_support.csv from Google Translate v3 APIs.

Snapshots the currently supported languages for Google's NMT and Translation
LLM endpoints by calling ``translate_v3.get_supported_languages``. Writes a
dated CSV and keeps a timestamped backup of the previous file so paper-time
analyses remain reproducible against their original snapshot.

Why the API instead of scraping cloud.google.com:
    The marketing/documentation page is HTML that changes layout; the v3 API
    returns a stable, structured list and survives Google's UI redesigns.

NMT is queried on ``global``. Translation LLM is region-pinned (currently
``us-central1``); if that endpoint refuses or returns nothing, the existing
LLM rows from the previous CSV are preserved with a ``snapshot_source`` of
``preserved_from_previous`` so coverage analyses do not silently lose data.

Usage:
    python -m scripts.experiment.curation.refresh_service_support
    python -m scripts.experiment.curation.refresh_service_support --dry-run
    python -m scripts.experiment.curation.refresh_service_support --no-backup
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

import apikey
import pandas as pd
from google.cloud import translate_v3
from google.oauth2 import service_account


CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "datasets" / "metadata_files" / "service_language_code_support.csv"
)

NMT_LOCATION = "global"
LLM_LOCATION = "us-central1"

NMT_MODEL_SUFFIX = "general/nmt"
LLM_MODEL_SUFFIX = "general/translation-llm"

COLUMNS = [
    "service", "service_code", "language_name",
    "support_level", "model", "notes",
    "snapshot_date", "snapshot_source",
]


def _load_credentials_and_project() -> tuple[service_account.Credentials, str]:
    key_path = apikey.load("GOOGLE_TRANSLATE_CREDENTIALS")
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    with open(key_path) as f:
        project_id = json.load(f)["project_id"]
    return creds, project_id


def _client_for(creds, location: str) -> translate_v3.TranslationServiceClient:
    # Regional locations (anything except 'global') require the regional endpoint.
    if location == "global":
        return translate_v3.TranslationServiceClient(credentials=creds)
    from google.api_core.client_options import ClientOptions
    return translate_v3.TranslationServiceClient(
        credentials=creds,
        client_options=ClientOptions(api_endpoint=f"translate-{location}.googleapis.com"),
    )


def _fetch_supported(client, project_id: str, location: str, model_suffix: str):
    parent = f"projects/{project_id}/locations/{location}"
    model = f"{parent}/models/{model_suffix}"
    response = client.get_supported_languages(
        parent=parent, model=model, display_language_code="en",
    )
    return [(lang.language_code, lang.display_name) for lang in response.languages]


def _build_rows(service: str, model_label: str, fetched, today: str, source: str):
    return [
        {
            "service": service,
            "service_code": code,
            "language_name": display_name,
            "support_level": "official",
            "model": model_label,
            "notes": "",
            "snapshot_date": today,
            "snapshot_source": source,
        }
        for code, display_name in sorted(fetched)
    ]


def _preserved_rows_from_previous(service: str) -> list[dict]:
    if not CSV_PATH.exists():
        return []
    existing = pd.read_csv(CSV_PATH, dtype=str, na_filter=False)
    keep = existing[existing["service"] == service]
    rows = []
    for _, row in keep.iterrows():
        d = {col: row.get(col, "") for col in COLUMNS}
        d["snapshot_source"] = d.get("snapshot_source") or "preserved_from_previous"
        rows.append(d)
    return rows


def _summarise_diff(old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> None:
    if old_df is None or old_df.empty:
        print(f"  (no previous CSV; new file has {len(new_df)} rows)")
        return
    for service, new_sdf in new_df.groupby("service"):
        old_sdf = old_df[old_df["service"] == service]
        old_codes = set(old_sdf["service_code"])
        new_codes = set(new_sdf["service_code"])
        added = sorted(new_codes - old_codes)
        removed = sorted(old_codes - new_codes)
        print(
            f"  {service}: {len(old_sdf)} → {len(new_sdf)} rows  "
            f"(+{len(added)} added, -{len(removed)} removed)"
        )
        if added:
            print(f"    added:   {', '.join(added)}")
        if removed:
            print(f"    removed: {', '.join(removed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip backing up the previous CSV before overwriting.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the diff but don't write the CSV.")
    args = parser.parse_args()

    creds, project_id = _load_credentials_and_project()
    today = datetime.date.today().isoformat()
    rows: list[dict] = []

    print(f"Fetching Google NMT supported languages from '{NMT_LOCATION}'...")
    nmt_client = _client_for(creds, NMT_LOCATION)
    nmt = _fetch_supported(nmt_client, project_id, NMT_LOCATION, NMT_MODEL_SUFFIX)
    print(f"  {len(nmt)} languages")
    rows.extend(_build_rows(
        "google_nmt", "nmt", nmt, today, "translate_v3.get_supported_languages",
    ))

    print(f"Fetching Google Translation LLM supported languages from '{LLM_LOCATION}'...")
    try:
        llm_client = _client_for(creds, LLM_LOCATION)
        llm = _fetch_supported(llm_client, project_id, LLM_LOCATION, LLM_MODEL_SUFFIX)
        if not llm:
            raise RuntimeError("API returned 0 languages")
        print(f"  {len(llm)} languages")
        rows.extend(_build_rows(
            "google_translation_llm", "translation_llm", llm, today,
            "translate_v3.get_supported_languages",
        ))
    except Exception as e:
        print(f"  API fetch failed: {e}")
        preserved = _preserved_rows_from_previous("google_translation_llm")
        print(f"  Preserving {len(preserved)} Translation LLM rows from previous snapshot.")
        rows.extend(preserved)

    new_df = pd.DataFrame(rows, columns=COLUMNS)

    old_df = pd.read_csv(CSV_PATH, dtype=str, na_filter=False) if CSV_PATH.exists() else None
    print("\nDiff vs previous snapshot:")
    _summarise_diff(old_df, new_df)

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return

    if CSV_PATH.exists() and not args.no_backup:
        backup_path = CSV_PATH.with_name(f"{CSV_PATH.stem}.backup_{today}{CSV_PATH.suffix}")
        if not backup_path.exists():
            shutil.copy2(CSV_PATH, backup_path)
            print(f"\nBackup: {backup_path.name}")

    new_df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(new_df)} rows → {CSV_PATH}")


if __name__ == "__main__":
    main()
