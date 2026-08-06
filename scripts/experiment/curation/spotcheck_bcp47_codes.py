"""Supplementary spot-check: does protocol-level BCP 47 normalization change translation output?

For each row in ``language_codes_comprehensive.csv`` where ``bcp47_tag`` substantively
differs from ``language_code`` (i.e. excluding pure hyphen/case normalization), this
script queries Google Translate under BOTH forms and reports whether the service
returns identical, divergent, or asymmetric (one-form-fails) translations.

This is an *additive* experiment, not a rerun. The pipeline's 881 language-code rows remain
keyed off ``language_code`` — the community/registry identifier. ``bcp47_tag`` records
IANA's preferred form. If both forms produce the same translation, the normalization
question is moot. If they diverge, that's a paper-grade finding about the semantic
consequences of protocol-level identifier choice (e.g. does Google treat ``tl`` and
``fil`` as different languages even though IANA says they're the same?).

Three classes of rows get checked:

1. **IANA Preferred-Value rewrites** (``mo``→``ro``, ``cnr``→``sr-ME``, ``sh``→``sr-Latn``,
   ``tl``→``fil``). Both forms are usually well-formed; the question is whether
   Google's MT treats them identically.
2. **Grandfathered Wikimedia fallbacks** (``bat-smg``→``sgs``, ``be-x-old``→``be-tarask``,
   ``fiu-vro``→``vro``, ``map-bms``→``jv``, ``roa-rup``→``rup``, ``tokipona``→``tok``,
   ``zh-classical``→``lzh``). The grandfathered form often fails outright; the
   canonical form may or may not be supported.

Usage:
    python -m scripts.experiment.curation.spotcheck_bcp47_codes
    python -m scripts.experiment.curation.spotcheck_bcp47_codes --term "Digital Humanities"
    python -m scripts.experiment.curation.spotcheck_bcp47_codes \\
        --csv-out datasets/metadata_files/bcp47_spotcheck_results.csv

Committed results live at ``datasets/metadata_files/bcp47_spotcheck_results.csv``
and are read by notebook 01's BCP 47 validation cell.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import apikey
import pandas as pd
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account


CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "datasets" / "metadata_files" / "language_codes_comprehensive.csv"
)


def _gt_client() -> translate.Client:
    key_path = apikey.load("GOOGLE_TRANSLATE_CREDENTIALS")
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return translate.Client(credentials=creds)


def _translate(client, text: str, target: str) -> tuple[str | None, str | None]:
    """Return (translated_text, error_message). Exactly one is None."""
    try:
        result = client.translate(text, source_language="en", target_language=target)
        # Google returns HTML-entity-encoded text for some scripts; decode for diffing.
        translated = html.unescape(result.get("translatedText", ""))
        return translated, None
    except Exception as e:
        msg = str(e).splitlines()[0][:120]
        return None, f"{type(e).__name__}: {msg}"


def _classify(row: pd.Series) -> str:
    if row["bcp47_source"] == "grandfathered_fallback":
        return "grandfathered"
    return "iana_preferred_value"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--term", default="Digital Humanities",
                        help='Source English term to translate (default: "Digital Humanities").')
    parser.add_argument("--csv-out", default=None,
                        help="Optional CSV path to write the full results.")
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH, dtype=str, na_filter=False)

    # Only rows where bcp47_tag substantively differs from language_code
    # (not just hyphen/case normalization like nds_nl → nds-NL).
    differs_mask = (
        df["bcp47_tag"].str.lower()
        != df["language_code"].str.replace("_", "-", regex=False).str.lower()
    )
    differs = df.loc[differs_mask, ["language_code", "language_name", "bcp47_tag", "bcp47_source"]].copy()
    differs["kind"] = differs.apply(_classify, axis=1)

    if differs.empty:
        print("No rows where bcp47_tag substantively differs from language_code. Nothing to check.")
        return

    print(f"Spot-checking {len(differs)} rows where bcp47_tag substantively differs from language_code.")
    print(f"  IANA Preferred-Value rewrites: {(differs['kind'] == 'iana_preferred_value').sum()}")
    print(f"  Grandfathered Wikimedia fallbacks: {(differs['kind'] == 'grandfathered').sum()}")
    print(f"Source term: {args.term!r}\n")

    client = _gt_client()
    rows = []
    for _, r in differs.iterrows():
        code = r["language_code"]
        bcp  = r["bcp47_tag"]
        proj_tx, proj_err = _translate(client, args.term, code)
        bcp_tx,  bcp_err  = _translate(client, args.term, bcp)

        proj_ok = proj_err is None
        bcp_ok  = bcp_err  is None
        agree   = proj_ok and bcp_ok and proj_tx == bcp_tx
        rows.append({
            "language_code": code,
            "language_name": r["language_name"],
            "bcp47_tag":     bcp,
            "kind":          r["kind"],
            "bcp47_source":  r["bcp47_source"],
            "project_translation": proj_tx if proj_ok else "",
            "project_error":       proj_err or "",
            "bcp47_translation":   bcp_tx if bcp_ok else "",
            "bcp47_error":         bcp_err or "",
            "agree":         agree,
            "outcome":       (
                "agree" if agree
                else "diverge" if proj_ok and bcp_ok
                else "project_only" if proj_ok and not bcp_ok
                else "bcp47_only"   if bcp_ok and not proj_ok
                else "both_failed"
            ),
        })

    out = pd.DataFrame(rows)

    # Human-readable summary
    print(f"{'code':<14} {'language':<24} {'bcp47':<12} {'outcome':<14} {'project':<22} {'bcp47':<22}")
    print("-" * 112)
    for _, r in out.iterrows():
        proj = (r["project_translation"] or f"[{r['project_error'][:18]}]")[:22]
        bcp  = (r["bcp47_translation"]   or f"[{r['bcp47_error'][:18]}]")[:22]
        name = r["language_name"][:23]
        print(f"{r['language_code']:<14} {name:<24} {r['bcp47_tag']:<12} {r['outcome']:<14} {proj:<22} {bcp:<22}")

    print()
    print("Outcome distribution:")
    for outcome, n in out["outcome"].value_counts().items():
        print(f"  {outcome:<16} {n}")
    print()
    print("Interpretation:")
    print("  agree         — Google returns identical text; normalization is semantically a no-op")
    print("  diverge       — Both forms accepted but produce different translations (paper-grade finding)")
    print("  project_only  — language_code accepted, bcp47_tag rejected (the community identifier is what's actually deployed)")
    print("  bcp47_only    — bcp47_tag accepted, language_code rejected (Google standardized to the IANA form)")
    print("  both_failed   — Neither form supported (mostly grandfathered Wikimedia tags)")

    if args.csv_out:
        out.to_csv(args.csv_out, index=False)
        print(f"\nFull results written to {args.csv_out}")


if __name__ == "__main__":
    main()
