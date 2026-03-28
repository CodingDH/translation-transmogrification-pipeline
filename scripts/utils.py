"""
Standalone utility functions for the translation pipeline.

These are extracted from the parent CodingDH utils.py so this repo is
self-contained and can be installed independently.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from rich.console import Console

console = Console()


def read_csv_file(file_path: str, error_bad_lines: bool = True) -> pd.DataFrame:
    """
    Read a CSV file with UTF-8 encoding, falling back to latin-1 on failure.

    Parameters
    ----------
    file_path : str
    error_bad_lines : bool
        Passed to pandas; set False to skip malformed rows.

    Returns
    -------
    pd.DataFrame
    """
    try:
        return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip' if not error_bad_lines else 'error')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip' if not error_bad_lines else 'error')


def log_error_to_file(
    error_file_path: str,
    additional_data: dict,
    status_code: int,
    error_url: str,
) -> None:
    """
    Append a structured error record to a CSV error log.

    Parameters
    ----------
    error_file_path : str
    additional_data : dict
        Extra columns to record (e.g. term_source, language_code).
    status_code : int
    error_url : str
        A description of the failing call.
    """
    error_df = pd.DataFrame([{
        "error_date": datetime.now().strftime("%Y-%m-%d"),
        "error_url": error_url,
        "status_code": status_code,
    }])
    error_df = pd.concat([error_df, pd.DataFrame([additional_data])], axis=1)

    if os.path.exists(error_file_path):
        error_df.to_csv(error_file_path, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
        error_df.to_csv(error_file_path, index=False)


def clean_write_error_file(
    error_file_path: str,
    drop_fields: Optional[List[str]] = None,
) -> None:
    """
    Deduplicate an error log file in place.

    Parameters
    ----------
    error_file_path : str
    drop_fields : list of str, optional
        Column subset used for deduplication.
    """
    if os.path.exists(error_file_path):
        error_df = read_csv_file(error_file_path)
        if 'error_date' in error_df.columns:
            error_df['error_date'] = pd.to_datetime(error_df['error_date'])
            error_df = error_df.sort_values('error_date').drop_duplicates(
                subset=drop_fields, keep='last'
            )
        elif drop_fields:
            error_df = error_df.drop_duplicates(subset=drop_fields, keep='last')
        error_df.to_csv(error_file_path, index=False)
    else:
        console.print('No error file to clean', style='bold blue')


def get_data_directory_path() -> str:
    """
    Return the data directory path from the DATA_DIR environment variable,
    falling back to a ``data/`` sibling of the repo root.

    Set ``DATA_DIR=/path/to/your/data`` in your environment or .env file.
    """
    env_path = os.environ.get('DATA_DIR')
    if env_path:
        return env_path
    # Default: data/ relative to repo root (two levels up from this file)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, 'data')