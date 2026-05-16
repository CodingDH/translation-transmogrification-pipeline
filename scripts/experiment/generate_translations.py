# Standard library imports
import csv
import html
import json
import os
import warnings
from typing import List, Tuple, Callable, Optional
import ast
import inspect
import datetime

# Related third-party imports
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.utils import get_data_directory_path, read_csv_file, log_error_to_file, clean_write_error_file
warnings.filterwarnings('ignore')

MAX_CONSECUTIVE_TIMEOUTS = 5  # Stop after 5 consecutive timeouts


# Import from new modules
from data_processing import (
    extract_dictionaries_from_string, extract_ollama_translated_term,
)
from generate_language_codes import load_language_codes
from translation_services import (
    get_gt_translation, get_enmt_translation, get_openai_translation,
    get_claude_translation, get_gemini_translation, get_lingvanex_translation,
    get_ollama_translation, get_deepseek_translation,
    check_if_wikipedia_page_exists,
    OLLAMA_LLAMA_MODEL, OLLAMA_GEMMA_MODEL, OLLAMA_QWEN_MODEL, OLLAMA_MISTRAL_MODEL,
    OllamaUnresponsiveError,
    console
)


def post_process_ollama(df: pd.DataFrame, col_prefix: str = 'ollama') -> pd.DataFrame:
	"""
	Post-process Ollama translations to extract dictionaries and select best translations.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing translated terms.
	col_prefix : str
		Column prefix for the model (e.g. 'llama', 'gemma', 'qwen', 'mistral').
	"""
	content_col = f'{col_prefix}_content'
	extracted_col = f'{col_prefix}_extracted_dictionaries'
	term_col = f'{col_prefix}_translated_term'
	translation_col = f'{col_prefix}_translation'

	if content_col not in df.columns:
		df[content_col] = None
	tqdm.pandas(desc="Extracting dictionaries")
	df[extracted_col] = df[content_col].progress_apply(extract_dictionaries_from_string)

	tqdm.pandas(desc="Selecting translated terms")
	df = df.progress_apply(lambda row: extract_ollama_translated_term(row, col_prefix), axis=1)

	# Treat empty strings the same as NaN throughout
	df[term_col] = df[term_col].replace('', None)
	df[translation_col] = df[translation_col].replace('', None)
	# Backfill: if the primary term is missing but the extraction column has a value, use it
	df.loc[df[term_col].isna() & df[translation_col].notna(), term_col] = df[translation_col]

	return df

TRANSIENT_STATUS_CODES = {500, 408}

def mark_errored_terms(df: pd.DataFrame, error_file: str, service: str, variant: str = None) -> pd.DataFrame:
	"""
	Mark terms and languages that have previously encountered errors when translating in the DataFrame.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame containing all terms and languages.
	error_file : str
		A path to the error file for logging errors.
	service : str
		A string indicating the translation service (e.g., 'gt', 'enmt', 'openai', 'ollama').
	variant : str, optional
		The current prompt variant (e.g. 'minimal', 'judge'). When provided and the error
		log contains a 'variant' column, only errors for this specific variant are excluded.
		If the error log has no 'variant' column (legacy logs), all errors apply regardless
		of variant (backward-compatible behaviour).

	Returns
	--------
	pd.DataFrame
		A DataFrame with errored terms marked.
	"""
	df = df.copy()  # prevent in-place mutation of the caller's DataFrame
	error_col = f'exclude_{service.replace(" ", "_").lower()}'
	if os.path.exists(error_file):
		error_df = read_csv_file(error_file, error_bad_lines=False)
		# Skip transient errors (500 server errors, 408 timeouts) — these should be retried,
		# not permanently excluded. Only exclude deterministic failures like 400 (unsupported
		# language) and 404 (model/resource not found).
		if 'status_code' in error_df.columns:
			error_df = error_df[~error_df['status_code'].isin(TRANSIENT_STATUS_CODES)]
		# When the error log records variants, restrict exclusions to the current variant.
		# Legacy logs without a variant column exclude unconditionally (old behaviour).
		if variant is not None and 'variant' in error_df.columns:
			error_df = error_df[error_df['variant'].astype(str) == str(variant)]
		# Normalize term_source column — entries may be stored as a plain string
		# (e.g. "Digital Humanities") or as a Python list-string (e.g. "['Digital Humanities']").
		expanded_rows = []
		for _, row in error_df.iterrows():
			raw = row['term_source']
			if not isinstance(raw, str):
				continue
			# Try to parse as a Python list first; fall back to treating as a plain string
			try:
				parsed = ast.literal_eval(raw)
				terms = parsed if isinstance(parsed, list) else [parsed]
			except (ValueError, SyntaxError):
				terms = [raw]
			for term in terms:
				expanded_rows.append({
					'language_code': row.get('language_code'),
					'term_source': term
				})

		# Create a DataFrame of (language_code, language_name, term_source) triplets to exclude
		if expanded_rows:
			expanded_error_df = pd.DataFrame(expanded_rows)
			exclude_set = set(expanded_error_df.itertuples(index=False, name=None))

			df[error_col] = df.apply(
				lambda row: (row.get('language_code'), row.get('term_source')) in exclude_set,
				axis=1
			)
		else:
			df[error_col] = False
	else:
		df[error_col] = False

	return df

def read_existing_translations_for_service(
	target_terms: List[str],
	data_directory_path: str,
	translation_column: str,
	subfolder: str = "historic_translations_data",
	file_suffix: str = "_translated_terms.csv",
	timestamp_column: str = "coding_dh_date",
) -> pd.DataFrame:
	"""
	Reads all existing translation files for a specific service and returns the most recent entries per language and term_source.

	Parameters
	----------
	target_terms : List[str]
		A list of terms whose translations to load.
	data_directory_path : str
		Base path to metadata_files/translated_terms.
	translation_column : str
		The column name where the translations are stored (e.g., 'openai_translated_term').
	subfolder : str, optional
		The subdirectory under each term's folder where the service translation files live.
	file_suffix : str, optional
		The suffix used in the translation file name.
	timestamp_column : str, optional
		The column name used for keeping the most recent translation.

	Returns
	-------
	pd.DataFrame
		Combined and deduplicated DataFrame of the most recent translations for the service.
	"""
	dfs = []

	for term_source in target_terms:
		term_slug = term_source.lower().replace(" ", "_")
		service_file = os.path.join(
			data_directory_path,
			"metadata_files",
			"translated_terms",
			term_slug,
			subfolder,
			f"{translation_column}{file_suffix}"
		)

		if os.path.exists(service_file):
			df = read_csv_file(service_file)
			df["term_source"] = term_source
			if timestamp_column in df.columns:
				df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
				df = df.sort_values(timestamp_column, ascending=False).drop_duplicates(
					subset=["language_code", "term_source"], keep="first"
				)
			dfs.append(df)

	if dfs:
		return pd.concat(dfs).reset_index(drop=True)
	else:
		return pd.DataFrame()

def get_translation_column(translation_columns: list, required_suffix: str = '_translated_term') -> Optional[str]:
	"""
	Returns the first translation column matching the required suffix, or None if not found.

	Parameters
	----------
	translation_columns : list
		A list of translation column names.
	required_suffix : str, optional
		The suffix to look for in the translation columns (default is '_translated_term').

	Returns
	-------
	str or None
		The first translation column that matches the required suffix, or None if no match is found.
	"""
	for col in translation_columns:
		if required_suffix in col:
			return col
	return None

def load_existing_translations_by_term_sources(
	base_path: str,
	translate_file_name: str,
	term_sources: List[str],
	translation_columns: List[str],
	timestamp_column: str = "coding_dh_date",
	use_cached_translations: bool = True,
	subfolder: str = "direct_services",
) -> pd.DataFrame:
	"""
	Loads existing translations from CSV files for specified term sources.
	Parameters
	----------
	base_path : str
		Base path where the term source directories are located.
	translate_file_name : str
		Name of the translation file to read (e.g., 'translated_terms.csv').
	term_sources : List[str]
		List of term sources to load translations for.
	translation_columns : List[str]
		List of translation columns to include in the final DataFrame.
	timestamp_column : str, optional
		Name of the timestamp column to use for deduplication (default is 'coding_dh_date').
	use_cached_translations : bool, optional
		Whether to use cached translations (default is True).
	subfolder : str, optional
		Subdirectory under each term slug where service files live.
		Use 'direct_services' for GT/EasyNMT/Lingvanex/Wikipedia/Ollama-first-pass,
		or 'prompt_services' for OpenAI/Claude/Gemini/second-Ollama.

	Returns
	-------
	pd.DataFrame
		A DataFrame containing the most recent translations for each term source, deduplicated by language and term source.
	"""
	all_dfs = []
	for term in term_sources:
		file_path = os.path.join(base_path, term.lower().replace(" ", "_"), subfolder, translate_file_name)
		if os.path.exists(file_path) and use_cached_translations:
			df = pd.read_csv(file_path, on_bad_lines='warn', converters={'language_code': str})
			df["term_source"] = term
			if timestamp_column in df.columns:
				df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
				df = df.sort_values(timestamp_column, ascending=False).drop_duplicates(
					subset=["language_code", "term_source"], keep="first"
				)
				translation_col = next((c for c in translation_columns if c.endswith('_translated_term')), None)
				filled = df[translation_col].notna().sum() if translation_col and translation_col in df.columns else '?'
				console.print(f"Loaded {len(df)} rows for {term} ({filled} with translations) from {file_path}", style="bold green")
			all_dfs.append(df)

	return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(columns=["language_code", "language_name", "term_source"] + translation_columns)

def process_individual_terms(
	translate_file_path: str,
	translate_file_name: str,
	final_df: pd.DataFrame,
	translation_columns: List[str],
	translation_function: Callable[[pd.Series], pd.Series],
	service_name: str,
	should_use_cached_translations: bool,
	should_override_wikipedia: bool,
	post_process_function: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
	exclude_errors_file: Optional[str] = None,
	variant: Optional[str] = None,
) -> pd.DataFrame:
	"""
	Translate terms using a service that operates row-by-row (one language per call).

	Handles the full translation lifecycle for individual-row services (OpenAI, Claude, Gemini, Ollama): loads cached translations, identifies what is missing, translates missing rows, optionally post-processes results, and persists each term's translations to its own historic_translations_data directory.

	Unlike process_grouped_terms, the translation_function is applied one row at a time rather than once per term across all languages.

	Parameters
	----------
	translate_file_path : str
		Root directory where per-term subdirectories are stored.
	translate_file_name : str
		Filename of the translation CSV (e.g., 'openai_translations.csv').
	final_df : pd.DataFrame
		DataFrame containing the terms and languages to translate.
	translation_columns : List[str]
		Column names produced by the translation service (e.g., ['openai_translated_term',
		'openai_translation_rationale']).
	translation_function : Callable[[pd.Series], pd.Series]
		Function applied to each row to produce a translation. Ollama returns a
		(row, consecutive_timeouts) tuple; all other services return a plain Series.
	service_name : str
		Human-readable name of the service (e.g., 'OpenAI'), used for logging and
		deriving the exclude column name.
	should_use_cached_translations : bool
		If True, load existing translations from disk before translating.
	should_override_wikipedia : bool
		If False, skip rows that already have a Wikipedia translation.
	post_process_function : Callable[[pd.DataFrame], pd.DataFrame], optional
		Optional function applied to the translated DataFrame before saving (e.g., to
		extract structured fields from raw Ollama output).
	exclude_errors_file : str, optional
		Path to an error log file. Rows matching prior errors are flagged and skipped
		for retranslation but preserved in the output.

	Returns
	-------
	pd.DataFrame
		Combined DataFrame of existing, newly translated, and excluded rows, deduplicated
		by language, term, and translation column. Saves each term's results to
		{translate_file_path}/{term}/historic_translations_data/{translate_file_name}.
	"""
	console.print(f"Processing terms for {service_name}", style="bold green")
	console.print(f"Our dataframe has {len(final_df)} rows", style="bold green")

	translation_column = get_translation_column(translation_columns)
	if not translation_column:
		console.print("No translation column found in translation_columns", style="bold red")
		return pd.DataFrame()

	exclude_col = f'exclude_{service_name.lower().replace(" ", "_")}'

	# Exclude Wikipedia translations unless override is set
	if 'wikipedia_translated_term' in final_df.columns and not should_override_wikipedia:
		final_df = final_df[final_df.wikipedia_translated_term.isna()]

	# Mark errored terms (adds exclude_{service} flag)
	if exclude_errors_file:
		final_df = mark_errored_terms(final_df, exclude_errors_file, service=service_name.lower(), variant=variant)
		if exclude_col in final_df.columns:
			console.print(f"Excluded terms: {len(final_df[final_df[exclude_col]])}", style="bold red")
			console.print(f"Working terms: {len(final_df[~final_df[exclude_col]])}", style="bold green")
			excluded_terms_df = final_df[final_df[exclude_col]].copy()
	else:
		excluded_terms_df = pd.DataFrame()

	# Load existing translations per term
	existing_translations = load_existing_translations_by_term_sources(
		base_path=translate_file_path,
		translate_file_name=translate_file_name,
		term_sources=final_df.term_source.unique().tolist(),
		translation_columns=translation_columns,
		use_cached_translations=should_use_cached_translations,
		subfolder="prompt_services",
	)

	if len(existing_translations) > 0:
		_tcol = translation_column if translation_column in existing_translations.columns else None
		_filled = existing_translations[_tcol].notna().sum() if _tcol else '?'
		console.print(f"Loaded {len(existing_translations)} existing rows ({_filled} with translations)", style="bold green")
		merge_cols = ['language_code', 'term_source']
		# Exclude coding_dh_date here — it lives on existing_translations for the final concat,
		# not on the working final_df. Including it causes _x/_y suffix collisions when
		# final_df already has a coding_dh_date column from earlier service merges.
		cols_to_merge = list(set(merge_cols + translation_columns) & set(existing_translations.columns))
		# _merge_translations drops any overlapping non-key columns from the incoming df,
		# preventing _x/_y suffixes on columns that final_df already carries.
		final_df = _merge_translations(final_df, existing_translations[cols_to_merge], merge_cols, how='left')

	# Identify missing terms
	if translation_column in final_df.columns:
		missing_terms_df = final_df[final_df[translation_column].isna()]
	else:
		missing_terms_df = final_df

	if exclude_col in missing_terms_df.columns:
		missing_terms_df = missing_terms_df[~missing_terms_df[exclude_col]]

	# Translate missing terms row-by-row, appending each result to disk immediately.
	# This means a timeout or crash can be resumed on the next run — the caching logic
	# in load_existing_translations_by_term_sources will skip any rows already on disk.
	if not missing_terms_df.empty:
		consecutive_timeouts = 0
		_fn_accepts_ct = len(inspect.signature(translation_function).parameters) > 1
		translated_rows = []

		try:
			for _, row in tqdm(missing_terms_df.iterrows(), total=len(missing_terms_df), desc=f"Translating missing terms using {service_name}"):
				if _fn_accepts_ct:
					result = translation_function(row, consecutive_timeouts)
				else:
					result = translation_function(row)

				if isinstance(result, tuple):
					translated_row, consecutive_timeouts = result
				else:
					translated_row = result

				# Apply post-processing per row (e.g. Ollama dict extraction) so the
				# translation column is populated before the incremental save.
				if post_process_function:
					row_df = pd.DataFrame([translated_row])
					row_df = post_process_function(row_df)
					translated_row = row_df.iloc[0]

				# Stamp the translation date before saving
				translated_row['coding_dh_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

				# Append translated row to the per-term file immediately
				term_slug = str(translated_row.get('term_source', '')).lower().replace(' ', '_')
				save_path = os.path.join(translate_file_path, term_slug, 'prompt_services', translate_file_name)
				os.makedirs(os.path.dirname(save_path), exist_ok=True)
				save_df = pd.DataFrame([translated_row])
				# Strip pipeline-internal housekeeping columns before writing to disk:
				# exclude_* from mark_errored_terms, merge artifacts.
				_cols_to_drop = (
					[c for c in save_df.columns if c.startswith('exclude_')] +
					[c for c in save_df.columns if c in ('coding_dh_date_x', 'coding_dh_date_y')]
				)
				save_df = save_df.drop(columns=_cols_to_drop, errors='ignore')
				# If the file already exists, align to its header column order so that
				# rows from different runs never land in the wrong columns.
				if os.path.exists(save_path):
					existing_header = pd.read_csv(save_path, nrows=0).columns.tolist()
					new_cols = [c for c in save_df.columns if c not in existing_header]
					for c in new_cols:
						# add any brand-new columns to the file before appending
						existing_full = pd.read_csv(save_path, on_bad_lines='warn', converters={'language_code': str})
						existing_full[c] = None
						existing_full.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)
						existing_header.append(c)
					save_df = save_df.reindex(columns=existing_header)
				save_df.to_csv(
					save_path, mode='a', header=not os.path.exists(save_path), index=False,
					quoting=csv.QUOTE_ALL,
				)

				translated_rows.append(translated_row)

		except Exception as e:
			error_str = str(e)
			error_log = exclude_errors_file or os.path.join(translate_file_path, "pipeline_errors.csv")
			log_error_to_file(
				error_file_path=error_log,
				additional_data={"term_source": service_name, "language_code": None},
				status_code=500,
				error_url=f"{service_name} translation failed: {error_str[:200]}",
			)
			console.print(
				f"\n✓ {len(translated_rows)} rows already saved to disk — rerun to continue from where it left off.",
				style="bold yellow"
			)
			raise

		missing_terms_df = pd.DataFrame(translated_rows) if translated_rows else missing_terms_df.iloc[0:0]

		# Filter excluded terms not already in existing translations
		if not excluded_terms_df.empty and {'language_code', 'language_name', 'term_source'}.issubset(excluded_terms_df.columns):
			excluded_terms_df = excluded_terms_df[
				~excluded_terms_df.set_index(['language_code', 'term_source']).index.isin(
					existing_translations.set_index(['language_code', 'term_source']).index
				)
			]
		try:
			def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
				"""Drop any duplicate columns, keeping the first occurrence."""
				return df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

			finalized_translations = pd.concat(
				[
					_dedup_cols(existing_translations),
					_dedup_cols(missing_terms_df),
					_dedup_cols(excluded_terms_df),
				],
				ignore_index=True
			)
		except Exception as concat_err:
			service_slug = service_name.lower().replace(" ", "_")
			error_log = exclude_errors_file or os.path.join(translate_file_path, "pipeline_errors.csv")
			log_error_to_file(
				error_file_path=error_log,
				additional_data={"term_source": service_name, "language_code": None},
				status_code=500,
				error_url=f"{service_name} DataFrame concat failed: {str(concat_err)[:200]}",
			)
			existing_translations.to_csv(os.path.join(translate_file_path, f"{service_slug}_debug_existing.csv"), index=False)
			missing_terms_df.to_csv(os.path.join(translate_file_path, f"{service_slug}_debug_missing.csv"), index=False)
			excluded_terms_df.to_csv(os.path.join(translate_file_path, f"{service_slug}_debug_excluded.csv"), index=False)
			raise ValueError(f"Error concatenating DataFrames for {service_name}. Debug files saved to {translate_file_path}.")

		console.print(f"Finalized translations have {len(finalized_translations)} rows", style="bold green")

		if 'coding_dh_date' in finalized_translations.columns:
			finalized_translations['coding_dh_date'] = pd.to_datetime(finalized_translations['coding_dh_date'], errors='coerce')
			finalized_translations = finalized_translations.sort_values('coding_dh_date', ascending=False)
		finalized_translations = finalized_translations.drop_duplicates(
			subset=['language_code', 'term_source'], keep='first'
		)

		console.print(f"After dropping duplicates: {len(finalized_translations)} rows", style="bold green")

		# Save each term's translations separately
		for term_source in finalized_translations.term_source.unique():
			single_df = finalized_translations[finalized_translations.term_source == term_source]
			single_path = os.path.join(
				translate_file_path,
				term_source.lower().replace(" ", "_"),
				"prompt_services",
				translate_file_name
			)
			os.makedirs(os.path.dirname(single_path), exist_ok=True)
			single_df.to_csv(single_path, index=False, quoting=csv.QUOTE_ALL)

	else:
		finalized_translations = existing_translations

	# Drop all exclude_* columns — only the current service's is added by mark_errored_terms,
	# but earlier services may have left their own on final_df which flow into finalized_translations.
	exclude_cols = [c for c in finalized_translations.columns if c.startswith('exclude_')]
	finalized_translations = finalized_translations.drop(columns=exclude_cols, errors='ignore')

	return finalized_translations

def process_grouped_terms(
	translate_file_path: str,
	translate_file_name: str,
	translation_columns: List[str], 
	final_df: pd.DataFrame, 
	translation_function: Callable[[pd.Series], pd.Series], 
	service_name: str,
	should_use_cached_translations: bool,
	should_override_wikipedia: bool,
	exclude_errors_file: Optional[str] = None
) -> pd.DataFrame:
	"""
	Process terms using a specified translation service, only translating terms missing translations for any language. Saves the final translations persistently to avoid redundant API calls.

	Parameters
	----------
	translate_file_path : str
		The path to the file storing translations.
	translate_file_name : str
		The name of the file where translations are stored (e.g., 'translated_terms.csv').
	translation_columns : List[str]
		A list of translation column names to include in the final DataFrame (e.g., ['gt_translated_term', 'enmt_translated_term']).
	final_df : pd.DataFrame
		The full DataFrame containing all language codes.
	translation_column : str
		The column name where translations are stored (e.g., 'gt_translated_term', 'enmt_translated_term').
	translation_function : Callable[[pd.Series], pd.Series]
		The function that performs translations for each row.
	service_name : str
		A friendly name for the translation service (for logging purposes).
	should_use_cached_translations : bool
		A boolean indicating whether to use cached translations. If True, files will not load and all data will be rerun.
	should_override_wikipedia: bool
		A boolean indicating whether to override Wikipedia translations.
	exclude_errors_file : Optional[str]
		A path to the error file for logging errors.

	Returns
	-------
	pd.DataFrame
		A DataFrame with all translated terms.
	"""
	console.print(f"Processing terms for {service_name}", style="bold green")
	console.print(f"Our dataframe has {len(final_df)} rows", style="bold green")

	translation_column = get_translation_column(translation_columns)
	if not translation_column:
		console.print("No translation column found in translation_columns", style="bold red")
		return pd.DataFrame()
	# Exclude Wikipedia translations unless override is set
	if "wikipedia_translated_term" in final_df.columns and not should_override_wikipedia:
		final_df = final_df[final_df.wikipedia_translated_term.isna()]

	# Mark error rows
	exclude_col = f'exclude_{service_name.lower().replace(" ", "_")}'
	console.print(f"Excluding terms with {exclude_col} column with {exclude_errors_file}", style="bold red")
	if exclude_errors_file:
		final_df = mark_errored_terms(final_df, exclude_errors_file, service=service_name.lower())

	if exclude_col in final_df.columns:
		excluded_terms_df = final_df[final_df[exclude_col]].copy()
		console.print(f"Excluded terms: {len(excluded_terms_df)}", style="bold red")
		working_df = final_df[~final_df[exclude_col]].copy()
		console.print(f"Working terms: {len(working_df)}", style="bold green")
	else:
		excluded_terms_df = pd.DataFrame()
		working_df = final_df.copy()
	# Load existing translations per term
	existing_translations = load_existing_translations_by_term_sources(
		base_path=translate_file_path,
		translate_file_name=translate_file_name,
		term_sources=working_df.term_source.unique().tolist(),
		translation_columns=translation_columns,
		use_cached_translations=should_use_cached_translations,
		subfolder="direct_services",
	)

	if len(existing_translations) > 0:
		_tcol = translation_column if translation_column in existing_translations.columns else None
		_filled = existing_translations[_tcol].notna().sum() if _tcol else '?'
		console.print(f"Loaded {len(existing_translations)} existing rows ({_filled} with translations)", style="bold green")
		merge_cols = ['language_code', 'term_source']
		cols_to_merge = list(set(merge_cols + translation_columns) & set(existing_translations.columns))
		working_df = working_df.merge(existing_translations[cols_to_merge], on=merge_cols, how='left')

	# Identify missing translations
	subset_df = working_df[working_df.get(translation_column).isna()] if translation_column in working_df.columns else working_df
	subset_df = subset_df.reset_index(drop=True)
	# Group for translation
	grouped_terms = subset_df.groupby(
		['language_code', 'language_name']
	).agg({'term_source': lambda x: list(x)}).reset_index()

	if not grouped_terms.empty:
		tqdm.pandas(desc=f"Translating missing terms using {service_name}")
		grouped_terms = grouped_terms.progress_apply(translation_function, axis=1)

		grouped_terms[translation_column] = grouped_terms[translation_column].apply(
			lambda x: [t[1] for t in x] if isinstance(x, list) else x
		)

		# Explode into one row per term
		exploded_grouped_terms = grouped_terms.explode(['term_source', translation_column])
		exploded_grouped_terms['coding_dh_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

		# Filter excluded terms not already in existing translations
		if not excluded_terms_df.empty and {'language_code', 'language_name', 'term_source'}.issubset(excluded_terms_df.columns):
			excluded_terms_df = excluded_terms_df[
				~excluded_terms_df.set_index(['language_code', 'term_source']).index.isin(
					existing_translations.set_index(['language_code', 'term_source']).index
				)
			]

		def _dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
			"""Drop any duplicate columns, keeping the first occurrence."""
			return df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

		finalized_translations = pd.concat(
			[_dedup_cols(existing_translations), _dedup_cols(exploded_grouped_terms), _dedup_cols(excluded_terms_df)],
			ignore_index=True
		)
		console.print(f"Finalized translations have {len(finalized_translations)} rows", style="bold green")
		if 'coding_dh_date' in finalized_translations.columns:
			finalized_translations['coding_dh_date'] = pd.to_datetime(finalized_translations['coding_dh_date'], errors='coerce')
			finalized_translations = finalized_translations.sort_values('coding_dh_date', ascending=False)
		finalized_translations = finalized_translations.drop_duplicates(
			subset=['language_code', 'term_source'], keep='first'
		)

		console.print(f"Finalized translations after dropping duplicates have {len(finalized_translations)} rows", style="bold green")
		# Save each term's translations separately
		for term_source in finalized_translations.term_source.unique():
			single_df = finalized_translations[finalized_translations.term_source == term_source]
			single_path = os.path.join(
				translate_file_path,
				term_source.lower().replace(" ", "_"),
				"direct_services",
				translate_file_name
			)
			os.makedirs(os.path.dirname(single_path), exist_ok=True)
			single_df.to_csv(single_path, index=False, quoting=csv.QUOTE_ALL)
	else:
		finalized_translations = existing_translations

	# Drop the temporary column if it exists
	if exclude_col in finalized_translations.columns:
		finalized_translations = finalized_translations.drop(columns=[exclude_col])

	return finalized_translations


def _merge_translations(base_df: pd.DataFrame, new_df: pd.DataFrame, join_cols: list, how: str = 'outer') -> pd.DataFrame:
	"""
	Merge new translation columns onto base_df without producing _x/_y duplicate suffixes.

	Any column on new_df that already exists on base_df (other than the join keys) is
	dropped from new_df before merging, so the existing values on base_df are preserved
	and pandas never needs to disambiguate with suffixes.
	"""
	overlap = [c for c in new_df.columns if c in base_df.columns and c not in join_cols]
	return base_df.merge(new_df.drop(columns=overlap), on=join_cols, how=how)

def get_variant_output_path(base_path: str, filename: str, variant: str = 'minimal') -> str:
	"""
	Get the output path for a file under the prompt_variants subdirectory.

	All variants write to the same prompt_variants/
	subfolder so no variant is treated as a baseline or given special status.

	Parameters
	----------
	base_path : str
		The base directory path for the term.
	filename : str
		The filename (e.g., 'initial_translated_terms.csv').
	variant : str
		The prompt variant name. Defaults to 'minimal'.

	Returns
	-------
	str
		{base_path}/prompt_variants/{variant}_{filename}
	"""
	variant_dir = os.path.join(base_path, 'prompt_variants')
	os.makedirs(variant_dir, exist_ok=True)
	return os.path.join(variant_dir, f"{variant}_{filename}")

def generate_initial_terms(target_terms: list, data_directory_path: str, process_dh: bool, use_gt_translate: bool, use_enmt_translate: bool, use_openai_translate: bool, use_claude_translate: bool, use_ollama_translate: bool, use_wikipedia: bool, override_wikipedia: bool, use_cached_translations: bool, return_all_data: bool = False, exclude_previous_errors: bool = False, use_gemini_translate: bool = False, use_lingvanex_translate: bool = False, use_deepseek_translate: bool = False, use_gemma_translate: bool = False, use_qwen_translate: bool = False, use_mistral_translate: bool = False, prompt_variant: str = 'minimal', term_contexts: dict = None) -> pd.DataFrame:
	"""
	Generate a dataframe with translated terms. This function assumes you want to at the very least translate Digital Humanities.

	Parameters
	----------
	target_terms : list
		A list of terms to translate in all ISO 639-1 languages.
	data_directory_path : str
		A path to the directory with the datasets.
	process_dh : bool
		A boolean indicating whether Digital Humanities terms already exist.
	use_gt_translate : bool
		A boolean indicating whether to use Google Cloud Translate.
	use_enmt_translate : bool
		A boolean indicating whether to use EasyNMT.
	use_openai_translate : bool
		A boolean indicating whether to use OpenAI API.
	use_wikipedia : bool
		A boolean indicating whether to check for Wikipedia pages.
	override_wikipedia : bool
		A boolean indicating whether to override existing Wikipedia translations for terms.
	use_cached_translations: bool
		A boolean indicating whether to use cached existing translations. If False, existing files will be ignored and all data will be reprocessed.
	return_all_data: bool
		A boolean indicating whether to return the full dataset or subset to only to existing terms. Defaults to only existing terms.
	exclude_previous_errors: bool
		A boolean indicating whether to exclude previous errors from the dataset. Defaults to False.

	Returns
	-------
	pd.DataFrame
		A dataframe with translated terms.
	"""
	error_dir = os.path.join(data_directory_path, "error_logs")
	os.makedirs(error_dir, exist_ok=True)

	gt_error_file       = os.path.join(error_dir, "gt_translation_errors.csv")
	enmt_error_file     = os.path.join(error_dir, "enmt_translation_errors.csv")
	openai_error_file   = os.path.join(error_dir, "openai_translation_errors.csv")
	claude_error_file   = os.path.join(error_dir, "claude_translation_errors.csv")
	llama_error_file    = os.path.join(error_dir, "llama_translation_errors.csv")
	gemini_error_file   = os.path.join(error_dir, "gemini_translation_errors.csv")
	lingvanex_error_file = os.path.join(error_dir, "lingvanex_translation_errors.csv")
	deepseek_error_file = os.path.join(error_dir, "deepseek_translation_errors.csv")
	gemma_error_file    = os.path.join(error_dir, "gemma_translation_errors.csv")
	qwen_error_file     = os.path.join(error_dir, "qwen_translation_errors.csv")
	mistral_error_file  = os.path.join(error_dir, "mistral_translation_errors.csv")
	for error_file in [gt_error_file, enmt_error_file, openai_error_file, claude_error_file,
	                   llama_error_file, gemini_error_file, lingvanex_error_file,
	                   deepseek_error_file, gemma_error_file, qwen_error_file, mistral_error_file]:
		clean_write_error_file(error_file, drop_fields=["term_source", "language_code", "error_url", "variant"])
	# Make temp dir wherever running the code
	if not os.path.exists("temp"):
		os.makedirs("temp", exist_ok=True)
	# Load Wikimedia language list from the comprehensive language codes CSV.
	iso_languages = load_language_codes()[['language_code', 'English language name']].rename(
		columns={'English language name': 'language_name'}
	).reset_index(drop=True)
	console.print(f"Language list: {len(iso_languages)} Wikimedia languages from language_codes_comprehensive.csv", style="cyan")

	# Process Digital Humanities terms if required
	if process_dh:
		dh_df_path = os.path.join(data_directory_path, "metadata_files", "en.Digital humanities.json")
		dh_df = pd.DataFrame([json.load(open(dh_df_path, 'r', encoding='utf-8-sig'))])
		dh_df = dh_df.melt()
		dh_df.columns = ['language_code', 'term']
		merged_dh = pd.merge(dh_df, iso_languages, on='language_code', how='outer')
		merged_dh['term_source'] = 'Digital Humanities'

	# Generate a dataframe with all the terms to translate in all ISO 639-1 languages
	# Skip 'Digital Humanities' here when process_dh=True — it's added below via merged_dh
	# which carries existing translations from the JSON file, so we avoid duplicating rows.
	languages_dfs = []
	for term in target_terms:
		if process_dh and term == 'Digital Humanities':
			continue
		term_df = iso_languages.copy()
		term_df['term_source'] = term
		languages_dfs.append(term_df)

	if process_dh:
		languages_dfs.append(merged_dh)

	final_df = pd.concat(languages_dfs).reset_index(drop=True)
	console.print(f"Number of languages to translate terms into: {len(final_df)}", style="bright_cyan")


	# Check if Wikipedia pages exist for the terms
	if use_wikipedia:
		# Check if Wikipedia pages exist for the terms
		unique_term_sources = final_df['term_source'].unique()
		translations_dfs = []
		# Create a file to store errors
		wikipedia_error_file = os.path.join(error_dir, "wikipedia_translation_errors.csv")
		clean_write_error_file(wikipedia_error_file, drop_fields=["term_source", "language_code", "error_url"])
		# Load any previous errors
		if os.path.exists(wikipedia_error_file):
			existing_errors_df = read_csv_file(wikipedia_error_file)
			errored_terms = set(existing_errors_df['term_source'].dropna().unique())
		else:
			errored_terms = set()
		for term_source in unique_term_sources:
			if term_source in errored_terms:
				console.print(f"Skipping {term_source} due to previous errors", style="bold red")
				continue
			folder_term_source = term_source.replace(" ", "_").lower()
			translation_file_path = os.path.join(data_directory_path, "translated_terms", folder_term_source, "direct_services", "wikipedia_translations.csv")
			if os.path.exists(translation_file_path) and use_cached_translations:
				translations_df = read_csv_file(translation_file_path)
				console.print(f"Loaded Wikipedia translations for {term_source} from file", style="bold green")
				if 'coding_dh_date' in translations_df.columns:
					translations_df['coding_dh_date'] = pd.to_datetime(translations_df['coding_dh_date'])
					# Keep only the most recent translations
					translations_df = translations_df.sort_values(by='coding_dh_date').drop_duplicates(subset=['language_code'], keep='last')
				translations_dfs.append(translations_df)
			else:
				translations = check_if_wikipedia_page_exists(term_source, error_file_path=wikipedia_error_file, console=console)
				if translations:
					translations_df = pd.DataFrame([translations])
					translations_df = translations_df.T.reset_index().rename(columns={"index": "language_code", 0: "wikipedia_translated_term"})
					translations_df['term_source'] = term_source
					translations_df['coding_dh_date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
					# write translations to a file
					os.makedirs(os.path.dirname(translation_file_path), exist_ok=True)
					translations_df.to_csv(translation_file_path, index=False)
					console.print(f"Added Wikipedia translations for {term_source}", style="bold green")
					translations_dfs.append(translations_df)
		if translations_dfs:
			final_translations_df = pd.concat(translations_dfs).reset_index(drop=True)
			if len(final_translations_df) > 0:
				merge_cols = list(set(final_df.columns.tolist()) & set(final_translations_df.columns.tolist()))
				final_df = final_df.merge(final_translations_df, on=merge_cols, how='outer')
				console.print(f"Added Wikipedia translations for {term_source}", style="bold green")
				console.print(f"Number of languages with Wikipedia translations: {len(final_df[final_df.wikipedia_translated_term.notna()])}", style="bright_magenta")
				console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")
			else:
				console.print("No Wikipedia translations found", style="bold red")
		else:
			console.print("No Wikipedia translations found", style="bold red")
				
	# Translate the terms using Google Cloud Translate
	terms_file_path = os.path.join(data_directory_path, "translated_terms")
	if use_gt_translate:
		exploded_grouped_gt_terms = process_grouped_terms(
			translate_file_path=terms_file_path,
			translate_file_name="gt_translations.csv",
			translation_columns=['gt_translated_term'],
			final_df=final_df,
			translation_function=lambda row: get_gt_translation(row, gt_error_file, console),
			service_name="Google Cloud Translate",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=gt_error_file if exclude_previous_errors else None
		)

		final_df = _merge_translations(final_df, exploded_grouped_gt_terms, ['language_code', 'term_source'])

		console.print(
			f"Number of languages with Google Translate translations: {len(final_df[final_df.gt_translated_term.notna()])}", 
			style="bright_magenta"
		)
		console.print(
			f"Size of dataset currently {len(final_df)}", 
			style="bright_cyan"
		)

	# Translate the terms using EasyNMT
	if use_enmt_translate:
		exploded_grouped_enmt_terms = process_grouped_terms(
			translate_file_path=terms_file_path,
			translate_file_name="enmt_translations.csv",
			translation_columns=['enmt_translated_term'],
			final_df=final_df,
			translation_function=lambda row: get_enmt_translation(row, enmt_error_file, console),
			service_name="EasyNMT",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=enmt_error_file if exclude_previous_errors else None
		)

		final_df = _merge_translations(final_df, exploded_grouped_enmt_terms, ['language_code', 'term_source'])

		console.print(
			f"Number of languages with EasyNMT translations: {len(final_df[final_df.enmt_translated_term.notna()])}", 
			style="bright_magenta"
		)
		console.print(
			f"Size of dataset currently {len(final_df)}", 
			style="bright_cyan"
		)

	# Translate the terms using Lingvanex (free, top-ranked primary service per Kraus et al. 2025)
	if use_lingvanex_translate:
		exploded_grouped_lingvanex_terms = process_grouped_terms(
			translate_file_path=terms_file_path,
			translate_file_name="lingvanex_translations.csv",
			translation_columns=['lingvanex_translated_term'],
			final_df=final_df,
			translation_function=lambda row: get_lingvanex_translation(row, lingvanex_error_file, console),
			service_name="Lingvanex",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=lingvanex_error_file if exclude_previous_errors else None
		)
		final_df = _merge_translations(final_df, exploded_grouped_lingvanex_terms, ['language_code', 'term_source'])
		console.print(
			f"Number of languages with Lingvanex translations: {len(final_df[final_df.lingvanex_translated_term.notna()])}",
			style="bright_magenta"
		)
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	def _ollama_cols(prefix: str) -> list:
		return [
			f'{prefix}_translated_term', f'{prefix}_content', f'{prefix}_created_at',
			f'{prefix}_done_reason', f'{prefix}_eval_count', f'{prefix}_eval_duration',
			f'{prefix}_extracted_dictionaries', f'{prefix}_load_duration', f'{prefix}_model',
			f'{prefix}_prompt_eval_count', f'{prefix}_prompt_eval_duration',
			f'{prefix}_total_duration', f'{prefix}_translation', f'{prefix}_translation_rationale',
			f'{prefix}_system_prompt', f'{prefix}_user_prompt',
		]

	def _run_ollama_model(model_tag: str, col_prefix: str, file_prefix: str, error_file: str) -> None:
		"""Run one Ollama model and write col_prefix_* columns directly — no rename needed."""
		nonlocal final_df
		try:
			raw = process_individual_terms(
				translate_file_path=terms_file_path,
				translate_file_name=f"{file_prefix}_{prompt_variant}_translations.csv",
				translation_columns=_ollama_cols(col_prefix),
				final_df=final_df,
				translation_function=lambda row, ct=0, _p=col_prefix: get_ollama_translation(
					row, error_file, console, prompt_variant, term_contexts,
					ollama_model=model_tag, request_delay=2.0, request_timeout=240,
					consecutive_timeouts=ct, col_prefix=_p,
				),
				service_name=file_prefix.capitalize(),
				should_use_cached_translations=use_cached_translations,
				should_override_wikipedia=override_wikipedia,
				post_process_function=lambda df, _p=col_prefix: post_process_ollama(df, _p),
				exclude_errors_file=error_file if exclude_previous_errors else None,
				variant=prompt_variant,
			)
		except OllamaUnresponsiveError:
			console.print(f"✗ {file_prefix} stopped early — Ollama unresponsive. Progress saved; rerun to continue.", style="bold red")
			return
		new_cols = [c for c in raw.columns if c.startswith(f'{col_prefix}_')]
		final_df = _merge_translations(final_df, raw[['language_code', 'term_source'] + new_cols], ['language_code', 'term_source'])
		term_col = f'{col_prefix}_translated_term'
		if term_col in final_df.columns:
			console.print(f"Languages with {file_prefix} translations: {final_df[term_col].notna().sum()}", style="bright_magenta")

	# Translate using Llama (local Ollama)
	if use_ollama_translate:
		_run_ollama_model(OLLAMA_LLAMA_MODEL, 'llama', 'llama', llama_error_file)
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate using Gemma (local Ollama)
	if use_gemma_translate:
		_run_ollama_model(OLLAMA_GEMMA_MODEL, 'gemma', 'gemma', gemma_error_file)
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate using Qwen (local Ollama)
	if use_qwen_translate:
		_run_ollama_model(OLLAMA_QWEN_MODEL, 'qwen', 'qwen', qwen_error_file)
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate using Mistral (local Ollama)
	if use_mistral_translate:
		_run_ollama_model(OLLAMA_MISTRAL_MODEL, 'mistral', 'mistral', mistral_error_file)
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using OpenAI
	if use_openai_translate:
		final_openai_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name=f"openai_{prompt_variant}_translations.csv",
			translation_columns=['openai_translated_term', 'openai_completion_tokens', 'openai_created', 'openai_finish_reason',
			'openai_model', 'openai_prompt_tokens', 'openai_total_tokens',
			'openai_translation_rationale', 'openai_translation'],
			final_df=final_df,
			translation_function=lambda row: get_openai_translation(row, openai_error_file, console, prompt_variant, term_contexts),
			service_name="OpenAI",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=openai_error_file if exclude_previous_errors else None,
			variant=prompt_variant,
		)

		# Extract all columns that include 'openai'
		openai_cols  = [col for col in final_openai_translations.columns.tolist() if 'openai' in col]
		# Merge results back
		final_df = _merge_translations(final_df, final_openai_translations[['language_code', 'term_source'] + openai_cols], ['language_code', 'term_source'])
		console.print(f"Number of languages with OpenAI translations: {len(final_df[final_df.openai_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using Claude
	if use_claude_translate:
		final_claude_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name=f"claude_{prompt_variant}_translations.csv",
			translation_columns=['claude_translated_term', 'claude_output_tokens', 'claude_created', 'claude_stop_reason',
			'claude_model', 'claude_input_tokens', 'claude_total_tokens',
			'claude_translation_rationale', 'claude_translation'],
			final_df=final_df,
			translation_function=lambda row: get_claude_translation(row, claude_error_file, console, prompt_variant, term_contexts),
			service_name="Claude",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=claude_error_file if exclude_previous_errors else None,
			variant=prompt_variant,
		)

		# Extract all columns that include 'claude'
		claude_cols = [col for col in final_claude_translations.columns.tolist() if 'claude' in col]
		# Merge results back
		final_df = _merge_translations(final_df, final_claude_translations[['language_code', 'term_source'] + claude_cols], ['language_code', 'term_source'])
		console.print(f"Number of languages with Claude translations: {len(final_df[final_df.claude_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using Gemini (gemini-2.0-flash, top-ranked LLM per Kraus et al. 2025)
	if use_gemini_translate:
		final_gemini_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name=f"gemini_{prompt_variant}_translations.csv",
			translation_columns=['gemini_translated_term', 'gemini_translation_rationale', 'gemini_model', 'gemini_created'],
			final_df=final_df,
			translation_function=lambda row: get_gemini_translation(row, gemini_error_file, console, prompt_variant, term_contexts),
			service_name="Gemini",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=gemini_error_file if exclude_previous_errors else None,
			variant=prompt_variant,
		)
		gemini_cols = [col for col in final_gemini_translations.columns.tolist() if 'gemini' in col]
		final_df = _merge_translations(final_df, final_gemini_translations[['language_code', 'term_source'] + gemini_cols], ['language_code', 'term_source'])
		console.print(f"Number of languages with Gemini translations: {len(final_df[final_df.gemini_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	# Translate the terms using DeepSeek (deepseek-chat / DeepSeek-V3)
	if use_deepseek_translate:
		final_deepseek_translations = process_individual_terms(
			translate_file_path=terms_file_path,
			translate_file_name=f"deepseek_{prompt_variant}_translations.csv",
			translation_columns=['deepseek_translated_term', 'deepseek_completion_tokens', 'deepseek_created',
			'deepseek_finish_reason', 'deepseek_model', 'deepseek_prompt_tokens', 'deepseek_total_tokens',
			'deepseek_translation_rationale', 'deepseek_translation'],
			final_df=final_df,
			translation_function=lambda row: get_deepseek_translation(row, deepseek_error_file, console, prompt_variant, term_contexts),
			service_name="DeepSeek",
			should_use_cached_translations=use_cached_translations,
			should_override_wikipedia=override_wikipedia,
			exclude_errors_file=deepseek_error_file if exclude_previous_errors else None,
			variant=prompt_variant,
		)
		deepseek_cols = [col for col in final_deepseek_translations.columns.tolist() if 'deepseek' in col]
		final_df = _merge_translations(final_df, final_deepseek_translations[['language_code', 'term_source'] + deepseek_cols], ['language_code', 'term_source'])
		console.print(f"Number of languages with DeepSeek translations: {len(final_df[final_df.deepseek_translated_term.notna()])}", style="bright_magenta")
		console.print(f"Size of dataset currently {len(final_df)}", style="bright_cyan")

	console.print(f"Columns of final_df currently after getting all terms: {final_df.columns}", style="bright_cyan")

	# Auto-continue without prompting
	console.print("Do you want to continue? (y/n): y [auto]", style="dim")
	
	# Clean up the final DataFrame
	cleaned_df = final_df.reset_index(drop=True)

	# Initialize 'term' column if not present
	if 'term' not in cleaned_df.columns:
		cleaned_df['term'] = None

	# Prioritize translations in order of preference
	translation_sources = [
		'wikipedia_translated_term',
		'gemini_translated_term', 'openai_translated_term', 'claude_translated_term', 'deepseek_translated_term',
		'llama_translated_term', 'gemma_translated_term', 'qwen_translated_term', 'mistral_translated_term',
		'lingvanex_translated_term', 'gt_translated_term', 'enmt_translated_term',
	]

	for source in translation_sources:
		if source in cleaned_df.columns:
			cleaned_df.loc[(cleaned_df[source].notna()) & (cleaned_df.term.isna()), 'term'] = cleaned_df[source]

	console.print(f"Number of languages with terms: {len(cleaned_df[cleaned_df.term.notna()])}", style="bright_magenta")

	# Ensure we only apply html.unescape to non-None values
	for col in translation_sources + ['term']:
		if col in cleaned_df.columns:
			cleaned_df[col] = cleaned_df[col].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

	# Keep temp folder (auto-no)
	console.print("Do you want to delete the temp folder? (y/n): n [auto]", style="dim")

	if not return_all_data:
		cleaned_df = cleaned_df[cleaned_df.term.notna()]

	return cleaned_df

def combine_language_data(directionality_df: pd.DataFrame, translated_terms_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Merge directionality metadata with translated terms and return per-term DataFrames.

	For each unique term_source, this function:
	  - Joins the directionality DataFrame (LTR/RTL only) with the translated terms.
	  - Builds a grouped summary (one row per unique translation, with language counts).
	  - Verifies directionality against the full directionality table.

	No files are written here — per-service files are saved by process_individual_terms /
	process_grouped_terms, and the validated output is written by evaluate_human_review.py.

	Parameters
	----------
	directionality_df : pd.DataFrame
		DataFrame with columns including ``language_code`` and ``directionality`` (values: 'ltr', 'rtl').
	translated_terms_df : pd.DataFrame
		DataFrame containing translated terms, with at minimum ``language_code``, ``term_source``,
		and ``term`` columns.

	Returns
	-------
	Tuple[pd.DataFrame, pd.DataFrame]
		``(processed_translations_df, grouped_translations_df)`` — the row-per-language
		translations and the grouped-by-term summary across all term_sources.
	"""
	# Subset directionality to LTR and RTL languages
	directionality_df = directionality_df[directionality_df.directionality.isin(['ltr', 'rtl'])]

	# Merge the directionality data with the translated terms
	merged_lang_terms_df = pd.merge(directionality_df[['language_code', 'directionality', 'English language name', 'local language name', 'comment', 'local or English Wikipedia article']], translated_terms_df, on='language_code', how="outer")
	merged_lang_terms_df = merged_lang_terms_df[merged_lang_terms_df.language_code != "see also Test languages"]

	console.print(f"Our data now contains info for {merged_lang_terms_df[merged_lang_terms_df.term.notna()]['English language name'].nunique()} but we also are missing terms for the following number of languages {merged_lang_terms_df[merged_lang_terms_df.term.isna()]['English language name'].nunique()}", style="bold green")
	merged_lang_terms_df = merged_lang_terms_df[merged_lang_terms_df.term.notna()]
	# Fill NaN values with an empty string
	merged_lang_terms_df['English language name'] = merged_lang_terms_df['English language name'].fillna('').astype(str)
	merged_lang_terms_df['local language name'] = merged_lang_terms_df['local language name'].fillna('').astype(str)
	merged_lang_terms_df['local or English Wikipedia article'] = merged_lang_terms_df['local or English Wikipedia article'].fillna('').astype(str)
	merged_lang_terms_df['comment'] = merged_lang_terms_df['comment'].fillna('').astype(str)
	merged_lang_terms_df = merged_lang_terms_df.reset_index(drop=True)
	processed_translations_dfs = []
	grouped_translations_dfs = []
	for term_source in merged_lang_terms_df.term_source.unique():
		console.print(f"Processing term: {term_source}", style="bold green")
		subset_merged_lang_terms_df = merged_lang_terms_df[merged_lang_terms_df.term_source == term_source]
		subset_merged_lang_terms_df = subset_merged_lang_terms_df.reset_index(drop=True)
		processed_translations_dfs.append(subset_merged_lang_terms_df)
		grouped_terms_df = subset_merged_lang_terms_df.groupby(['term_source', 'term']).agg({
			'language_code': lambda x: ', '.join(x.dropna().astype(str)),
			'term': 'count',
			'directionality': lambda x: ','.join(set(x.dropna().astype(str))),
			'English language name': lambda x: ', '.join(x.dropna().astype(str)),
			'local language name': lambda x: ', '.join(set(x.dropna().astype(str))),
			'local or English Wikipedia article': lambda x: ', '.join(set(x.dropna().astype(str))),
			'comment': lambda x: ', '.join(set(x.dropna().astype(str))),
		}).reset_index(level=0)
		grouped_terms_df = grouped_terms_df.rename(columns={'term': 'counts'})
		grouped_terms_df['term'] = grouped_terms_df.index
		grouped_terms_df = grouped_terms_df.reset_index(drop=True)
		grouped_translations_dfs.append(grouped_terms_df)
	processed_translations_df = pd.concat(processed_translations_dfs)
	grouped_translations_df = pd.concat(grouped_translations_dfs)
	return processed_translations_df, grouped_translations_df


def load_existing_data(file_paths: List[str], skip_existing_files: bool) -> Tuple[bool, List[pd.DataFrame]]:
	"""
	Check and load existing data from a list of file paths.

	Parameters:
		file_paths (List[str]): A list of file paths to check for existing data.
		skip_existing_files (bool): A flag indicating whether to load existing datasets

	Returns:
		(bool, List[pd.DataFrame]): A flag indicating if all data exists and a list of dataframes loaded.
	"""
	all_files_exist = all(os.path.exists(path) for path in file_paths)
	if all_files_exist and not skip_existing_files:
		return True, [read_csv_file(path) for path in file_paths]
	return False, []

def generate_translated_terms(data_directory_path: str, target_terms: List[str], directionality_df: pd.DataFrame, gt_translate: bool, enmt_translate: bool, openai_translate: bool, claude_translate: bool, ollama_translate: bool, wikipedia_translate: bool, override_wikipedia: bool, cached_translations: bool, return_full_terms_data: bool, excluding_previous_errors: bool, prompt_variant: str = 'minimal', gemini_translate: bool = False, lingvanex_translate: bool = False, deepseek_translate: bool = False, gemma_translate: bool = False, qwen_translate: bool = False, mistral_translate: bool = False, term_contexts: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	Generate translated terms for a list of target terms across all configured services.

	Orchestrates the full translation pipeline: loads cached data where available, runs each enabled service in order (direct services first, then LLMs), verifies directionality, and combines results into three output DataFrames.


	Parameters
	----------
	data_directory_path : str
		Root directory containing metadata_files and translation output subdirectories.
	target_terms : List[str]
		Terms to translate across all ISO 639-1 languages.
	directionality_df : pd.DataFrame
		DataFrame mapping language codes to text directionality (LTR/RTL).
	gt_translate : bool
		Whether to run Google Cloud Translate.
	enmt_translate : bool
		Whether to run EasyNMT.
	openai_translate : bool
		Whether to run OpenAI (GPT-4o).
	claude_translate : bool
		Whether to run Claude.
	ollama_translate : bool
		Whether to run the local Ollama model (first pass).
	wikipedia_translate : bool
		Whether to check for existing Wikipedia page translations.
	override_wikipedia : bool
		If True, translate rows that already have a Wikipedia translation. If False, skip them.
	cached_translations : bool
		If True, load previously saved per-service translations rather than re-calling APIs.
	return_full_terms_data : bool
		If True, return all rows including those with null translations. If False, drop nulls.
	excluding_previous_errors : bool
		If True, skip rows that previously errored (logged in service error files).
	prompt_variant : str
		LLM prompt strategy. Options: 'minimal', 'expert_persona', 'native_rationale', 'judge'. Defaults to 'minimal'.
	gemini_translate : bool
		Whether to run Gemini. Defaults to False.
	lingvanex_translate : bool
		Whether to run Lingvanex. Defaults to False.
	term_contexts : dict, optional
		Mapping of term_source -> plain-text context description, used by the
		'judge' prompt variant (structured per-language context from aggregate_variant_translations).

	Returns
	-------
	Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
		(combined_translated_df, combined_processed_df, combined_grouped_df) — the raw
		translations, the processed/verified translations, and the grouped-by-term output.
	"""
	term_contexts = term_contexts or {}
	console.print(f"Using prompt variant: {prompt_variant}", style="bold cyan")

	console.print(f"Generating translations for: {target_terms}", style="bold green")
	translated_terms_df = generate_initial_terms(
		target_terms, data_directory_path, 'Digital Humanities' in target_terms,
		gt_translate, enmt_translate, openai_translate, claude_translate, ollama_translate,
		wikipedia_translate, override_wikipedia, cached_translations, return_full_terms_data, excluding_previous_errors,
		use_gemini_translate=gemini_translate, use_lingvanex_translate=lingvanex_translate,
		use_deepseek_translate=deepseek_translate, use_gemma_translate=gemma_translate,
		use_qwen_translate=qwen_translate, use_mistral_translate=mistral_translate,
		prompt_variant=prompt_variant, term_contexts=term_contexts,
	)
	if not translated_terms_df.empty:
		combined_processed_df, combined_grouped_df = combine_language_data(directionality_df, translated_terms_df)
	else:
		console.print("No data available for processing. Skipping combine_language_data.", style="bold yellow")
		combined_processed_df, combined_grouped_df = pd.DataFrame(), pd.DataFrame()

	return translated_terms_df, combined_processed_df, combined_grouped_df


if __name__ == '__main__':
	import argparse as _argparse

	_ALL_VARIANTS = ['minimal', 'expert_persona', 'native_rationale', 'judge']

	_parser = _argparse.ArgumentParser(
		description="Run the translation pipeline.",
		formatter_class=_argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Examples:\n"
			"  # Run all four variants for local Ollama models overnight:\n"
			"  python generate_translations.py --terms 'Digital Humanities' --ollama-only --variant minimal expert_persona native_rationale judge\n\n"
			"  # Run all four variants for API models in a second terminal:\n"
			"  python generate_translations.py --terms 'Digital Humanities' --api-only --variant minimal expert_persona native_rationale judge\n\n"
			"  # Run a single variant with fine-grained control:\n"
			"  python generate_translations.py --terms 'Digital Humanities' --variant expert_persona --no-gt --no-enmt --no-lingvanex --no-wikipedia --no-openai --no-claude --no-gemini --no-deepseek"
		),
	)
	_parser.add_argument('--variant', nargs='+', default=['minimal'],
		choices=_ALL_VARIANTS, metavar='VARIANT',
		help=(
			'One or more prompt variants to run in sequence '
			'(choices: minimal, expert_persona, native_rationale, judge; default: minimal). '
			'E.g. --variant minimal expert_persona native_rationale judge'
		))
	_parser.add_argument('--terms', nargs='+', default=['Computational Humanities'],
		help='Target term(s) to translate (default: "Computational Humanities")')
	# ── Convenience group flags (mutually exclusive) ──────────────────────────
	_group = _parser.add_mutually_exclusive_group()
	_group.add_argument('--ollama-only', action='store_true',
		help='Run only local Ollama models (Llama, Gemma, Qwen, Mistral); skip all API LLMs and baseline services')
	_group.add_argument('--api-only', action='store_true',
		help='Run only API LLM models (OpenAI, Claude, Gemini, DeepSeek); skip all Ollama models and baseline services')
	# ── Individual skip flags ─────────────────────────────────────────────────
	_parser.add_argument('--no-gt',        action='store_true', help='Skip Google Translate')
	_parser.add_argument('--no-enmt',      action='store_true', help='Skip EasyNMT')
	_parser.add_argument('--no-openai',    action='store_true', help='Skip OpenAI')
	_parser.add_argument('--no-claude',    action='store_true', help='Skip Claude')
	_parser.add_argument('--no-ollama',    action='store_true', help='Skip Llama (local Ollama)')
	_parser.add_argument('--no-gemini',    action='store_true', help='Skip Gemini')
	_parser.add_argument('--no-lingvanex', action='store_true', help='Skip Lingvanex')
	_parser.add_argument('--no-wikipedia', action='store_true', help='Skip Wikipedia')
	_parser.add_argument('--no-deepseek',  action='store_true', help='Skip DeepSeek')
	_parser.add_argument('--no-gemma',     action='store_true', help='Skip Gemma (local Ollama)')
	_parser.add_argument('--no-qwen',      action='store_true', help='Skip Qwen (local Ollama)')
	_parser.add_argument('--no-mistral',   action='store_true', help='Skip Mistral (local Ollama)')
	_args = _parser.parse_args()

	local_data_directory_path = get_data_directory_path()
	local_target_terms: list = _args.terms
	existing_directionality_df = load_language_codes()

	# --ollama-only / --api-only override individual service flags
	_skip_baselines = _args.ollama_only or _args.api_only
	_skip_api_llms  = _args.ollama_only
	_skip_local_llms = _args.api_only

	should_use_gt_translate        = not (_args.no_gt        or _skip_baselines)
	should_use_enmt_translate      = not (_args.no_enmt      or _skip_baselines)
	should_use_lingvanex_translate = not (_args.no_lingvanex or _skip_baselines)
	should_use_wikipedia_translate = not (_args.no_wikipedia or _skip_baselines)
	should_use_openai_translate    = not (_args.no_openai    or _skip_api_llms)
	should_use_claude_translate    = not (_args.no_claude    or _skip_api_llms)
	should_use_gemini_translate    = not (_args.no_gemini    or _skip_api_llms)
	should_use_deepseek_translate  = not (_args.no_deepseek  or _skip_api_llms)
	should_use_ollama_translate    = not (_args.no_ollama    or _skip_local_llms)
	should_use_gemma_translate     = not (_args.no_gemma     or _skip_local_llms)
	should_use_qwen_translate      = not (_args.no_qwen      or _skip_local_llms)
	should_use_mistral_translate   = not (_args.no_mistral   or _skip_local_llms)
	should_override_wikipedia      = True
	should_use_cached_translations = True
	should_return_all_terms        = False
	should_exclude_previous_errors = True

	_judge_contexts: dict = {}

	for _variant in _args.variant:
		if _variant == 'judge':
			if not _judge_contexts:
				print("Aggregating translations from all prior variants and services for judge...")
				from generate_translation_prompts import aggregate_variant_translations
				_judge_contexts = aggregate_variant_translations(local_data_directory_path, local_target_terms)
			if not _judge_contexts:
				print("⚠ No prior variant outputs found — run minimal/expert_persona/native_rationale first.")
				continue
		combined_translated_df, combined_processed_df, combined_grouped_df = generate_translated_terms(
			local_data_directory_path, local_target_terms, existing_directionality_df,
			should_use_gt_translate, should_use_enmt_translate, should_use_openai_translate,
			should_use_claude_translate, should_use_ollama_translate, should_use_wikipedia_translate,
			should_override_wikipedia, should_use_cached_translations, should_return_all_terms,
			should_exclude_previous_errors, prompt_variant=_variant,
			gemini_translate=should_use_gemini_translate, lingvanex_translate=should_use_lingvanex_translate,
			deepseek_translate=should_use_deepseek_translate, gemma_translate=should_use_gemma_translate,
			qwen_translate=should_use_qwen_translate, mistral_translate=should_use_mistral_translate,
			term_contexts=_judge_contexts if _variant == 'judge' else {},
		)