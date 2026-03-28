"""
Translation service wrappers for multiple translation APIs.
Includes Google Translate, EasyNMT, OpenAI, Claude, Gemini, LingVanex, Ollama, and Wikipedia.
"""

import json
import time
import threading
from typing import Tuple
import pandas as pd
from rich.console import Console
from openai import OpenAI, OpenAIError
from anthropic import Anthropic, APIError
import datetime
import wikipediaapi
import ollama
from pydantic import ValidationError

from data_processing import parse_translation_response, is_enmt_model_available
from data_generation_scripts.utils import log_error_to_file
from data_generation_scripts.translation_prompts import get_prompt

# Constants
MAX_CONSECUTIVE_TIMEOUTS = 3

# These imports will be conditional based on availability
try:
	from google import genai as google_genai
	import google.genai.types as google_genai_types
except ImportError:
	google_genai = None
	google_genai_types = None

try:
	import translators as ts_lib
except ImportError:
	ts_lib = None


def get_gt_translation(row: pd.Series, error_file_path: str, console: Console, translate_client) -> pd.Series:
	"""
	Function to get translations of terms

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	translate_client
		Google Translate client object

	Returns
	-------
	pd.Series
		A series with the translated terms
	"""
	time.sleep(1)  # Add a delay to avoid API rate limits
	try:
		dh_terms = row.term_source
		target_language = row.language_code
		console.print(f"Translating {dh_terms} to {target_language}", style="bold green")

		# Translate terms in bulk
		text_results = translate_client.translate(dh_terms, target_language=target_language)

		# Match translations explicitly with the original terms
		translated_terms = [result['translatedText'] for result in text_results]
		row['gt_translated_term'] = list(zip(dh_terms, translated_terms))
	except Exception as e:
		error_str = str(e)

		# Check if it's an unsupported language error
		if 'Bad language pair' in error_str or 'badRequest' in error_str.lower():
			console.print(f"⚠ Google Translate does not support language: {target_language}", style="bold yellow")
			status_code = 400
			error_msg = f"Unsupported language: {target_language}"
		else:
			console.print(f"✗ Error translating {dh_terms} to {target_language}: {e}", style="bold red")
			status_code = 500
			error_msg = str(e)

		row['gt_translated_term'] = list(zip(dh_terms, [None] * len(dh_terms)))  # Handle error by returning a list of None
		# Log structured error
		additional_data = {
			'term_source': dh_terms,
			'language_code': target_language,
			'gt_translated_term': None
		}
		log_error_to_file(
			error_file_path=error_file_path,
			additional_data=additional_data,
			status_code=status_code,
			error_url=f"translate_client.translate({target_language}) - {error_msg}"
		)

	return row

def get_enmt_translation(row: pd.Series, error_file_path: str, console: Console, model) -> pd.Series:
	"""
	Function to get translations of terms using EasyNMT

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	model
		EasyNMT model object

	Returns
	-------
	pd.Series
		A series with the translated terms
	"""
	time.sleep(1)  # Add a delay to avoid rate limits or excessive load
	try:
		dh_terms = row.term_source
		target_language = row.language_code
		console.print(f"Translating {dh_terms} to {target_language}", style="bold green")

		# Check if the model is available for the target language
		if is_enmt_model_available(target_language):
			# Translate terms in bulk
			translated_terms = model.translate(dh_terms, target_lang=target_language, source_lang='en')
			row['enmt_translated_term'] = list(zip(dh_terms, translated_terms))
		else:
			console.print(f"No model available for {target_language}. Skipping translation.", style="bold yellow")
			row['enmt_translated_term'] = list(zip(dh_terms, [None] * len(dh_terms)))
			# Log structured error for missing model
			additional_data = {
				'term_source': dh_terms,
				'language_code': target_language,
				'enmt_translated_term': None
			}
			log_error_to_file(
				error_file_path=error_file_path,
				additional_data=additional_data,
				status_code=404,
				error_url=f"EasyNMT model not found: opus-mt-en-{target_language}"
			)

	except Exception as e:
		console.print(f"Error translating {dh_terms} to {target_language}: {e}", style="bold red")
		row['enmt_translated_term'] = list(zip(dh_terms, [None] * len(dh_terms)))  # Handle error by returning a list of None
		# Log structured error
		additional_data = {
			'term_source': dh_terms,
			'language_code': target_language,
			'enmt_translated_term': None
		}
		log_error_to_file(
			error_file_path=error_file_path,
			additional_data=additional_data,
			status_code=500,
			error_url=f"model.translate({target_language})"
		)


	return row

def get_openai_translation(row: pd.Series, error_file_path: str, console: Console, client: OpenAI,
							current_prompt_variant: str, current_term_contexts: dict) -> pd.Series:
	"""
	Function to get translations of terms using the OpenAI API.

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	client : OpenAI
		OpenAI client object
	current_prompt_variant : str
		The prompt variant to use ('comparative' or other)
	current_term_contexts : dict
		Dictionary mapping terms to their contexts

	Returns
	-------
	pd.Series
		A series with the translated terms from OpenAI.
	"""
	time.sleep(1)  # Avoid hitting API rate limits
	try:
		term_source = row.term_source
		language_name = row.language_name
		language_code = row.language_code
		gt_translated_term = row.gt_translated_term if 'gt_translated_term' in row else None
		enmt_translated_term = row.enmt_translated_term if 'enmt_translated_term' in row else None
		ollama_translated_term = row.ollama_translated_term if 'ollama_translated_term' in row else None
		wikipedia_translated_term = row.wikipedia_translated_term if 'wikipedia_translated_term' in row else None

		console.print(f"Translating {term_source} to {language_name} using OpenAI API", style="bold green")

		# Get the prompt variant
		existing_translations = None
		if current_prompt_variant == 'comparative':
			existing_translations = {
				'gt': gt_translated_term if pd.notna(gt_translated_term) else None,
				'enmt': enmt_translated_term if pd.notna(enmt_translated_term) else None,
				'lingvanex': row.lingvanex_translated_term if 'lingvanex_translated_term' in row and pd.notna(row.lingvanex_translated_term) else None,
				'ollama': ollama_translated_term if pd.notna(ollama_translated_term) else None,
				'wikipedia': wikipedia_translated_term if pd.notna(wikipedia_translated_term) else None,
			}

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
			context=current_term_contexts.get(term_source),
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		# Define sample messages
		prompt_messages = [
			{"role": "system", "content": f"You are a {term_source} scholar who speaks many languages."},
			{"role": "user", "content": user_content}
		]

		# Log the input being sent to the OpenAI API
		console.print(f"Prompt messages: {json.dumps(prompt_messages, indent=2)}", style="bold yellow")

		# Call the OpenAI Chat API directly using the `openai` module
		chat_completion = client.chat.completions.create(
			model="gpt-4o",
			messages=prompt_messages,
		)
		console.print(f"Chat completion: {chat_completion}", style="bright_cyan")
		# Extract the response and the rationale
		message_content = chat_completion.choices[0].message.content
		response = parse_translation_response(message_content)
		translated_term = response.translated_term
		translation_rationale = response.translation_rationale

		# Extract metadata
		metadata = {
			'translated_term': translated_term,
			'translation_rationale': translation_rationale,
			'model': chat_completion.model,
			'finish_reason': chat_completion.choices[0].finish_reason,
			'created': chat_completion.created,
			'total_tokens': chat_completion.usage.total_tokens,
			'prompt_tokens': chat_completion.usage.prompt_tokens,
			'completion_tokens': chat_completion.usage.completion_tokens
		}

		# Update the DataFrame row with the metadata
		for key, value in metadata.items():
			row["openai_" + key] = value
	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from OpenAI: {e}", style="bold red")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"openai_translated_term": None
			},
			status_code=400,
			error_url="openai.chat.completions.create - JSONDecodeError"
		)
	except OpenAIError as e:
		error_str = str(e)
		# Quota / billing errors will never recover mid-run — abort the entire OpenAI
		# pass immediately rather than hammering the API for every remaining row.
		if 'insufficient_quota' in error_str or 'quota' in error_str.lower() or getattr(e, 'status_code', None) == 429:
			console.print(f"✗ OpenAI quota exceeded — aborting OpenAI translation pass: {e}", style="bold red")
			log_error_to_file(
				error_file_path,
				additional_data={
					"term_source": term_source,
					"language_code": language_code,
					"openai_translated_term": None
				},
				status_code=429,
				error_url="openai.chat.completions.create - QuotaExceeded"
			)
			raise Exception("OpenAI quota exceeded — aborting OpenAI translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using OpenAI: {e}", style="bold red")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"openai_translated_term": None
			},
			status_code=500,
			error_url="openai.chat.completions.create - OpenAIError"
		)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"openai_translated_term": None
			},
			status_code=500,
			error_url="openai.chat.completions.create - General Exception"
		)

	return row

def get_claude_translation(row: pd.Series, error_file_path: str, console: Console, claude_client: Anthropic,
							current_prompt_variant: str, current_term_contexts: dict) -> pd.Series:
	"""
	Function to get translations of terms using the Claude API.

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	claude_client : Anthropic
		Claude client object
	current_prompt_variant : str
		The prompt variant to use ('comparative' or other)
	current_term_contexts : dict
		Dictionary mapping terms to their contexts

	Returns
	-------
	pd.Series
		A series with the translated terms from Claude.
	"""
	time.sleep(1)  # Avoid hitting API rate limits
	try:
		term_source = row.term_source
		language_name = row.language_name
		language_code = row.language_code
		gt_translated_term = row.gt_translated_term if 'gt_translated_term' in row else None
		enmt_translated_term = row.enmt_translated_term if 'enmt_translated_term' in row else None
		openai_translated_term = row.openai_translated_term if 'openai_translated_term' in row else None
		ollama_translated_term = row.ollama_translated_term if 'ollama_translated_term' in row else None
		wikipedia_translated_term = row.wikipedia_translated_term if 'wikipedia_translated_term' in row else None

		console.print(f"Translating {term_source} to {language_name} using Claude API", style="bold green")

		# Get the prompt variant
		existing_translations = None
		if current_prompt_variant == 'comparative':
			existing_translations = {
				'gt': gt_translated_term if pd.notna(gt_translated_term) else None,
				'enmt': enmt_translated_term if pd.notna(enmt_translated_term) else None,
				'lingvanex': row.lingvanex_translated_term if 'lingvanex_translated_term' in row and pd.notna(row.lingvanex_translated_term) else None,
				'openai': openai_translated_term if pd.notna(openai_translated_term) else None,
				'ollama': ollama_translated_term if pd.notna(ollama_translated_term) else None,
				'wikipedia': wikipedia_translated_term if pd.notna(wikipedia_translated_term) else None,
			}

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
			context=current_term_contexts.get(term_source),
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")

		# Define system message
		system_content = f"You are a {term_source} scholar who speaks many languages."
		console.print(f"Claude user prompt: {user_content}", style="bold yellow")

		# Call the Claude API
		message = claude_client.messages.create(
			model="claude-sonnet-4-5",
			max_tokens=500,
			system=system_content,
			messages=[
				{"role": "user", "content": user_content}
			]
		)
		console.print(f"Claude response: {message}", style="bright_cyan")

		# Extract the response
		message_content = message.content[0].text
		response = parse_translation_response(message_content)
		translated_term = response.translated_term
		translation_rationale = response.translation_rationale

		# Extract metadata
		metadata = {
			'translated_term': translated_term,
			'translation_rationale': translation_rationale,
			'model': message.model,
			'stop_reason': message.stop_reason,
			'created': datetime.datetime.now().isoformat(),
			'total_tokens': message.usage.input_tokens + message.usage.output_tokens,
			'input_tokens': message.usage.input_tokens,
			'output_tokens': message.usage.output_tokens
		}

		# Update the DataFrame row with the metadata
		for key, value in metadata.items():
			row["claude_" + key] = value
	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from Claude: {e}", style="bold red")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"claude_translated_term": None
			},
			status_code=400,
			error_url="claude.messages.create - JSONDecodeError"
		)
	except APIError as e:
		error_str = str(e)
		status_code = getattr(e, 'status_code', 500)
		# 404 means the model name is wrong/retired — this is an infrastructure problem,
		# not a per-language failure. Don't log it to the persistent error file or it
		# will exclude every language on the next run even after the model is fixed.
		if status_code == 404 or 'not_found_error' in error_str:
			console.print(f"✗ Claude model not found — check model string in get_claude_translation: {e}", style="bold red")
			raise Exception("Claude model not found — aborting Claude translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using Claude: {e}", style="bold red")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"claude_translated_term": None
			},
			status_code=status_code,
			error_url="claude.messages.create - APIError"
		)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"claude_translated_term": None
			},
			status_code=500,
			error_url="claude.messages.create - General Exception"
		)

	return row

def get_gemini_translation(row: pd.Series, error_file_path: str, console: Console, gemini_client,
							GEMINI_AVAILABLE: bool, GEMINI_MODEL: str, current_prompt_variant: str,
							current_term_contexts: dict) -> pd.Series:
	"""
	Translate a single row using the Google Gemini API (gemini-2.0-flash by default).

	Gemini 2.0 Flash was the top-ranked LLM in Kraus et al. (2025) for DH thesaurus
	translation, outperforming GPT-4o and Claude 3.5 Sonnet on quality while being
	significantly cheaper and having higher rate limits.

	Parameters
	----------
	row : pd.Series
		Must have term_source, language_name, language_code.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	gemini_client
		Gemini client object
	GEMINI_AVAILABLE : bool
		Whether Gemini is available
	GEMINI_MODEL : str
		The Gemini model name to use
	current_prompt_variant : str
		The prompt variant to use ('comparative' or other)
	current_term_contexts : dict
		Dictionary mapping terms to their contexts

	Returns
	-------
	pd.Series
		Row with gemini_* columns populated.
	"""
	if not GEMINI_AVAILABLE or gemini_client is None:
		console.print("⚠ Gemini not available — skipping row", style="bold yellow")
		row['gemini_translated_term'] = None
		return row

	time.sleep(1)
	try:
		term_source = row.term_source
		language_name = row.language_name
		language_code = row.language_code
		gt_translated_term = row.gt_translated_term if 'gt_translated_term' in row else None
		enmt_translated_term = row.enmt_translated_term if 'enmt_translated_term' in row else None
		openai_translated_term = row.openai_translated_term if 'openai_translated_term' in row else None
		ollama_translated_term = row.ollama_translated_term if 'ollama_translated_term' in row else None
		wikipedia_translated_term = row.wikipedia_translated_term if 'wikipedia_translated_term' in row else None

		console.print(f"Translating {term_source} to {language_name} using Gemini API ({GEMINI_MODEL})", style="bold green")

		existing_translations = None
		if current_prompt_variant == 'comparative':
			existing_translations = {
				'gt': gt_translated_term if pd.notna(gt_translated_term) else None,
				'enmt': enmt_translated_term if pd.notna(enmt_translated_term) else None,
				'lingvanex': row.lingvanex_translated_term if 'lingvanex_translated_term' in row and pd.notna(row.lingvanex_translated_term) else None,
				'openai': openai_translated_term if pd.notna(openai_translated_term) else None,
				'claude': row.claude_translated_term if 'claude_translated_term' in row and pd.notna(row.claude_translated_term) else None,
				'ollama': ollama_translated_term if pd.notna(ollama_translated_term) else None,
				'wikipedia': wikipedia_translated_term if pd.notna(wikipedia_translated_term) else None,
			}

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
			context=current_term_contexts.get(term_source),
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		console.print(f"Gemini user prompt: {user_content}", style="bold yellow")

		system_content = f"You are a {term_source} scholar who speaks many languages."
		response = gemini_client.models.generate_content(
			model=GEMINI_MODEL,
			contents=user_content,
			config=google_genai_types.GenerateContentConfig(
				system_instruction=system_content,
				temperature=0,
				max_output_tokens=500,
			),
		)
		console.print(f"Gemini response: {response}", style="bright_cyan")

		message_content = response.text
		parsed = parse_translation_response(message_content)

		metadata = {
			'translated_term': parsed.translated_term,
			'translation_rationale': parsed.translation_rationale,
			'model': GEMINI_MODEL,
			'created': datetime.datetime.now().isoformat(),
		}
		for key, value in metadata.items():
			row[f"gemini_{key}"] = value

	except Exception as e:
		error_str = str(e)
		if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
			console.print(f"✗ Gemini rate limit — aborting Gemini pass: {e}", style="bold red")
			raise Exception("Gemini rate limit exceeded — aborting Gemini translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using Gemini: {e}", style="bold red")
		row['gemini_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "gemini_translated_term": None},
			status_code=500,
			error_url=f"gemini.models.generate_content - {type(e).__name__}",
		)

	return row


def get_lingvanex_translation(row: pd.Series, error_file_path: str, console: Console,
							  LINGVANEX_AVAILABLE: bool) -> pd.Series:
	"""
	Translate a grouped row of terms using Lingvanex (via the translators library).

	Lingvanex ranked first among primary translation services in Kraus et al. (2025),
	outperforming Google Translate on most DH thesaurus metrics while being free.
	It is a grouped engine like GT and EasyNMT — one API call per language covering
	all terms.

	Parameters
	----------
	row : pd.Series
		Must have term_source (list of terms) and language_code.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	LINGVANEX_AVAILABLE : bool
		Whether the translators library is available

	Returns
	-------
	pd.Series
		Row with lingvanex_translated_term set to list of (term, translation) tuples.
	"""
	if not LINGVANEX_AVAILABLE:
		console.print("⚠ translators library not available — skipping Lingvanex", style="bold yellow")
		dh_terms = row.term_source
		row['lingvanex_translated_term'] = list(zip(dh_terms, [None] * len(dh_terms)))
		return row

	# Lingvanex uses slightly different codes for some languages
	_LINGVANEX_LANG_MAP = {'zh': 'zh-Hans', 'sr': 'sr-Cyrl'}

	time.sleep(1)
	try:
		dh_terms = row.term_source
		target_language = row.language_code
		lingvanex_code = _LINGVANEX_LANG_MAP.get(target_language, target_language)
		console.print(f"Translating {dh_terms} to {target_language} (Lingvanex)", style="bold green")

		translated_terms = []
		for term in dh_terms:
			try:
				result = ts_lib.translate_text(
					query_text=term,
					translator='lingvanex',
					to_language=lingvanex_code,
					from_language='en',
				)
				translated_terms.append(result)
			except Exception as term_err:
				console.print(f"⚠ Lingvanex failed for '{term}' → {target_language}: {term_err}", style="bold yellow")
				translated_terms.append(None)

		row['lingvanex_translated_term'] = list(zip(dh_terms, translated_terms))

	except Exception as e:
		console.print(f"✗ Lingvanex error for {row.term_source} → {row.language_code}: {e}", style="bold red")
		dh_terms = row.term_source
		target_language = row.language_code
		row['lingvanex_translated_term'] = list(zip(dh_terms, [None] * len(dh_terms)))
		log_error_to_file(
			error_file_path=error_file_path,
			additional_data={'term_source': dh_terms, 'language_code': target_language, 'lingvanex_translated_term': None},
			status_code=500,
			error_url=f"translators.translate_text(lingvanex, {row.language_code})",
		)

	return row


def get_ollama_translation(row: pd.Series, error_file_path: str, console: Console, current_prompt_variant: str,
							current_term_contexts: dict, ollama_model: str = 'llama3.1', request_delay: float = 2.0,
							request_timeout: int = 120, consecutive_timeouts: int = 0) -> Tuple[pd.Series, int]:
	"""
	Function to get translations of terms using a local Ollama model with throttling.

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	current_prompt_variant : str
		The prompt variant to use ('comparative' or other)
	current_term_contexts : dict
		Dictionary mapping terms to their contexts
	ollama_model : str
		The Ollama model to use (default: 'llama3.1', previously 'llama3.2')
	request_delay : float
		Delay in seconds between Ollama requests to prevent overwhelming the service (default: 2.0)
	request_timeout : int
		Maximum seconds to wait for Ollama response before timing out (default: 120)
	consecutive_timeouts : int
		Running count of consecutive timeouts passed in by the caller (default: 0)

	Returns
	-------
	Tuple[pd.Series, int]
		A tuple of (row with translated terms, updated consecutive_timeouts count)
	"""

	# Throttle requests to prevent Ollama from getting overwhelmed
	time.sleep(request_delay)
	try:
		term_source = row.term_source
		language_name = row.language_name
		language_code = row.language_code
		gt_translated_term = row.gt_translated_term if 'gt_translated_term' in row else None
		enmt_translated_term = row.enmt_translated_term if 'enmt_translated_term' in row else None  # Fixed typo: was 'emnt_translated_term'
		openai_translated_term = row.openai_translated_term if 'openai_translated_term' in row else None
		wikipedia_translated_term = row.wikipedia_translated_term if 'wikipedia_translated_term' in row else None
		first_ollama_translated_term = row.first_ollama_translated_term if 'first_ollama_translated_term' in row else None

		console.print(f"Translating {term_source} to {language_name} using {ollama_model} (Ollama)", style="bold green")

		# Build prompt, passing existing translations if using the comparative variant
		existing_translations = None
		if current_prompt_variant == 'comparative':
			existing_translations = {
				'gt': gt_translated_term if pd.notna(gt_translated_term) else None,
				'enmt': enmt_translated_term if pd.notna(enmt_translated_term) else None,
				'lingvanex': row.lingvanex_translated_term if 'lingvanex_translated_term' in row and pd.notna(row.lingvanex_translated_term) else None,
				'openai': openai_translated_term if pd.notna(openai_translated_term) else None,
				'claude': row.claude_translated_term if 'claude_translated_term' in row and pd.notna(row.claude_translated_term) else None,
				'gemini': row.gemini_translated_term if 'gemini_translated_term' in row and pd.notna(row.gemini_translated_term) else None,
				'ollama': first_ollama_translated_term if pd.notna(first_ollama_translated_term) else None,
				'wikipedia': wikipedia_translated_term if pd.notna(wikipedia_translated_term) else None,
			}

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
			context=current_term_contexts.get(term_source),
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		console.print(f"Initial prompt is {user_content}", style="bright_cyan")
		console.print(f"Using Ollama model: {ollama_model} (timeout: {request_timeout}s)", style="bold cyan")

		# Make Ollama call with timeout using threading
		ollama_response = None
		timeout_error = None

		def call_ollama():
			nonlocal ollama_response, timeout_error
			try:
				ollama_response = ollama.chat(
					model=ollama_model,
					messages=[
						{
							'role': 'system',
							'content': f"You are a {term_source} scholar who speaks many languages."
						},
						{
							'role': 'user',
							'content': user_content
						}
					]
				)
			except Exception as e:
				timeout_error = e

		# Run Ollama call in a thread with timeout
		ollama_thread = threading.Thread(target=call_ollama, daemon=True)
		ollama_thread.start()
		ollama_thread.join(timeout=request_timeout)

		# Check if thread is still alive (timed out)
		if ollama_thread.is_alive():
			consecutive_timeouts += 1
			console.print(f"⚠ Ollama request timed out after {request_timeout}s - skipping ({consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS})", style="bold yellow")

			if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
				console.print(f"✗ Too many consecutive Ollama timeouts ({MAX_CONSECUTIVE_TIMEOUTS}). Stopping to save progress.", style="bold red")
				raise Exception(f"Ollama service unresponsive - stopping after {MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts")

			row['ollama_translated_term'] = None
			row['ollama_translation_rationale'] = None
			log_error_to_file(error_file_path, additional_data={"term_source": term_source, "language_code": language_code, "ollama_translated_term": None}, status_code=408, error_url="ollama.chat - Request Timeout")
			return row, consecutive_timeouts

		if timeout_error:
			console.print(f"⚠ Ollama request failed: {timeout_error}", style="bold yellow")
			row['ollama_translated_term'] = None
			row['ollama_translation_rationale'] = None
			log_error_to_file(error_file_path, additional_data={"term_source": term_source, "language_code": language_code, "ollama_translated_term": None}, status_code=500, error_url=f"ollama.chat - {str(timeout_error)}")
			return row, consecutive_timeouts

		if ollama_response is None:
			console.print(f"⚠ No response from Ollama", style="bold yellow")
			row['ollama_translated_term'] = None
			row['ollama_translation_rationale'] = None
			return row, consecutive_timeouts

		console.print(f"Ollama response: {ollama_response}", style="bright_magenta")
		message_content = ollama_response['message']['content']
		row['ollama_content'] = message_content
		row['ollama_model'] = ollama_response['model']
		row['ollama_created_at'] = ollama_response['created_at']
		row['ollama_total_duration'] = ollama_response['total_duration']
		row['ollama_load_duration'] = ollama_response['load_duration']
		row['ollama_prompt_eval_count'] = ollama_response['prompt_eval_count']
		row['ollama_prompt_eval_duration'] = ollama_response['prompt_eval_duration']
		row['ollama_eval_count'] = ollama_response['eval_count']
		row['ollama_eval_duration'] = ollama_response['eval_duration']
		row['ollama_done_reason'] = ollama_response['done_reason']

		try:
			response = parse_translation_response(message_content)
			row['ollama_translated_term'] = response.translated_term
			row['ollama_translation_rationale'] = response.translation_rationale
			# Reset timeout counter on successful translation
			consecutive_timeouts = 0
			console.print(f"Llama translation: {row.ollama_translated_term}", style="bold green")

		except (KeyError, ValueError, SyntaxError, ValidationError) as e:
			console.print(f"Error parsing Ollama response: {e}", style="bold red")
			row['ollama_translated_term'] = None
			row['ollama_translation_rationale'] = None
			log_error_to_file(
				error_file_path,
				additional_data={
					"term_source": term_source,
					"language_code": language_code,
					"ollama_translated_term": None
				},
				status_code=400,
				error_url="ollama.chat - Parsing Error"
			)

	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from Llama: {e}", style="bold red")
		row['ollama_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"ollama_translated_term": None
			},
			status_code=400,
			error_url="ollama.chat - JSONDecodeError"
		)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row['ollama_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"ollama_translated_term": None
			},
			status_code=500,
			error_url="ollama.chat - General Exception"
		)

	return row, consecutive_timeouts

def check_if_wikipedia_page_exists(term_source: str, error_file_path: str, console: Console) -> dict:
	"""
	Check if a Wikipedia page exists for a given term.

	Parameters
	----------
	term_source : str
		The term to check for in Wikipedia.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output

	Returns
	-------
	dict
		A dictionary with translations of the term.
	"""
	# Currently hardcoding for english wikipedia
	wiki_wiki = wikipediaapi.Wikipedia(
		language='en',
		user_agent='MyProject/1.0 (https://example.org/myproject/; myemail@example.org)'
	)
	# Fetch the Wikipedia page
	term_source = term_source.lower()
	page = wiki_wiki.page(term_source)

	if not page.exists():
		console.print(f"Page '{term_source}' does not exist in English Wikipedia.", style="bold red")
		# Log structured error
		additional_data = {
			'term_source': term_source,
			'language_code': 'en',
			'wikipedia_translated_term': None
		}
		log_error_to_file(
			error_file_path=error_file_path,
			additional_data=additional_data,
			status_code=404,
			error_url=f"wikipediaapi.Wikipedia.page({term_source})"
		)
		return {}

	console.print(f"Page '{term_source}' exists in English Wikipedia.", style="bold green")
	# Get available translations in other languages
	translations = page.langlinks
	console.print(f"Has this number of {len(translations)} translated pages", style="bold green")
	keep_page = console.input("Do you want to keep this page? (y/n): ")
	if keep_page == 'n':
		console.print(f"Page '{term_source}' will not be kept.", style="bold red")
		# Log structured error
		additional_data = {
			'term_source': term_source,
			'language_code': 'en',
			'wikipedia_translated_term': None
		}
		log_error_to_file(
			error_file_path=error_file_path,
			additional_data=additional_data,
			status_code=404,
			error_url=f"wikipediaapi.Wikipedia.page({term_source})"
		)
		return {}
	else:
		translation_dict = {lang: translations[lang].title for lang in translations}
		translation_dict['en'] = term_source
		return translation_dict

