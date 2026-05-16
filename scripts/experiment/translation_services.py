"""
Translation service wrappers for multiple translation APIs. Includes Google Translate, EasyNMT, OpenAI, Claude, Gemini, LingVanex, Ollama, and Wikipedia.
"""

import json
import time
import threading
from typing import Tuple


class OllamaUnresponsiveError(BaseException):
    """Raised when Ollama hits MAX_CONSECUTIVE_TIMEOUTS; inherits BaseException so
    it propagates through all `except Exception` handlers without being swallowed."""
import pandas as pd
from rich.console import Console
from openai import OpenAI, OpenAIError
from anthropic import Anthropic, APIError
import datetime
import wikipediaapi
import ollama
from pydantic import ValidationError
import apikey
from easynmt import EasyNMT
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from google import genai as google_genai

import arabic_reshaper
from bidi.algorithm import get_display

from data_processing import parse_translation_response, is_enmt_model_available, TranslationRefusedError
from scripts.utils import log_error_to_file
from translation_prompts import get_prompt, get_system_prompt
import google.genai.types as google_genai_types
import translators as ts_lib

def _display_text(text: str) -> str:
    """Reshape and apply bidi algorithm so RTL text (Arabic, Hebrew, etc.) renders correctly in terminals."""
    if text is None:
        return ''
    try:
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except Exception:
        return str(text)


def _clean_translation(val):
    """Return None for NaN/null/empty values so they are skipped in context-aware prompts."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return None if s in ('nan', 'None', '') else s


def _build_existing_translations(row, prompt_variant: str, term_contexts: dict) -> dict:
    """
    Build the existing_translations dict passed to the judge prompt variant.

    judge — receives the pre-aggregated unique-translation structure built by
            generate_translation_prompts.aggregate_variant_translations().
            Stored in term_contexts as {term_source: {language_code: ctx}}.
    """
    if prompt_variant == 'judge':
        per_lang = term_contexts.get(row.get('term_source'), {})
        if isinstance(per_lang, dict):
            return per_lang.get(row.get('language_code'))
    return None


# Constants
MAX_CONSECUTIVE_TIMEOUTS = 3

# ── Client Initialization ────────────────────────────────────────────────────────────────

# Initialize console
console = Console()

# Load Google Cloud Translate credentials
google_translate_key_path = apikey.load("GOOGLE_TRANSLATE_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(
	google_translate_key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
translate_client = translate.Client(credentials=credentials)

# Load OpenAI credentials
openai_key_path = apikey.load("CODING_DH_OPENAI_KEY")
openai_project_id = apikey.load("CODING_DH_OPENAI_PROJECT_ID")
openai_organization_id = apikey.load("CODING_DH_OPENAI_ORGANIZATION_ID")
client = OpenAI(
	api_key=openai_key_path,
	project=openai_project_id,
	organization=openai_organization_id
)

# Load Claude API credentials
claude_api_key = apikey.load("CODING_DH_CLAUDE_KEY")
claude_client = Anthropic(api_key=claude_api_key)

# Load Gemini credentials (optional — only used if CODING_DH_GEMINI_KEY is set)
gemini_client = None
GEMINI_MODEL = "gemini-2.5-flash"
gemini_api_key = apikey.load("CODING_DH_GEMINI_KEY")
gemini_client = google_genai.Client(api_key=gemini_api_key)

# Load DeepSeek credentials (OpenAI-compatible API)
DEEPSEEK_MODEL = "deepseek-chat"  # points to DeepSeek-V3
deepseek_api_key = apikey.load("CODING_DH_DEEPSEEK_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Pinned Ollama model tags for reproducibility
OLLAMA_LLAMA_MODEL   = "llama3.1:latest"
OLLAMA_GEMMA_MODEL   = "gemma3:12b"
OLLAMA_QWEN_MODEL    = "qwen2.5:7b"
OLLAMA_MISTRAL_MODEL = "mistral:7b"

# Initialize the EasyNMT model once
model = EasyNMT('opus-mt')


# ── Wikipedia Availability Checker ────────────────────────────────────────────────────
# Note: Wikipedia is different from other services — it checks for page existence rather than performing translations. Grouped separately.

def check_if_wikipedia_page_exists(term_source: str, error_file_path: str, console: Console) -> dict:
	"""
	Check if a Wikipedia page exists for a given term. We use Wikipedia as a source of ground truth translations where available, and also to test the ability of LLMs to match Wikipedia's translations.

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


# ── Direct Translation Services ──────────────────────────────────────────────────
# Services that directly translate terms without using prompts. These are prompt-invariant baselines used to evaluate the added value of LLM-based translation and as context input for the judge prompt variant.

def get_gt_translation(row: pd.Series, error_file_path: str, console: Console) -> pd.Series:
	"""
	Function to get translations of terms from Google Translate API. This function requires you to have an api key for Google Cloud Translation API and to set up the client accordingly.

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output

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

		# Check if it's an unsupported language error.
		# Google Cloud API errors include the HTTP status at the start of the string
		# (e.g. "400 POST https://..."), and unsupported language codes come back as
		# "Invalid Value" with a fieldViolations entry for the target field.
		if ('Bad language pair' in error_str or 'badRequest' in error_str.lower()
				or error_str.startswith('400') or 'fieldViolations' in error_str):
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
			error_url=f"translate_client.translate({target_language}) - {error_msg}",
			error_message=error_msg,
		)

	return row

def get_enmt_translation(row: pd.Series, error_file_path: str, console: Console) -> pd.Series:
	"""
	Function to get translations of terms using EasyNMT. Checks if a model is available for the target language before attempting translation.

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
			error_url=f"model.translate({target_language})",
			error_message=str(e),
		)


	return row

def get_lingvanex_translation(row: pd.Series, error_file_path: str, console: Console) -> pd.Series:
	"""
	Translate a grouped row of terms using Lingvanex (via the translators library).

	Lingvanex ranked first among primary translation services in Kraus et al. (2025), outperforming Google Translate on most DH thesaurus metrics while being free. It is a grouped engine like GT and EasyNMT — one API call per language covering all terms.

	Parameters
	----------
	row : pd.Series
		Must have term_source (list of terms) and language_code.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output

	Returns
	-------
	pd.Series
		Row with lingvanex_translated_term set to list of (term, translation) tuples.
	"""
	# Full mapping from ISO 639-1 codes to Lingvanex locale codes.
	# Languages absent from this map are not supported by Lingvanex and are skipped silently.
	_LINGVANEX_LANG_MAP = {
		'af': 'af_ZA', 'am': 'am_ET', 'ar': 'ar_EG', 'as': 'as_IN', 'az': 'az_AZ',
		'be': 'be_BY', 'bg': 'bg_BG', 'bn': 'bn_BD', 'bs': 'bs_BA', 'ca': 'ca_ES',
		'ceb': 'ceb_PH', 'co': 'co_FR', 'cs': 'cs_CZ', 'cy': 'cy_GB', 'da': 'da_DK',
		'de': 'de_DE', 'el': 'el_GR', 'en': 'en_US', 'eo': 'eo_WORLD', 'es': 'es_ES',
		'et': 'et_EE', 'eu': 'eu_ES', 'fa': 'fa_IR', 'fi': 'fi_FI', 'fr': 'fr_FR',
		'fy': 'fy_NL', 'ga': 'ga_IE', 'gd': 'gd_GB', 'gl': 'gl_ES', 'gu': 'gu_IN',
		'ha': 'ha_NE', 'haw': 'haw_US', 'he': 'he_IL', 'hi': 'hi_IN', 'hmn': 'hmn_CN',
		'hr': 'hr_HR', 'ht': 'ht_HT', 'hu': 'hu_HU', 'hy': 'hy_AM', 'id': 'id_ID',
		'ig': 'ig_NG', 'is': 'is_IS', 'it': 'it_IT', 'ja': 'ja_JP', 'jv': 'jv_ID',
		'ka': 'ka_GE', 'kk': 'kk_KZ', 'km': 'km_KH', 'kn': 'kn_IN', 'ko': 'ko_KR',
		'ku': 'ku_IR', 'ky': 'ky_KG', 'la': 'la_VAT', 'lb': 'lb_LU', 'lo': 'lo_LA',
		'lt': 'lt_LT', 'lv': 'lv_LV', 'mg': 'mg_MG', 'mi': 'mi_NZ', 'mk': 'mk_MK',
		'ml': 'ml_IN', 'mn': 'mn_MN', 'mr': 'mr_IN', 'ms': 'ms_MY', 'mt': 'mt_MT',
		'my': 'my_MM', 'ne': 'ne_NP', 'nl': 'nl_NL', 'no': 'no_NO', 'ny': 'ny_MW',
		'or': 'or_OR', 'pa': 'pa_PK', 'pl': 'pl_PL', 'ps': 'ps_AF', 'pt': 'pt_PT',
		'ro': 'ro_RO', 'ru': 'ru_RU', 'rw': 'rw_RW', 'sd': 'sd_PK', 'si': 'si_LK',
		'sk': 'sk_SK', 'sl': 'sl_SI', 'sm': 'sm_WS', 'sn': 'sn_ZW', 'so': 'so_SO',
		'sq': 'sq_AL', 'sr': 'sr-Cyrl_RS', 'st': 'st_LS', 'su': 'su_ID', 'sv': 'sv_SE',
		'sw': 'sw_TZ', 'ta': 'ta_IN', 'te': 'te_IN', 'tg': 'tg_TJ', 'th': 'th_TH',
		'tk': 'tk_TM', 'tl': 'tl_PH', 'tr': 'tr_TR', 'tt': 'tt_TT', 'ug': 'ug_CN',
		'uk': 'uk_UA', 'ur': 'ur_PK', 'uz': 'uz_UZ', 'vi': 'vi_VN', 'xh': 'xh_ZA',
		'yi': 'yi_IL', 'yo': 'yo_NG', 'zh': 'zh-Hans_CN', 'zh-tw': 'zh-Hant_TW',
		'zu': 'zu_ZA',
	}

	time.sleep(1)
	try:
		dh_terms = row.term_source
		target_language = row.language_code
		lingvanex_code = _LINGVANEX_LANG_MAP.get(target_language)
		if lingvanex_code is None:
			row['lingvanex_translated_term'] = list(zip(dh_terms, [None] * len(dh_terms)))
			log_error_to_file(
				error_file_path=error_file_path,
				additional_data={'term_source': dh_terms, 'language_code': target_language, 'lingvanex_translated_term': None},
				status_code=400,
				error_url=f"lingvanex.unsupported_language.{target_language}",
			)
			return row
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
				term_err_str = str(term_err)
				term_status = 400 if ('400' in term_err_str or 'Bad Request' in term_err_str) else 500
				console.print(f"⚠ Lingvanex failed for '{term}' → {target_language}: {term_err}", style="bold yellow")
				translated_terms.append(None)
				log_error_to_file(
					error_file_path=error_file_path,
					additional_data={'term_source': dh_terms, 'language_code': target_language, 'lingvanex_translated_term': None},
					status_code=term_status,
					error_url=f"translators.translate_text(lingvanex, {target_language})",
					error_message=term_err_str,
				)

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
			error_message=str(e),
		)

	return row


# ── LLM Translation Services ─────────────────────────────────────────────────────
# Services that use prompts and Large Language Models to perform translations

def get_openai_translation(row: pd.Series, error_file_path: str, console: Console, current_prompt_variant: str, current_term_contexts: dict) -> pd.Series:
	"""
	Function to get translations of terms using the OpenAI API. This function requires you to have an API key for OpenAI and to set up the client accordingly. It also uses the prompt variants defined in translation_prompts.py to test different strategies for improving translation quality.

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
		The prompt variant to use ('minimal', 'expert_persona', 'native_rationale', 'judge')
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

		console.print(f"Translating {term_source} to {language_name} using OpenAI API", style="bold green")

		existing_translations = _build_existing_translations(row, current_prompt_variant, current_term_contexts)

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		# System prompt is variant-dependent; minimal returns None to mean
		# "send no system message at all" (not an empty string, which some
		# providers treat as a real signal).
		openai_system_content = get_system_prompt(current_prompt_variant, term_source)
		if openai_system_content is None:
			prompt_messages = [{"role": "user", "content": user_content}]
		else:
			prompt_messages = [
				{"role": "system", "content": openai_system_content},
				{"role": "user", "content": user_content}
			]
		console.print(f"Prompt messages: {json.dumps(prompt_messages, indent=2)}", style="bold yellow")

		# Save prompts to row for audit trail
		row['openai_system_prompt'] = openai_system_content
		row['openai_user_prompt'] = user_content

		# Call the OpenAI Chat API directly using the `openai` module
		chat_completion = client.chat.completions.create(
			model="gpt-4o",
			messages=prompt_messages,
			temperature=0,
			top_p=1,
			max_tokens=2048,
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
	except TranslationRefusedError as e:
		console.print(f"⚠ OpenAI declined to translate {term_source} → {language_name}: {e.rationale}", style="bold yellow")
		row['openai_translated_term'] = None
		row['openai_translation_rationale'] = e.rationale
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "openai_translated_term": None},
			status_code=400,
			error_url=f"openai.chat.completions.create - TranslationRefused",
			error_message=e.rationale,
		)
	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from OpenAI: {e}", style="bold red")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"openai_translated_term": None
			},
			status_code=400,
			error_url="openai.chat.completions.create - JSONDecodeError",
			error_message=str(e),
		)
	except ValueError as e:
		# parse_translation_response raised ValueError — model returned prose instead of JSON.
		# Deterministic (retrying won't help), so log as 400 to skip on subsequent runs.
		console.print(f"⚠ OpenAI parse failure for {term_source} → {language_name}: {e}", style="bold yellow")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"openai_translated_term": None
			},
			status_code=400,
			error_url="openai.chat.completions.create - ParseError",
			error_message=str(e),
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
					"variant": current_prompt_variant,
					"openai_translated_term": None
				},
				status_code=429,
				error_url="openai.chat.completions.create - QuotaExceeded",
				error_message=error_str,
			)
			raise Exception("OpenAI quota exceeded — aborting OpenAI translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using OpenAI: {e}", style="bold red")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"openai_translated_term": None
			},
			status_code=500,
			error_url="openai.chat.completions.create - OpenAIError",
			error_message=error_str,
		)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row['openai_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"openai_translated_term": None
			},
			status_code=500,
			error_url="openai.chat.completions.create - General Exception",
			error_message=str(e),
		)

	return row


def get_deepseek_translation(row: pd.Series, error_file_path: str, console: Console, current_prompt_variant: str, current_term_contexts: dict) -> pd.Series:
	"""
	Translate a single row using the DeepSeek API (OpenAI-compatible).

	Uses DEEPSEEK_MODEL ("deepseek-chat", which points to DeepSeek-V3).
	Mirrors get_openai_translation() exactly; columns are prefixed 'deepseek_'.
	"""
	time.sleep(1)
	try:
		term_source = row.term_source
		language_name = row.language_name
		language_code = row.language_code

		console.print(f"Translating {term_source} to {language_name} using DeepSeek API", style="bold green")

		existing_translations = _build_existing_translations(row, current_prompt_variant, current_term_contexts)

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		deepseek_system_content = get_system_prompt(current_prompt_variant, term_source)
		if deepseek_system_content is None:
			prompt_messages = [{"role": "user", "content": user_content}]
		else:
			prompt_messages = [
				{"role": "system", "content": deepseek_system_content},
				{"role": "user", "content": user_content}
			]
		console.print(f"Prompt messages: {json.dumps(prompt_messages, indent=2)}", style="bold yellow")

		row['deepseek_system_prompt'] = deepseek_system_content
		row['deepseek_user_prompt'] = user_content

		chat_completion = deepseek_client.chat.completions.create(
			model=DEEPSEEK_MODEL,
			messages=prompt_messages,
			temperature=0,
			top_p=1,
			max_tokens=2048,
		)
		console.print(f"Chat completion: {chat_completion}", style="bright_cyan")
		message_content = chat_completion.choices[0].message.content
		response = parse_translation_response(message_content)

		metadata = {
			'translated_term':      response.translated_term,
			'translation_rationale': response.translation_rationale,
			'model':               chat_completion.model,
			'finish_reason':       chat_completion.choices[0].finish_reason,
			'created':             chat_completion.created,
			'total_tokens':        chat_completion.usage.total_tokens,
			'prompt_tokens':       chat_completion.usage.prompt_tokens,
			'completion_tokens':   chat_completion.usage.completion_tokens,
		}
		for key, value in metadata.items():
			row["deepseek_" + key] = value

	except TranslationRefusedError as e:
		console.print(f"⚠ DeepSeek declined to translate {term_source} → {language_name}: {e.rationale}", style="bold yellow")
		row['deepseek_translated_term'] = None
		row['deepseek_translation_rationale'] = e.rationale
		log_error_to_file(error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "deepseek_translated_term": None},
			status_code=400, error_url="deepseek.chat.completions.create - TranslationRefused", error_message=e.rationale)
	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from DeepSeek: {e}", style="bold red")
		row['deepseek_translated_term'] = None
		log_error_to_file(error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "deepseek_translated_term": None},
			status_code=400, error_url="deepseek.chat.completions.create - JSONDecodeError", error_message=str(e))
	except ValueError as e:
		console.print(f"⚠ DeepSeek parse failure for {term_source} → {language_name}: {e}", style="bold yellow")
		row['deepseek_translated_term'] = None
		log_error_to_file(error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "deepseek_translated_term": None},
			status_code=400, error_url="deepseek.chat.completions.create - ParseError", error_message=str(e))
	except OpenAIError as e:
		error_str = str(e)
		if 'insufficient_quota' in error_str or 'quota' in error_str.lower() or getattr(e, 'status_code', None) == 429:
			console.print(f"✗ DeepSeek quota exceeded — aborting DeepSeek translation pass: {e}", style="bold red")
			log_error_to_file(error_file_path,
				additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "deepseek_translated_term": None},
				status_code=429, error_url="deepseek.chat.completions.create - QuotaExceeded", error_message=error_str)
			raise Exception("DeepSeek quota exceeded — aborting DeepSeek translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using DeepSeek: {e}", style="bold red")
		row['deepseek_translated_term'] = None
		log_error_to_file(error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "deepseek_translated_term": None},
			status_code=500, error_url="deepseek.chat.completions.create - OpenAIError", error_message=error_str)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row['deepseek_translated_term'] = None
		log_error_to_file(error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "deepseek_translated_term": None},
			status_code=500, error_url="deepseek.chat.completions.create - General Exception", error_message=str(e))

	return row


def get_claude_translation(row: pd.Series, error_file_path: str, console: Console, current_prompt_variant: str, current_term_contexts: dict) -> pd.Series:
	"""
	Function to get translations of terms using the Claude API. This function requires you to have an API key for Anthropic and to set up the client accordingly. It also uses the prompt variants defined in translation_prompts.py to test different strategies for improving translation quality.

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
		The prompt variant to use ('minimal', 'expert_persona', 'native_rationale', 'judge')
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

		console.print(f"Translating {term_source} to {language_name} using Claude API", style="bold green")

		existing_translations = _build_existing_translations(row, current_prompt_variant, current_term_contexts)

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")

		system_content = get_system_prompt(current_prompt_variant, term_source)
		console.print(f"Claude user prompt: {user_content}", style="bold yellow")

		# Save prompts to row for audit trail
		row['claude_system_prompt'] = system_content
		row['claude_user_prompt'] = user_content

		# Call the Claude API. Omit `system=` entirely for the minimal variant
		# rather than passing an empty string — Anthropic's API treats empty
		# strings as a real signal.
		claude_create_kwargs = dict(
			model="claude-sonnet-4-5",
			max_tokens=2048,
			temperature=0,
			messages=[{"role": "user", "content": user_content}]
		)
		if system_content is not None:
			claude_create_kwargs['system'] = system_content
		message = claude_client.messages.create(**claude_create_kwargs)
		console.print(f"Claude response: {message}", style="bright_cyan")

		if message.stop_reason == 'max_tokens':
			raise ValueError(
				f"Claude response truncated (max_tokens) for {term_source} → {language_name}. "
				"Response JSON will be incomplete."
			)

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
	except TranslationRefusedError as e:
		console.print(f"⚠ Claude declined to translate {term_source} → {language_name}: {e.rationale}", style="bold yellow")
		row['claude_translated_term'] = None
		row['claude_translation_rationale'] = e.rationale
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "claude_translated_term": None},
			status_code=400,
			error_url=f"claude.messages.create - TranslationRefused",
			error_message=e.rationale,
		)
	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from Claude: {e}", style="bold red")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"claude_translated_term": None
			},
			status_code=400,
			error_url="claude.messages.create - JSONDecodeError",
			error_message=str(e),
		)
	except ValueError as e:
		console.print(f"⚠ Claude parse failure for {term_source} → {language_name}: {e}", style="bold yellow")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"claude_translated_term": None
			},
			status_code=400,
			error_url="claude.messages.create - ParseError",
			error_message=str(e),
		)
	except APIError as e:
		error_str = str(e)
		status_code = getattr(e, 'status_code', 500)
		# 404 means the model name is wrong/retired — infrastructure problem, not per-language.
		if status_code == 404 or 'not_found_error' in error_str:
			console.print(f"✗ Claude model not found — check model string in get_claude_translation: {e}", style="bold red")
			raise Exception("Claude model not found — aborting Claude translation pass") from e
		# Billing failure — no point continuing, every subsequent call will fail the same way.
		if 'credit balance is too low' in error_str or 'insufficient_funds' in error_str:
			console.print(f"✗ Claude billing error — out of credits. Stopping Claude translation pass.", style="bold red")
			raise Exception("Claude billing error — aborting Claude translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using Claude: {e}", style="bold red")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"claude_translated_term": None
			},
			status_code=status_code,
			error_url="claude.messages.create - APIError",
			error_message=error_str,
		)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row['claude_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				"claude_translated_term": None
			},
			status_code=500,
			error_url="claude.messages.create - General Exception",
			error_message=str(e),
		)

	return row

def _fetch_gemini_rationale(term_source: str, language_name: str, salvaged_term: str, console: Console) -> str:
	"""
	Fetch only the translation rationale for a term that was already translated but whose
	rationale was cut off due to MAX_TOKENS in the primary call. Uses a short, focused prompt
	so the output stays well within token limits.

	Returns the rationale string, or a failure notice if the follow-up call also fails.
	"""
	prompt = (
		f"You previously translated the Digital Humanities term '{term_source}' into {language_name} "
		f"as '{salvaged_term}'. In 2–3 sentences, explain your reasoning for this translation choice."
	)
	try:
		response = gemini_client.models.generate_content(
			model=GEMINI_MODEL,
			contents=prompt,
			config=google_genai_types.GenerateContentConfig(
				system_instruction=f"You are a {term_source} scholar who speaks many languages.",
				temperature=0,
				top_p=1.0,
				max_output_tokens=2048,
				thinking_config=google_genai_types.ThinkingConfig(thinking_budget=0),
				http_options=google_genai_types.HttpOptions(timeout=60_000),
			),
		)
		finish_reason = response.candidates[0].finish_reason.name if response.candidates else None
		if finish_reason == 'MAX_TOKENS':
			console.print(f"⚠ Gemini rationale follow-up also hit MAX_TOKENS for {language_name}.", style="bold yellow")
			return "[rationale unavailable — follow-up call also hit MAX_TOKENS]"
		return response.text.strip()
	except Exception as e:
		console.print(f"⚠ Gemini rationale follow-up failed for {language_name}: {e}", style="bold yellow")
		return f"[rationale unavailable — follow-up call failed: {type(e).__name__}]"

def get_gemini_translation(row: pd.Series, error_file_path: str, console: Console, current_prompt_variant: str, current_term_contexts: dict) -> pd.Series:
	"""
	Translate a single row using the Google Gemini API (gemini-2.0-flash by default). This function requires you to have an API key for Google Cloud and to set up the Gemini client accordingly. It also uses the prompt variants defined in translation_prompts.py to test different strategies for improving translation quality.

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
	GEMINI_MODEL : str
		The Gemini model name to use
	current_prompt_variant : str
		The prompt variant to use ('minimal', 'expert_persona', 'native_rationale', 'judge')
	current_term_contexts : dict
		Dictionary mapping terms to their contexts

	Returns
	-------
	pd.Series
		Row with gemini_* columns populated.
	"""
	if gemini_client is None:
		console.print("⚠ Gemini client not initialised — check CODING_DH_GEMINI_KEY. Skipping row.", style="bold yellow")
		row['gemini_translated_term'] = None
		return row

	time.sleep(1)
	try:
		term_source = row.term_source
		language_name = row.language_name
		language_code = row.language_code

		console.print(f"Translating {term_source} to {language_name} using Gemini API ({GEMINI_MODEL})", style="bold green")

		existing_translations = _build_existing_translations(row, current_prompt_variant, current_term_contexts)

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		console.print(f"Gemini user prompt: {user_content}", style="bold yellow")

		system_content = get_system_prompt(current_prompt_variant, term_source)

		# Save prompts to row for audit trail
		row['gemini_system_prompt'] = system_content
		row['gemini_user_prompt'] = user_content

		gemini_config_kwargs = dict(
			temperature=0,
			top_p=1.0,
			max_output_tokens=2048,
			thinking_config=google_genai_types.ThinkingConfig(thinking_budget=0),
			http_options=google_genai_types.HttpOptions(timeout=60_000),
		)
		# Omit system_instruction entirely for minimal (None), rather than
		# passing an empty string.
		if system_content is not None:
			gemini_config_kwargs['system_instruction'] = system_content
		response = gemini_client.models.generate_content(
			model=GEMINI_MODEL,
			contents=user_content,
			config=google_genai_types.GenerateContentConfig(**gemini_config_kwargs),
		)
		console.print(f"Gemini response: {response}", style="bright_cyan")

		message_content = response.text
		finish_reason = response.candidates[0].finish_reason.name if response.candidates else None

		if finish_reason == 'MAX_TOKENS':
			# Rationale was truncated (often a repetition loop on rare/extinct languages).
			# Salvage the translated_term from the partial JSON, then fetch the rationale
			# via a short follow-up call so the record stays meaningful.
			import re as _re
			_m = _re.search(r'"translated_term"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', message_content)
			if _m:
				salvaged = _m.group(1)
				if len(salvaged) > 300:
					# Hallucination loop — model generated a wall of characters, not a real translation.
					console.print(f"⚠ Gemini MAX_TOKENS for {language_name} — salvaged term is suspiciously long ({len(salvaged)} chars), discarding.", style="bold yellow")
					raise ValueError(f"MAX_TOKENS hallucination loop for {language_name} — salvaged term too long to be valid")
				console.print(f"⚠ Gemini MAX_TOKENS for {language_name} — term salvaged, fetching rationale via follow-up call.", style="bold yellow")
				followup_rationale = _fetch_gemini_rationale(
					term_source=term_source,
					language_name=language_name,
					salvaged_term=salvaged,
					console=console,
				)
				metadata = {
					'translated_term': salvaged,
					'translation_rationale': followup_rationale,
					'model': GEMINI_MODEL,
					'created': datetime.datetime.now().isoformat(),
					'rationale_from_followup': True,
				}
				for key, value in metadata.items():
					row[f"gemini_{key}"] = value
				return row
			else:
				raise ValueError(f"MAX_TOKENS and could not salvage translated_term for {language_name}")

		parsed = parse_translation_response(message_content)

		metadata = {
			'translated_term': parsed.translated_term,
			'translation_rationale': parsed.translation_rationale,
			'model': GEMINI_MODEL,
			'created': datetime.datetime.now().isoformat(),
		}
		for key, value in metadata.items():
			row[f"gemini_{key}"] = value

	except TranslationRefusedError as e:
		console.print(f"⚠ Gemini declined to translate {term_source} → {language_name}: {e.rationale}", style="bold yellow")
		row['gemini_translated_term'] = None
		row['gemini_translation_rationale'] = e.rationale
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "gemini_translated_term": None},
			status_code=400,
			error_url=f"gemini.models.generate_content - TranslationRefused",
			error_message=e.rationale,
		)
	except ValueError as e:
		console.print(f"⚠ Gemini parse failure for {term_source} → {language_name}: {e}", style="bold yellow")
		row['gemini_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "gemini_translated_term": None},
			status_code=400,
			error_url=f"gemini.models.generate_content - ParseError",
			error_message=str(e),
		)
	except Exception as e:
		error_str = str(e)
		if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
			console.print(f"✗ Gemini rate limit — aborting Gemini pass: {e}", style="bold red")
			raise Exception("Gemini rate limit exceeded — aborting Gemini translation pass") from e
		console.print(f"Error translating {term_source} to {language_name} using Gemini: {e}", style="bold red")
		row['gemini_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, "gemini_translated_term": None},
			status_code=500,
			error_url=f"gemini.models.generate_content - {type(e).__name__}",
			error_message=str(e),
		)

	return row


def get_ollama_translation(row: pd.Series, error_file_path: str, console: Console, current_prompt_variant: str, current_term_contexts: dict, ollama_model: str = 'llama3.1', request_delay: float = 2.0, request_timeout: int = 120, consecutive_timeouts: int = 0, col_prefix: str = 'ollama') -> Tuple[pd.Series, int]:
	"""
	Function to get translations of terms using a local Ollama model with throttling. This function assumes you have Ollama set up locally with the specified model and that the Ollama server is running. It also uses the prompt variants defined in translation_prompts.py to test different strategies for improving translation quality.

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with terms and a language.
	error_file_path : str
		A path to the error file for logging errors
	console : Console
		Rich console for printing output
	current_prompt_variant : str
		The prompt variant to use ('minimal', 'expert_persona', 'native_rationale', 'judge')
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

		console.print(f"Translating {term_source} to {language_name} using {ollama_model} (Ollama)", style="bold green")

		existing_translations = _build_existing_translations(row, current_prompt_variant, current_term_contexts)

		user_content = get_prompt(
			variant=current_prompt_variant,
			term_source=term_source,
			language_name=language_name,
			language_code=language_code,
			existing_translations=existing_translations,
		)
		console.print(f"Using prompt variant: {current_prompt_variant}", style="bold cyan")
		console.print(f"Initial prompt is {_display_text(user_content)}", style="bright_cyan")
		console.print(f"Using Ollama model: {ollama_model} (timeout: {request_timeout}s)", style="bold cyan")

		system_content = get_system_prompt(current_prompt_variant, term_source)

		# Save prompts to row for audit trail
		row[f'{col_prefix}_system_prompt'] = system_content
		row[f'{col_prefix}_user_prompt'] = user_content

		# Build message list — omit system message entirely for minimal (None)
		# rather than passing an empty string.
		if system_content is not None:
			ollama_messages = [
				{'role': 'system', 'content': system_content},
				{'role': 'user', 'content': user_content},
			]
		else:
			ollama_messages = [{'role': 'user', 'content': user_content}]

		# Make Ollama call with timeout using threading
		ollama_response = None
		timeout_error = None

		def call_ollama():
			nonlocal ollama_response, timeout_error
			try:
				ollama_response = ollama.chat(
					model=ollama_model,
					messages=ollama_messages,
					options={'temperature': 0, 'top_p': 1, 'num_predict': 2048},
				)
			except Exception as e:
				timeout_error = e

		# Run Ollama call in a thread with timeout
		ollama_thread = threading.Thread(target=call_ollama, daemon=True)
		t_start = time.monotonic()
		if consecutive_timeouts > 0:
			console.print(f"  ⚠ Starting request with {consecutive_timeouts} consecutive timeout(s) already on record", style="bold yellow")
		ollama_thread.start()
		# Poll every 60s so we can see the model is still running vs. frozen
		_poll = 60
		while True:
			ollama_thread.join(timeout=_poll)
			if not ollama_thread.is_alive():
				break
			elapsed_so_far = time.monotonic() - t_start
			if elapsed_so_far >= request_timeout:
				break
			console.print(f"  ⏳ Still waiting... ({elapsed_so_far:.0f}s / {request_timeout}s) — {language_name} ({language_code})", style="dim yellow")
		elapsed = time.monotonic() - t_start

		# Check if thread is still alive (timed out)
		if ollama_thread.is_alive():
			consecutive_timeouts += 1
			console.print(f"⚠ Ollama request timed out after {elapsed:.1f}s (limit {request_timeout}s) - skipping ({consecutive_timeouts}/{MAX_CONSECUTIVE_TIMEOUTS})", style="bold yellow")
			console.print(f"  Language: {language_name} ({language_code}) | Model: {ollama_model}", style="dim yellow")

			if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
				console.print(f"✗ Too many consecutive Ollama timeouts ({MAX_CONSECUTIVE_TIMEOUTS}). Stopping to save progress.", style="bold red")
				raise OllamaUnresponsiveError(f"Ollama service unresponsive - stopping after {MAX_CONSECUTIVE_TIMEOUTS} consecutive timeouts")

			row[f'{col_prefix}_translated_term'] = None
			row[f'{col_prefix}_translation_rationale'] = None
			log_error_to_file(error_file_path, additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, f"{col_prefix}_translated_term": None}, status_code=408, error_url="ollama.chat - Request Timeout", error_message=f"Request timed out after {request_timeout}s")
			return row, consecutive_timeouts

		if timeout_error:
			console.print(f"⚠ Ollama request failed: {timeout_error}", style="bold yellow")
			row[f'{col_prefix}_translated_term'] = None
			row[f'{col_prefix}_translation_rationale'] = None
			log_error_to_file(error_file_path, additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, f"{col_prefix}_translated_term": None}, status_code=500, error_url=f"ollama.chat - {str(timeout_error)}", error_message=str(timeout_error))
			return row, consecutive_timeouts

		if ollama_response is None:
			console.print(f"⚠ No response from Ollama", style="bold yellow")
			row[f'{col_prefix}_translated_term'] = None
			row[f'{col_prefix}_translation_rationale'] = None
			return row, consecutive_timeouts

		console.print(f"Ollama response: {ollama_response}", style="bright_magenta")

		# done=False means the model hit num_predict mid-generation (repetition loop).
		# Timing fields are None in this state, so dict-style access raises KeyError.
		# Treat it as a deterministic failure (400) — retrying won't fix a hallucination loop.
		_done = getattr(ollama_response, 'done', True)
		if not _done:
			console.print(
				f"⚠ Ollama returned done=False for {language_name} ({language_code}) — "
				f"model hit token limit mid-generation (repetition loop). Skipping.",
				style="bold yellow"
			)
			row[f'{col_prefix}_translated_term'] = None
			row[f'{col_prefix}_translation_rationale'] = None
			log_error_to_file(
				error_file_path,
				additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, f"{col_prefix}_translated_term": None},
				status_code=400,
				error_url=f"ollama.chat - HallucinationLoop",
				error_message=f"done=False for {language_name} ({language_code}) — model hit token limit mid-generation",
			)
			return row, consecutive_timeouts

		message_content = ollama_response['message']['content']
		# Escape lone surrogates as \uXXXX literals so PyArrow can store them without
		# data loss — the escaped form is reversible and preserves what the model returned.
		message_content = ''.join(
			f'\\u{ord(c):04x}' if 0xD800 <= ord(c) <= 0xDFFF else c
			for c in message_content
		)
		row[f'{col_prefix}_content'] = message_content
		# Use getattr for timing metadata — these fields are None when done=False and
		# the ollama library's __getitem__ raises KeyError for None-valued fields.
		row[f'{col_prefix}_model'] = getattr(ollama_response, 'model', None)
		row[f'{col_prefix}_created_at'] = getattr(ollama_response, 'created_at', None)
		row[f'{col_prefix}_total_duration'] = getattr(ollama_response, 'total_duration', None)
		row[f'{col_prefix}_load_duration'] = getattr(ollama_response, 'load_duration', None)
		row[f'{col_prefix}_prompt_eval_count'] = getattr(ollama_response, 'prompt_eval_count', None)
		row[f'{col_prefix}_prompt_eval_duration'] = getattr(ollama_response, 'prompt_eval_duration', None)
		row[f'{col_prefix}_eval_count'] = getattr(ollama_response, 'eval_count', None)
		row[f'{col_prefix}_eval_duration'] = getattr(ollama_response, 'eval_duration', None)
		row[f'{col_prefix}_done_reason'] = getattr(ollama_response, 'done_reason', None)

		try:
			response = parse_translation_response(message_content)
			row[f'{col_prefix}_translated_term'] = response.translated_term
			row[f'{col_prefix}_translation_rationale'] = response.translation_rationale
			# Reset timeout counter on successful translation
			consecutive_timeouts = 0
			console.print(f"{col_prefix.capitalize()} translation: {_display_text(row[f'{col_prefix}_translated_term'])}", style="bold green")

		except (KeyError, ValueError, SyntaxError, ValidationError) as e:
			console.print(f"Error parsing Ollama response: {e}", style="bold red")
			row[f'{col_prefix}_translated_term'] = None
			row[f'{col_prefix}_translation_rationale'] = None
			log_error_to_file(
				error_file_path,
				additional_data={
					"term_source": term_source,
					"language_code": language_code,
					"variant": current_prompt_variant,
					f"{col_prefix}_translated_term": None
				},
				status_code=400,
				error_url="ollama.chat - Parsing Error",
				error_message=str(e),
			)

	except TranslationRefusedError as e:
		console.print(f"⚠ Ollama declined to translate {term_source} → {language_name}: {e.rationale}", style="bold yellow")
		row[f'{col_prefix}_translated_term'] = None
		row[f'{col_prefix}_translation_rationale'] = e.rationale
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, f"{col_prefix}_translated_term": None},
			status_code=400,
			error_url=f"ollama.chat - TranslationRefused",
			error_message=e.rationale,
		)
	except json.JSONDecodeError as e:
		console.print(f"Error decoding JSON response from Ollama: {e}", style="bold red")
		row[f'{col_prefix}_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				f"{col_prefix}_translated_term": None
			},
			status_code=400,
			error_url="ollama.chat - JSONDecodeError",
			error_message=str(e),
		)
	except ValueError as e:
		console.print(f"⚠ Ollama parse failure for {term_source} → {language_name}: {e}", style="bold yellow")
		row[f'{col_prefix}_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={"term_source": term_source, "language_code": language_code, "variant": current_prompt_variant, f"{col_prefix}_translated_term": None},
			status_code=400,
			error_url="ollama.chat - ParseError",
			error_message=str(e),
		)
	except Exception as e:
		console.print(f"Unexpected error: {e}", style="bold red")
		row[f'{col_prefix}_translated_term'] = None
		log_error_to_file(
			error_file_path,
			additional_data={
				"term_source": term_source,
				"language_code": language_code,
				"variant": current_prompt_variant,
				f"{col_prefix}_translated_term": None
			},
			status_code=500,
			error_url="ollama.chat - General Exception",
			error_message=str(e),
		)

	return row, consecutive_timeouts