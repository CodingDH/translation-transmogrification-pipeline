"""
Data processing utilities for translation pipeline.
Includes parsing, language detection, extraction, and utility functions.
"""

import json
import os
import re
import ast
from typing import List
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from pydantic import BaseModel, Field, ValidationError

from scripts.utils import read_csv_file

# Pydantic model for parsing translation responses
class TranslationResponse(BaseModel):
	translated_term: str = Field(..., description="The translated term")
	translation_rationale: str = Field(..., description="The rationale for the translation")

def parse_translation_response(message_content: str) -> TranslationResponse:
	"""
	Robustly parse a translation response from an LLM into a TranslationResponse.

	LLMs frequently ignore instructions to return plain JSON and wrap their output
	in markdown code fences (```json ... ```) or use Python-style single-quoted dicts.
	They also sometimes return near-miss key names (e.g. 'translated_target' instead
	of 'translated_term'). This function tries four strategies in order:
	  1. Strip markdown fences and parse as JSON directly.
	  2. Extract a dict-like pattern and try ast.literal_eval (handles single quotes).
	  3. Clean quotes and retry JSON parsing.
	  4. Manually extract fields with regex as a last resort.

	Raises ValueError if no strategy succeeds.
	"""
	# Known near-miss key names that models produce instead of the canonical ones.
	TERM_ALIASES = {'translated_target', 'translation', 'translated_text', 'translated_word', 'term'}
	RATIONALE_ALIASES = {'rationale', 'reason', 'explanation', 'translation_reason', 'translation_explanation'}

	def normalize_keys(d: dict) -> dict:
		"""Map near-miss keys onto the canonical TranslationResponse field names."""
		result = dict(d)
		if 'translated_term' not in result:
			for alias in TERM_ALIASES:
				if alias in result:
					result['translated_term'] = result.pop(alias)
					break
		if 'translation_rationale' not in result:
			for alias in RATIONALE_ALIASES:
				if alias in result:
					result['translation_rationale'] = result.pop(alias)
					break
		# Provide a default so Pydantic validation doesn't fail on a missing rationale
		result.setdefault('translation_rationale', 'No rationale provided')
		return result

	# Strategy 1: strip markdown fences and try JSON
	cleaned = re.sub(r'^```(?:json)?\s*', '', message_content.strip(), flags=re.IGNORECASE)
	cleaned = re.sub(r'```\s*$', '', cleaned.strip())
	try:
		return TranslationResponse(**normalize_keys(json.loads(cleaned)))
	except (json.JSONDecodeError, ValidationError):
		pass

	# Strategy 2: extract first {...} block and try ast.literal_eval
	dict_match = re.search(r'\{[^{}]*(?:\'[^\']*\'[^{}]*)*\}', message_content, re.DOTALL)
	if dict_match:
		try:
			return TranslationResponse(**normalize_keys(ast.literal_eval(dict_match.group())))
		except (SyntaxError, ValueError, ValidationError):
			pass

		# Strategy 3: clean quotes and retry JSON
		try:
			clean_str = dict_match.group().replace("\\'", "'").replace("'", '"')
			return TranslationResponse(**normalize_keys(json.loads(clean_str)))
		except (json.JSONDecodeError, ValidationError):
			pass

	# Strategy 4: regex field extraction — match canonical name or any known alias
	term_pattern = r'[\'"](' + '|'.join(['translated_term'] + list(TERM_ALIASES)) + r')[\'"]\s*:\s*[\'"]([^\'\"]*)[\'"]'
	t_match = re.search(term_pattern, message_content)
	r_match = re.search(
		r'[\'"]translation_rationale[\'"]\s*:\s*[\'"](.+?)(?:[\'"](?=\s*[,}\\])|[\'"]?\s*$)',
		message_content, re.DOTALL
	)
	if t_match:
		translated = t_match.group(2).replace("\\'", "'").replace('\\"', '"')
		rationale = r_match.group(1).strip().rstrip("'\">").replace("\\'", "'") if r_match else 'No rationale provided'
		return TranslationResponse(translated_term=translated, translation_rationale=rationale)

	raise ValueError(f"Could not parse translation response: {message_content[:200]}")

def check_detect_language(row: pd.Series, is_repo: bool = False, console=None, translate_client=None) -> pd.Series:
	"""
	Checks the detected language of a row of text using Google Cloud Translate API

	Parameters
	----------
	row : pd.Series
		A row of a dataframe with a text column
	is_repo : bool
		A boolean indicating whether the text is a repo description or a bio
	console : Console
		Rich console for logging output
	translate_client : google.cloud.translate_v2.Client
		Google Translate client

	Returns a series with the detected language and confidence score
	"""
	text = row.description if is_repo else row.bio
	if pd.notna(text) and len(text) > 1:  # Additional check if text is not NaN
		try:
			result = translate_client.detect_language(text)
			row['detected_language'] = result['language']
			row['detected_language_confidence'] = result['confidence']
		except Exception as e:
			if console:
				console.print(f"Error detecting language for {text}: {e}", style="bold red")
			row['detected_language'] = None
			row['detected_language_confidence'] = None
	else:
		row['detected_language'] = None
		row['detected_language_confidence'] = None
	return row

def extract_dictionaries_from_string(text: str) -> list:
	"""
	Extract dictionaries from a string.

	Parameters
	----------
	text : str
		The input string containing dictionary-like patterns.

	Returns
	-------
	list
		A list of extracted dictionaries.
	"""
	if pd.isna(text):
		return []

	# Find all dictionary-like patterns in the string
	pattern = r"\{.*?\}"
	matches = re.findall(pattern, text)

	# Parse the found patterns into dictionaries
	dictionaries = []
	for match in matches:
		try:
			dictionaries.append(ast.literal_eval(match))
		except (ValueError, SyntaxError):
			continue

	return dictionaries

def extract_ollama_translated_term(row: pd.Series) -> pd.Series:
	"""
	Extract the translated term from the OLLAMA content.

	Parameters
	----------
	row : pd.Series
		A row of the DataFrame.

	Returns
	-------
	str
		The translated term.
	"""
	ollama_translation = None
	for dictionary in row['ollama_extracted_dictionaries']:
		if 'translated_term' in dictionary:
			ollama_translation = dictionary['translated_term']
	row['ollama_translation'] = ollama_translation
	return row

def get_directionality(directionality_path: str) -> pd.DataFrame:
	"""
	Function to get language directionality from Wikimedia

	Parameters
	----------
	directionality_path : str
	Returns a dataframe with language directionality
	"""
	# Read in the directionality data if it exists, otherwise scrape it from Wikimedia
	if os.path.exists(directionality_path):
		df = read_csv_file(directionality_path)
	else:
		# Get the directionality data from Wikimedia using BeautifulSoup
		url = "https://meta.wikimedia.org/wiki/Template:List_of_language_names_ordered_by_code"
		response = requests.get(url, timeout=10)
		soup = BeautifulSoup(response.text, 'html.parser')
		table = soup.find_all('table')[0]
		df = pd.read_html(str(table))[0]
		df.to_csv(
			directionality_path, index=False)
	if 'code' in df.columns:
		df = df.rename(columns={'code': 'language_code'})
	return df

def is_enmt_model_available(target_lang: str, console=None) -> bool:
	"""
	Function to check if a model is available for a target language in EasyNMT

	Parameters
	----------
	target_lang : str
		A target language code
	console : Console
		Rich console for logging output

	Returns a boolean indicating whether the model is available
	"""
	model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
	try:
		# Attempt to load the model tokenizer to check availability
		_ = AutoTokenizer.from_pretrained(model_name)
		return True
	except Exception as e:
		if console:
			console.print(f"Model not available for {target_lang}: {e}", style="bold red")
		return False
