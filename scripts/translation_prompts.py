"""
Translation prompt variants for comparative testing.

Each prompt is designed to test different strategies for improving translation quality
and understanding the impact of prompt engineering on LLM translation output.
"""

def get_minimal_prompt(term_source: str, language_name: str, language_code: str) -> str:
	"""
	Minimal/lean prompt - basic instruction with no frills.
	Tests: Whether additional context actually improves translation or just adds noise.

	Args:
	 term_source (str): The term to translate.
	 language_name (str): The name of the target language.
	 language_code (str): The ISO code of the target language.

	Returns:
	 str: The minimal prompt.
	"""
	return f"""Translate '{term_source}' into {language_name} (ISO code: {language_code}). IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format: {{"translated_term": "your translation", "translation_rationale": "brief reason"}}"""


def get_comparative_prompt(term_source: str, language_name: str, language_code: str, existing_translations: dict = None) -> str:
	"""
	Comparative prompt - shows existing translations and asks for refinement.
	Tests: Whether showing prior attempts helps or anchors the model to suboptimal solutions.

	Args:
		term_source (str): The term to translate.
		language_name (str): The name of the target language.
		language_code (str): The ISO code of the target language.
		existing_translations (dict, optional): Existing translations from other services.

	Returns:
		str: The comparative prompt.
	"""
	prompt = f"You speak {language_name} (ISO code: {language_code}). How would you translate the term '{term_source}' into {language_name}?\n"

	if existing_translations:
		suggestions = []
		if existing_translations.get('gt'):
			suggestions.append(f"Google Translate: '{existing_translations['gt']}'")
		if existing_translations.get('openai'):
			suggestions.append(f"OpenAI: '{existing_translations['openai']}'")
		if existing_translations.get('claude'):
			suggestions.append(f"Claude: '{existing_translations['claude']}'")
		if existing_translations.get('enmt'):
			suggestions.append(f"EasyNMT: '{existing_translations['enmt']}'")

		if suggestions:
			prompt += f"Consider these existing suggestions: {', '.join(suggestions)}. "
			prompt += "Feel free to modify or refine them based on your linguistic expertise."

	prompt += f"\n\nIMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:\n{{'translated_term': 'your translation', 'translation_rationale': 'your rationale'}}"
	return prompt


def get_expert_persona_prompt(term_source: str, language_name: str, language_code: str) -> str:
	"""
	Expert persona prompt - positions model as domain expert and native speaker.
	Tests: Whether role-playing improves translation quality and confidence.

	Args:
		term_source (str): The term to translate.
		language_name (str): The name of the target language.
		language_code (str): The ISO code of the target language.
	Returns:
		str: The expert persona prompt.
	"""
	return f"""You are an expert {term_source} scholar and native {language_name} speaker with a deep knowledge of academic terminology. Your task: Translate the term '{term_source}' into {language_name} (ISO code: {language_code}).

	Provide your translation with consideration for:
	1. Scholarly accuracy 
	2. Cultural equivalence 
	3. Academic conventions in {language_name}

	IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:
	{{'translated_term': 'your translation', 'translation_rationale': 'your detailed rationale'}}"""


def get_contextual_prompt(term_source: str, language_name: str, language_code: str, context: str = None) -> str:
	"""
	Contextual prompt - provides rich context about what the term means.
	Tests: Whether providing semantic/definitional context improves translation accuracy.

	Args:
		term_source (str): The term to translate.
		language_name (str): The name of the target language.
		language_code (str): The ISO code of the target language.
		context (str, optional): A plain-text description of the term and field. If None,
			a generic academic-field framing is used. Callers should supply field-specific
			context when translating terms with no obvious cross-linguistic equivalent.

	Returns:
		str: The contextual prompt.
	"""
	if context is None:
		context = (
			f"\'{term_source}\' is an academic field or scholarly concept. "
			f"Provide a translation that accurately conveys its meaning in an academic context, "
			f"preserving both its technical precision and its disciplinary significance."
		)

	return f"""You speak {language_name} (ISO code: {language_code}).

	Translate the following term into {language_name}:

	**Term:** {term_source}

	**Definition & Context:**
	{context}

	IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:
	{{'translated_term': 'your translation', 'translation_rationale': 'explain your choices'}}"""


def get_native_rationale_prompt(term_source: str, language_name: str, language_code: str) -> str:
	"""
	Native rationale prompt - asks for rationale to be written in the target language.
	Tests: Whether explaining in the target language forces deeper linguistic understanding.

	Args:
		term_source (str): The term to translate.
		language_name (str): The name of the target language.
		language_code (str): The ISO code of the target language.
	Returns:
		str: The native rationale prompt.
	"""
	return f"""You are a native {language_name} speaker and translation expert. Translate '{term_source}' into {language_name} (ISO code: {language_code}).

	Please provide your rationale (explanation) in {language_name}, not English. Unless you have a strong reason to use English (such as when {language_name} is English), your rationale should be in {language_name} to demonstrate your deep understanding of the language and the term. 

	IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format (with explanation in {language_name}):
	{{'translated_term': 'the translation in {language_name}', 'translation_rationale': 'your explanation in {language_name}'}}"""


# Mapping for easy access
PROMPT_VARIANTS = {
	'minimal': get_minimal_prompt,
	'comparative': get_comparative_prompt,
	'expert_persona': get_expert_persona_prompt,
	'contextual': get_contextual_prompt,
	'native_rationale': get_native_rationale_prompt,
}

PROMPT_DESCRIPTIONS = {
	'minimal': 'Minimal/lean instruction with no additional context',
	'comparative': 'Shows existing translations and asks for refinement (original approach)',
	'expert_persona': 'Positions model as domain expert and native speaker',
	'contextual': 'Provides rich semantic and definitional context',
	'native_rationale': 'Asks for rationale to be written in target language',
}


def get_prompt(variant: str, term_source: str, language_name: str, language_code: str,
			   existing_translations: dict = None, context: str = None) -> str:
	"""
	Get a prompt for the specified variant.

	Parameters
	----------
	variant : str
		One of: 'minimal', 'comparative', 'expert_persona', 'contextual', 'native_rationale'
	term_source : str
		The term to translate
	language_name : str
		The language name (e.g., 'French')
	language_code : str
		The ISO 639-1 language code (e.g., 'fr')
	existing_translations : dict, optional
		Dictionary of existing translations from other services (for comparative)
	context : str, optional
		Plain-text description of the term for the 'contextual' variant. When None a
		generic academic framing is used. Pass a field-specific description when
		translating terms that may have no obvious equivalent (e.g. 'Digital Humanities',
		'Computational Social Science').

	Returns
	-------
	str
		The formatted prompt
	"""
	if variant not in PROMPT_VARIANTS:
		raise ValueError(f"Unknown prompt variant: {variant}. Choose from {list(PROMPT_VARIANTS.keys())}")

	func = PROMPT_VARIANTS[variant]

	if variant == 'comparative':
		return func(term_source, language_name, language_code, existing_translations)
	elif variant == 'contextual':
		return func(term_source, language_name, language_code, context)
	else:
		return func(term_source, language_name, language_code)