"""
Translation prompt variants for comparative testing.

Four variants, run in this fixed order:
  minimal          - bare instruction, no additional context (baseline)
  expert_persona   - positions model as domain expert and native speaker
  native_rationale - asks for rationale written in the target language
  judge            - runs last; shows all unique translations from every prior variant
                     and service, deduplicated by value, and asks for the best synthesis
"""

# Brief descriptions shown inside the judge prompt so it understands each source.
VARIANT_DESCRIPTIONS = {
    'minimal':          'bare instruction to an LLM with no additional context',
    'expert_persona':   'LLM prompted as a domain expert and native speaker',
    'native_rationale': 'LLM asked to provide its reasoning in the target language',
    'Google Translate': 'automated machine translation (Google)',
    'EasyNMT':          'neural machine translation (Helsinki-NLP models)',
    'Lingvanex':        'automated machine translation (Lingvanex)',
    'Wikipedia':        'existing Wikipedia article title for this concept',
}

# Human-readable labels for each LLM service (used when building judge context).
LLM_SERVICE_LABELS = {
    'openai':  'OpenAI',
    'claude':  'Claude',
    'gemini':  'Gemini',
    'ollama':  'Ollama',
}

DIRECT_SERVICE_LABELS = {
    'gt':        'Google Translate',
    'enmt':      'EasyNMT',
    'lingvanex': 'Lingvanex',
    'wikipedia': 'Wikipedia',
}


def get_minimal_prompt(term_source: str, language_name: str, language_code: str) -> str:
    """
    Minimal/lean prompt — bare instruction with no frills.
    Tests whether additional context actually improves translation or just adds noise.
    """
    return (
        f"Translate '{term_source}' into {language_name} (ISO code: {language_code}). "
        f"IMPORTANT: Return ONLY valid JSON with no other text. "
        f"Use this exact format: "
        f'{{\"translated_term\": \"your translation\", \"translation_rationale\": \"brief reason\"}}'
    )


def get_expert_persona_prompt(term_source: str, language_name: str, language_code: str) -> str:
    """
    Expert persona prompt — positions model as domain expert and native speaker.
    Tests whether role-playing improves translation quality and confidence.
    """
    return (
        f"You are an expert {term_source} scholar and native {language_name} speaker "
        f"with deep knowledge of academic terminology. Your task: Translate the term "
        f"'{term_source}' into {language_name} (ISO code: {language_code}).\n\n"
        f"Provide your translation with consideration for:\n"
        f"1. Scholarly accuracy\n"
        f"2. Cultural equivalence\n"
        f"3. Academic conventions in {language_name}\n\n"
        f"IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:\n"
        f'{{\"translated_term\": \"your translation\", \"translation_rationale\": \"your detailed rationale\"}}'
    )


def get_native_rationale_prompt(term_source: str, language_name: str, language_code: str) -> str:
    """
    Native rationale prompt — asks for rationale written in the target language.
    Tests whether explaining in the target language forces deeper linguistic understanding.
    """
    return (
        f"You are a native {language_name} speaker and translation expert. "
        f"Translate '{term_source}' into {language_name} (ISO code: {language_code}).\n\n"
        f"Please provide your rationale in {language_name}, not English — unless {language_name} "
        f"is English or you have a strong reason to use English. Writing in {language_name} "
        f"demonstrates deep understanding of both the language and the term.\n\n"
        f"IMPORTANT: Return ONLY valid JSON with no other text. "
        f"Use this exact format (with explanation in {language_name}):\n"
        f'{{\"translated_term\": \"your translation in {language_name}\", '
        f'\"translation_rationale\": \"your explanation in {language_name}\"}}'
    )



def get_judge_prompt(
    term_source: str,
    language_name: str,
    language_code: str,
    existing_translations: dict = None,
) -> str:
    """
    Judge/synthesis prompt — runs last after all other variants.

    Receives all unique translations from every prior variant and service,
    grouped by identical value so consensus is visible. Null sources are noted.

    existing_translations format:
        {
            'unique_translations': [
                {'translation': str, 'sources': [str, ...]},
                ...  # sorted by number of agreeing sources, descending
            ],
            'missing_sources': [str, ...]   # sources that produced no translation
        }
    """
    prompt = (
        f"You are a {term_source} scholar. "
        f"Your task is to judge the following proposed translations of the academic term "
        f"'{term_source}' into {language_name} (ISO code: {language_code}), "
        f"then either select the best one or generate an improved translation.\n\n"
    )

    if existing_translations:
        unique = existing_translations.get('unique_translations', [])
        missing = existing_translations.get('missing_sources', [])

        if unique:
            prompt += "Approach descriptions:\n"
            for key, desc in VARIANT_DESCRIPTIONS.items():
                prompt += f"  - {key}: {desc}\n"
            prompt += "\n"

            prompt += "Proposed translations:\n"
            for i, entry in enumerate(unique, 1):
                sources_str = ', '.join(entry['sources'])
                prompt += f"  {i}. ({sources_str}): \"{entry['translation']}\"\n"

        if missing:
            prompt += (
                f"\nNote: The following sources produced no translation for this language: "
                f"{', '.join(missing)}.\n"
            )

        if unique:
            prompt += (
                "\nJudge these translations, noting where sources agreed or diverged. "
                "Select the best translation or generate a new one if none are satisfactory. "
                "Provide your reasoning.\n\n"
            )

    prompt += (
        "IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:\n"
        "{\"translated_term\": \"your translation\", "
        "\"translation_rationale\": \"your evaluation and reasoning\"}"
    )
    return prompt


# Ordered dict — judge must always be last.
PROMPT_VARIANTS = {
    'minimal':          get_minimal_prompt,
    'expert_persona':   get_expert_persona_prompt,
    'native_rationale': get_native_rationale_prompt,
    'judge':            get_judge_prompt,
}

PROMPT_DESCRIPTIONS = {
    'minimal':          'Bare instruction with no additional context (baseline)',
    'expert_persona':   'Model positioned as domain expert and native speaker',
    'native_rationale': 'Rationale requested in the target language',
    'judge':            'Synthesis: all unique translations from prior variants, deduplicated by value',
}

NON_JUDGE_VARIANTS = [v for v in PROMPT_VARIANTS if v != 'judge']


def get_prompt(
    variant: str,
    term_source: str,
    language_name: str,
    language_code: str,
    existing_translations: dict = None,
) -> str:
    """
    Return the formatted prompt for the given variant.

    Parameters
    ----------
    variant : str
        One of: 'minimal', 'expert_persona', 'native_rationale', 'judge'
    term_source : str
        The term to translate.
    language_name : str
        Target language name (e.g. 'French').
    language_code : str
        ISO 639-1/2 code (e.g. 'fr').
    existing_translations : dict, optional
        For 'judge': structured dict
            {'unique_translations': [...], 'missing_sources': [...]}.
        Ignored by other variants.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt variant: {variant!r}. "
            f"Choose from {list(PROMPT_VARIANTS.keys())}"
        )

    func = PROMPT_VARIANTS[variant]

    if variant == 'judge':
        return func(term_source, language_name, language_code, existing_translations)
    return func(term_source, language_name, language_code)
