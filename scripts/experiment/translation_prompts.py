"""
Translation prompt variants for comparative testing.

Four variants, run in this fixed order:
1. minimal: bare instruction, no system prompt, no scaffolding (baseline)
2. fluent_speaker: asks for rationale written in the target language
3. github_searcher: frames translation as studying GitHub communities across many languages (goal-orientation framing)
4. judge: runs last; shows all unique translations from every prior variant and service, deduplicated by value, and asks for the best synthesis

System prompts
--------------
Each variant exposes a paired (system, user) prompt via get_system_prompt() and get_user_prompt(). For the `minimal` baseline, get_system_prompt() returns None to signal that NO system prompt should be sent. This matters: the minimal variant is the comparison point against which fluent_speaker and github_searcher earn their keep, and adding any system prompt (even a generic 'helpful assistant' framing) would pre-load behavior that the other variants are supposed to be introducing.

Per-provider handling for `minimal` (None system prompt):
  - Anthropic API: omit the `system=...` argument entirely
  - OpenAI / DeepSeek / Llama-style: omit the {'role': 'system', ...} message from the messages list — do NOT send an empty string, which some providers interpret as a real system prompt
  - Google Gemini: omit `system_instruction` from the GenerateContentConfig
  - Ollama-hosted local models: omit the system message in messages list

The point is that 'no system prompt' should be an actual absence, not an empty string, since providers handle empty-string system prompts differently (some apply defaults, some treat as a real signal). Always omit the parameter.
"""

# ── Source categorisation for the judge prompt ───────────────────────────────
# Three epistemic stances on "what is a translation":
#   1. LLM prompt variants — generative model output under different framings
#   2. Machine translation services — algorithmic baselines (statistical/neural MT)
#   3. Community resources — human-curated reference translations (extensible)

# Descriptions for the LLM prompt variants only.
PROMPT_VARIANT_DESCRIPTIONS = {
    'minimal':         'bare instruction to an LLM with no system prompt and no scaffolding',
    'fluent_speaker':  'LLM asked to provide its reasoning in the target language',
    'github_searcher': 'LLM framed as a researcher studying communities on GitHub across many languages',
}

# Descriptions for the machine translation baseline services.
MT_SERVICE_DESCRIPTIONS = {
    'Google Translate': 'automated machine translation (Google)',
    'EasyNMT':          'neural machine translation (Helsinki-NLP models)',
    'Lingvanex':        'automated machine translation (Lingvanex)',
}

# Descriptions for community-curated reference resources. Extensible: future
# additions might include IATE, LCSH, DH journal subject headings, Wikidata, etc.
COMMUNITY_RESOURCE_DESCRIPTIONS = {
    'Wikipedia': 'community-curated translation from Wikipedia article interlanguage links (human editorial consensus, not machine translation)',
}

# Backward-compatible union for any code that still reads VARIANT_DESCRIPTIONS
# as a single flat dict. New code should reference the three split dicts above.
VARIANT_DESCRIPTIONS = {
    **PROMPT_VARIANT_DESCRIPTIONS,
    **MT_SERVICE_DESCRIPTIONS,
    **COMMUNITY_RESOURCE_DESCRIPTIONS,
}

# Human-readable labels for each LLM service (used when building judge context).
LLM_SERVICE_LABELS = {
    'openai':   'OpenAI (GPT-4o)',
    'claude':   'Claude',
    'gemini':   'Gemini',
    'deepseek': 'DeepSeek (V3)',
    'llama':    'Llama (local)',
    'gemma':    'Gemma (local)',
    'qwen':     'Qwen (local)',
    'mistral':  'Mistral (local)',
}

# One-line descriptions of each service shown in the judge prompt.
SERVICE_DESCRIPTIONS = {
    'OpenAI (GPT-4o)':  'cloud LLM by OpenAI (GPT-4o)',
    'Claude':           'cloud LLM by Anthropic (Claude Sonnet 4.5)',
    'Gemini':           'cloud LLM by Google (Gemini 2.5 Flash)',
    'DeepSeek (V3)':    'cloud LLM by DeepSeek (DeepSeek-V3 / deepseek-chat)',
    'Llama (local)':    'open-weight LLM running locally via Ollama (Llama 3.1)',
    'Gemma (local)':    'open-weight LLM running locally via Ollama (Gemma 3 12B)',
    'Qwen (local)':     'open-weight LLM running locally via Ollama (Qwen 2.5 7B)',
    'Mistral (local)':  'open-weight LLM running locally via Ollama (Mistral 7B)',
}

# Per-source filename → display-name maps. Split conceptually but they share
# the direct_services/ directory on disk; the split lives at the methodology
# layer, not the filesystem layer.
MT_SERVICE_LABELS = {
    'gt':        'Google Translate',
    'enmt':      'EasyNMT',
    'lingvanex': 'Lingvanex',
}

COMMUNITY_RESOURCE_LABELS = {
    'wikipedia': 'Wikipedia',
}

# Backward-compatible union: all per-language files written to direct_services/.
DIRECT_SERVICE_LABELS = {**MT_SERVICE_LABELS, **COMMUNITY_RESOURCE_LABELS}


# ── System prompts (variant-specific) ────────────────────────────────────────

# Default scholar system prompt used by fluent_speaker and judge. The judge
# variant carries its scholar framing inline because it also needs to inject
# the source term and language into the system role for stronger grounding.
_SCHOLAR_SYSTEM_PROMPT = (
    "You are a {term_source} scholar who speaks many languages."
)

# Goal-oriented system prompt for github_searcher. Frames the model as a
# corpus-builder rather than a scholar. The contrast with _SCHOLAR_SYSTEM_PROMPT
# is the variant's core test: does goal-orientation (translate-to-find) shape
# translation choices differently than scholarly framing (translate-for-accuracy)?
_GITHUB_SEARCHER_SYSTEM_PROMPT = (
    "You are a researcher studying {term_source} communities on GitHub. "
    "Because GitHub spans many language communities, you need to translate "
    "{term_source} into multiple languages to identify relevant activity "
    "on the platform."
)


def get_system_prompt(variant: str, term_source: str) -> str | None:
    """
    Return the system prompt for a given variant, or None for `minimal`.

    Returning None signals that NO system prompt should be sent — callers must omit the system argument entirely from the provider API call, not substitute an empty string. See the module docstring for per-provider
    handling.

    Parameters
    ----------
    variant : str
        One of 'minimal', 'fluent_speaker', 'github_searcher', 'judge'.
    term_source : str
        The source term being translated. Used to interpolate scholar
        or corpus-builder framing for non-minimal variants.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt variant: {variant!r}. "
            f"Choose from {list(PROMPT_VARIANTS.keys())}"
        )
    if variant == 'minimal':
        return None
    if variant == 'github_searcher':
        return _GITHUB_SEARCHER_SYSTEM_PROMPT.format(term_source=term_source)
    return _SCHOLAR_SYSTEM_PROMPT.format(term_source=term_source)


# ── User prompts (variant-specific) ──────────────────────────────────────────

def get_minimal_prompt(term_source: str, language_name: str, language_code: str) -> str:
    """
    Minimal/baseline user prompt — bare translation request plus a JSON schema for parseable output. No scholarly framing, no urgency markers, no role-play. Paired with NO system prompt (see get_system_prompt()).

    The ISO code is a disambiguation control (not scaffolding) — it ensures all variants target the same language, which matters for codes like zh vs zh-tw or ambiguous language names. The JSON requirement is plumbing.
    
    Parameters
    ----------
    term_source : str
		The term to translate (e.g. "Digital Humanities").
    language_name : str
		The target language name (e.g. "French").
    language_code : str
		The target language ISO code (e.g. "fr"). Used for disambiguation, not as scaffolding.
        
    Returns
    -------
    str		 The formatted user prompt for the minimal variant.
    
    """
    return (
        f"Translate '{term_source}' into {language_name} (ISO {language_code}). "
        f"Reply in JSON: "
        f'{{"translated_term": "...", "translation_rationale": "..."}}'
    )


def get_fluent_speaker_prompt(term_source: str, language_name: str, language_code: str) -> str:
    """
    Fluent speaker prompt — asks for rationale written in the target language.
    Tests whether explaining in the target language forces deeper linguistic understanding.
    """
    return (
        f"Translate '{term_source}' into {language_name} (ISO code: {language_code}).\n\n"
        f"Write your rationale in {language_name}, not English. (If the target "
        f"language is English, write in English.) Writing in {language_name} "
        f"demonstrates deep understanding of both the language and the term.\n\n"
        f"IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:\n"
        f'{{"translated_term": "...", "translation_rationale": "..."}}'
    )



def get_github_searcher_prompt(term_source: str, language_name: str, language_code: str) -> str:
    """
    GitHub searcher prompt — frames translation as building a multilingual
    GitHub search corpus. Tests whether explicit goal-orientation (the translation
    will be used as a search query) shapes translation choices differently than
    scholarly framing.
    """
    return (
        f"Translate '{term_source}' into {language_name} (ISO code: {language_code}). "
        f"This translation will be used as a search query to find '{term_source}' "
        f"scholarship on GitHub — including repositories, users, issues, and topics.\n\n"
        f"IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:\n"
        f'{{"translated_term": "...", "translation_rationale": "..."}}'
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
        f"Your task is to judge the following proposed translations of the academic term "
        f"'{term_source}' into {language_name} (ISO code: {language_code}), "
        f"then either select the best one or generate an improved translation. "
        f"Please explain your evaluation rationale.\n\n"
    )

    if existing_translations:
        unique = existing_translations.get('unique_translations', [])
        missing = existing_translations.get('missing_sources', [])

        if unique:
            all_sources = {src for entry in unique for src in entry['sources']}
            # Extract which LLM service labels appear in the sources
            present_services = {
                svc for svc in SERVICE_DESCRIPTIONS
                if any(svc in src for src in all_sources)
            }

            # LLM prompt strategies
            prompt_variants_present = [
                (k, v) for k, v in PROMPT_VARIANT_DESCRIPTIONS.items()
                if any(k in src for src in all_sources)
            ]
            if prompt_variants_present:
                prompt += "Prompt strategies (LLM variants):\n"
                for key, desc in prompt_variants_present:
                    prompt += f"  - {key}: {desc}\n"
                prompt += "\n"

            if present_services:
                prompt += "LLM sources:\n"
                for svc, desc in SERVICE_DESCRIPTIONS.items():
                    if svc in present_services:
                        prompt += f"  - {svc}: {desc}\n"
                prompt += "\n"

            # Machine translation baselines
            mt_present = [
                (k, v) for k, v in MT_SERVICE_DESCRIPTIONS.items()
                if any(k in src for src in all_sources)
            ]
            if mt_present:
                prompt += "Machine translation baselines:\n"
                for key, desc in mt_present:
                    prompt += f"  - {key}: {desc}\n"
                prompt += "\n"

            # Community-curated resources (human editorial consensus)
            cr_present = [
                (k, v) for k, v in COMMUNITY_RESOURCE_DESCRIPTIONS.items()
                if any(k in src for src in all_sources)
            ]
            if cr_present:
                prompt += "Community-curated references:\n"
                for key, desc in cr_present:
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
                "\nAs a reminder, your goal is to judge these translations, noting "
                "where sources agreed or diverged. Then you should select the best "
                "translation or generate a new one if none are satisfactory. Please "
                "provide your reasoning for your evaluation.\n\n"
            )

    prompt += (
        "IMPORTANT: Return ONLY valid JSON with no other text. Use this exact format:\n"
        "{\"translated_term\": \"...\", \"translation_rationale\": \"...\"}"
    )
    return prompt


# Ordered dict — judge must always be last.
PROMPT_VARIANTS = {
    'minimal':          get_minimal_prompt,
    'fluent_speaker':   get_fluent_speaker_prompt,
    'github_searcher':  get_github_searcher_prompt,
    'judge':            get_judge_prompt,
}

PROMPT_DESCRIPTIONS = {
    'minimal':          'Bare user instruction, no system prompt (baseline)',
    'fluent_speaker':   'Rationale requested in the target language',
    'github_searcher':  'Framing as researcher studying GitHub communities across many languages',
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
    Return the formatted USER prompt for the given variant.

    For the paired SYSTEM prompt (which may be None for minimal), call
    get_system_prompt(variant, term_source) separately.

    Parameters
    ----------
    variant : str
        One of: 'minimal', 'fluent_speaker', 'github_searcher', 'judge'
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

    # 'nan' is the ISO 639-3 code for Min Nan Chinese. LLMs misread it as null/NaN,
    # so annotate it explicitly to prevent refusals across all variants.
    if language_code == 'nan':
        language_code = 'nan (ISO 639-3 code for Min Nan Chinese — this is a real language, not a null value)'

    if variant == 'judge':
        return func(term_source, language_name, language_code, existing_translations)
    return func(term_source, language_name, language_code)