"""
translator.py — GutBot Language Translation Utility

Provides automatic language detection and Hindi <-> English translation
using deep-translator (free, no API key required).

Supported languages: English ('en') and Hindi ('hi')
"""

import logging
from functools import lru_cache

logger = logging.getLogger("GutBot-Translator")

# Supported languages mapping
SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
}

# Hindi greetings/common words override (fast-path, no network needed)
_HINDI_GREETINGS = {
    "हाँ": "yes",
    "हां": "yes",
    "नहीं": "no",
    "नमस्ते": "hello",
    "नमस्कार": "hello",
    "धन्यवाद": "thank you",
    "शुक्रिया": "thank you",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Returns a language code like 'en', 'hi', etc.
    Falls back to 'en' on any error.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42  # Make detection deterministic
        lang = detect(text.strip())
        return lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"


def translate_to_english(text: str, src_lang: str = "auto") -> str:
    """
    Translate the given text to English.
    Returns the original text if translation fails.
    """
    text = text.strip()
    if not text:
        return text

    # Fast-path: check if already English
    if src_lang == "en":
        return text

    # Fast-path: short Hindi overrides
    if text in _HINDI_GREETINGS:
        return _HINDI_GREETINGS[text]

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=src_lang if src_lang != "auto" else "auto", target="en").translate(text)
        if translated:
            logger.info(f"Translated [{src_lang}→en]: '{text[:40]}' → '{translated[:40]}'")
            return translated
    except Exception as e:
        logger.warning(f"Translation to English failed: {e}")

    # Fallback: return original
    return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate an English response to the target language.
    Returns the original English text if translation fails.
    """
    text = text.strip()
    if not text or target_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator

        # deep-translator has a character limit per call (~5000 chars)
        # Split long texts into paragraphs and translate each
        MAX_CHUNK = 4500
        if len(text) <= MAX_CHUNK:
            translated = GoogleTranslator(source="en", target=target_lang).translate(text)
            if translated:
                logger.info(f"Translated [en→{target_lang}]: first 40 chars done")
                return translated
        else:
            # Chunk by newlines for long responses
            lines = text.split("\n")
            translated_lines = []
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) + 1 < MAX_CHUNK:
                    chunk += line + "\n"
                else:
                    if chunk:
                        t = GoogleTranslator(source="en", target=target_lang).translate(chunk.strip())
                        translated_lines.append(t if t else chunk)
                    chunk = line + "\n"
            if chunk:
                t = GoogleTranslator(source="en", target=target_lang).translate(chunk.strip())
                translated_lines.append(t if t else chunk)
            return "\n".join(translated_lines)

    except Exception as e:
        logger.warning(f"Translation from English failed: {e}")

    return text


def is_supported_lang(lang_code: str) -> bool:
    """Check if a language code is supported."""
    return lang_code in SUPPORTED_LANGS
