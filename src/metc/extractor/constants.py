whitespace_and_newlines = {
    "\u0020",  # SPACE
    "\u00a0",  # NO-BREAK SPACE
    "\u1680",  # OGHAM SPACE MARK
    "\u2000",  # EN QUAD
    "\u2001",  # EM QUAD
    "\u2002",  # EN SPACE
    "\u2003",  # EM SPACE
    "\u2004",  # THREE-PER-EM SPACE
    "\u2005",  # FOUR-PER-EM SPACE
    "\u2006",  # SIX-PER-EM SPACE
    "\u2007",  # FIGURE SPACE
    "\u2008",  # PUNCTUATION SPACE
    "\u2009",  # THIN SPACE
    "\u200a",  # HAIR SPACE
    "\u2028",  # LINE SEPARATOR
    "\u2029",  # PARAGRAPH SEPARATOR
    "\u202f",  # NARROW NO-BREAK SPACE
    "\u205f",  # MEDIUM MATHEMATICAL SPACE
    "\u3000",  # IDEOGRAPHIC SPACE
    "\u000a",  # LINE FEED (LF)
    "\u000b",  # VERTICAL TAB (VT)
    "\u0009",  # TAB
    "\u000c",  # FORM FEED (FF)
    "\u000d",  # CARRIAGE RETURN (CR)
    "\u0085",  # NEXT LINE (NEL)
}

alternative_dash = {
    "–",  # En Dash: U+2013
    "—",  # Em Dash: U+2014
    "‒",  # Figure Dash: U+2012
    "―",  # Horizontal Bar: U+2015
    "−",  # Minus Sign: U+2212
    "﹣",  # Small Hyphen-Minus: U+FE63
    "－",  # Fullwidth Hyphen-Minus: U+FF0D
}

dual_quotation = {
    '"',  # Quotation Mark: U+0022
    "'",  # Apostrophe: U+0027
    "‛",  # Single High-Reversed-9 Quotation Mark: U+201B
    "`",  # Grave Accent: U+0060
    "‟",  # Double High-Reversed-9 Quotation Mark: U+201F
}

opening_quotation = {
    '"',  # Quotation Mark: U+0022
    "«",  # Left Double Guillemet: U+00AB
    "‹",  # Left Single Guillemet: U+2039
    "\u201c",  # Double Left Quotation Mark: U+201C
    "„",  # Double Low-9 Quotation Mark: U+201E
    "\u2019",  # Single Right Quotation Mark: U+2019
    "\u2018",  # Single Left Quotation Mark: U+2018
    "‚",  # Single Low-9 Quotation Mark: U+201A
}

closing_quotation = {
    '"',  # Quotation Mark: U+0022
    "\u201d",  # Double Right Quotation Mark: U+201D
    "»",  # Right Double Guillemet: U+00BB
    "›",  # Right Single Guillemet: U+203A
    "\u2019",  # Single Right Quotation Mark: U+2019
}

separator_punctuation = {
    ",",  # COMMA (U+002C)
    "?",  # QUESTION MARK (U+003F)
    "!",  # EXCLAMATION MARK (U+0021)
    ":",  # COLON (U+003A)
    ";",  # SEMICOLON (U+003B)
    "…",  # HORIZONTAL ELLIPSIS (U+2026)
    "•",  # BULLET (U+2022)
}

bracket_punctuation = {
    "(",  # LEFT PARENTHESIS (U+0028)
    ")",  # RIGHT PARENTHESIS (U+0029)
    "[",  # LEFT SQUARE BRACKET (U+005B)
    "]",  # RIGHT SQUARE BRACKET (U+005D)
}

punctuation_marks = {
    ".",
    ",",
    ";",
    ":",
    "?",
    "!",
    "…",
    "-",  # Hyphen-minus U+002D
    "\u2013",  # En dash U+2013
    "\u2014",  # Em dash U+2014
    "(",
    ")",
    "[",
    "]",
    '"',
    "\u201a",  # ‚ SINGLE LOW-9 QUOTATION MARK
    "‛",
    "“",
    "”",
    "„",
    "‟",
    "‹",
    "›",
    "«",
    "»",
    "¿",
    "¡",
    "‽",
}

closing_punctuation_marks = {".", "?", "!", "…", "‽", ";"}

colon_punctuation_marks = {":", ";"}

OBJECT_REPLACEMENT_CHAR = "\ufffc"

special_characters = {
    OBJECT_REPLACEMENT_CHAR,
}

word_delimiters = (
    whitespace_and_newlines
    | alternative_dash
    | dual_quotation
    | opening_quotation
    | closing_quotation
    | separator_punctuation
    | bracket_punctuation
    | special_characters
)

# Sentence-ending punctuation including CJK and Devanagari variants.
eos_chars = {".", "…", "?", "!", "。", "।", "॥"}

# Punctuation that may trail after an EOS char at the very end of the text.
# Includes ASCII quotes/brackets as well as common CJK closing quotes.
trailing_punct_after_eos = {
    '"',
    "'",
    "»",
    "›",
    ")",
    "]",
    "}",
    "\u201d",  # " RIGHT DOUBLE QUOTATION MARK
    "\u2018",  # '
    "\u2019",  # '
    "」",  # CJK RIGHT CORNER BRACKET
    "』",  # CJK RIGHT WHITE CORNER BRACKET
    "》",  # CJK RIGHT DOUBLE ANGLE BRACKET
}


def is_chinese_japanese_char(char: str) -> bool:
    """Check if character is Chinese or Japanese.

    Each Chinese/Japanese character should be treated as a separate token
    since these languages don't use spaces between words.
    """
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)  # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
        or (0x3040 <= cp <= 0x309F)  # Hiragana
        or (0x30A0 <= cp <= 0x30FF)  # Katakana
        or (0xFF00 <= cp <= 0xFF60)  # Fullwidth ASCII variants
        or (0xFF61 <= cp <= 0xFF9F)  # Halfwidth Katakana
        or (0xFFE0 <= cp <= 0xFFEF)  # Fullwidth symbol variants
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0x2A700 <= cp <= 0x2B73F)  # CJK Extension C
        or (0x2B740 <= cp <= 0x2B81F)  # CJK Extension D
        or (0x2B820 <= cp <= 0x2CEAF)  # CJK Extension E
        or (0x2CEB0 <= cp <= 0x2EBEF)  # CJK Extension F
        or (0x30000 <= cp <= 0x3134F)  # CJK Extension G
        or (0x31350 <= cp <= 0x323AF)  # CJK Extension H
    )


def is_korean_char(char: str) -> bool:
    """Check if character is Korean Hangul.

    Korean uses spaces between words and has an alphabetic syllable-based
    writing system (Hangul), so it should be tokenized like European languages
    rather than character-by-character like Chinese/Japanese.
    """
    cp = ord(char)
    return (
        (0xAC00 <= cp <= 0xD7AF)  # Korean Hangul Syllables
        or (0x1100 <= cp <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= cp <= 0x318F)  # Hangul Compatibility Jamo
        or (0xA960 <= cp <= 0xA97F)  # Hangul Jamo Extended-A
        or (0xD7B0 <= cp <= 0xD7FF)  # Hangul Jamo Extended-B
    )


def is_alternative_dash(char):
    return char in alternative_dash


def is_dual_quotation(char):
    return char in dual_quotation


def is_opening_quotation(char):
    return char in opening_quotation


def is_closing_quotation(char):
    return char in closing_quotation


def is_quotation(char):
    return (
        is_opening_quotation(char)
        or is_closing_quotation(char)
        or is_dual_quotation(char)
    )
