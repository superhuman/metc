import re
from typing import List, Tuple

import diff_match_patch as dmp_module

from .tokenizer import tokenize_text


def diff_linesToWords(text1, text2):
    lineArray = []
    lineHash = {}
    lineArray.append("")

    def diff_linesToCharsMunge(text):
        chars = []

        for line, lineStart, lineEnd in tokenize_text(text):
            if line in lineHash:
                chars.append(chr(lineHash[line]))
            else:
                if len(lineArray) == maxLines:
                    line = text[lineStart:]
                    lineEnd = len(text)
                lineArray.append(line)
                lineHash[line] = len(lineArray) - 1
                chars.append(chr(len(lineArray) - 1))
                if len(lineArray) == maxLines:
                    break
        return "".join(chars)

    maxLines = 40000
    chars1 = diff_linesToCharsMunge(text1)
    maxLines = 65535
    chars2 = diff_linesToCharsMunge(text2)
    return (chars1, chars2, lineArray)


def _cleanup_semantic_post_process(diffs):
    result = []
    i = 0
    while i < len(diffs):
        cur = diffs[i]
        next = diffs[i + 1] if i + 1 < len(diffs) else None
        nnext = diffs[i + 2] if i + 2 < len(diffs) else None

        if next and cur[0] == -1 and next[0] == -1:
            result.append((-1, cur[1] + next[1]))
            i += 2
        elif next and cur[0] == 1 and next[0] == 1:
            result.append((1, cur[1] + next[1]))
            i += 2
        elif nnext and cur[0] == 1 and next[0] == -1 and nnext[0] == 1:
            result.append(next)
            result.append((1, cur[1] + nnext[1]))
            i += 3
        else:
            result.append(cur)
            i += 1
    return result


def diffByWords(original_text, new_text, diff_cleanup_semantic=False):
    if original_text == new_text:
        return [(0, original_text)]

    dmp = dmp_module.diff_match_patch()

    words_to_chars = diff_linesToWords(original_text, new_text)

    words1 = words_to_chars[0]
    words2 = words_to_chars[1]
    words_array = words_to_chars[2]
    diffs = dmp.diff_main(words1, words2, False)
    if diff_cleanup_semantic:
        dmp.diff_cleanupSemantic(diffs)
        diffs = _cleanup_semantic_post_process(diffs)

    dmp.diff_charsToLines(diffs, words_array)

    return diffs


def dmp_ced(s1: str, s2: str) -> Tuple[int, int, int]:
    dmp = dmp_module.diff_match_patch()
    s1 = re.sub(r"\s+", "", s1)
    s2 = re.sub(r"\s+", "", s2)
    return dmp.diff_levenshtein(dmp.diff_main(s1, s2, False)), len(s1), len(s2)


def dmp_wed(s1: str, s2: str) -> Tuple[int, int, int]:
    dmp = dmp_module.diff_match_patch()
    words1, words2, words_array = diff_linesToWords(s1, s2)

    space_words = set([chr(i) for i, w in enumerate(words_array) if w.isspace()])
    words1 = "".join([w for w in words1 if w not in space_words])
    words2 = "".join([w for w in words2 if w not in space_words])
    diffs = dmp.diff_main(words1, words2, False)
    return dmp.diff_levenshtein(diffs), len(words1), len(words2)
