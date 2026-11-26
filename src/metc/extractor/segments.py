import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import diff_match_patch as dmp_module

from ..merge import _java_split_space
from .constants import (
    closing_punctuation_marks,
    colon_punctuation_marks,
    punctuation_marks,
    whitespace_and_newlines,
)
from .metc_diff_extension import diffByWords
from .tokenizer import tokenize_text


def _utf16_len(s: str) -> int:
    # Java/UTF-16 code unit count. Non-BMP code points (incl. all emoji) take
    # 2 code units in Java's String, 1 in a Python str — so the position
    # arithmetic below must use this count, not len(), to stay aligned with
    # the Java caller's UTF-16 indexing of the same source string.
    return len(s) + sum(1 for c in s if ord(c) > 0xFFFF)


# Whitespace code points that LLM rewrites typically normalize to U+0020.
# We apply the same upfront on the source so the diff doesn't see a fake
# "exotic-space -> regular-space" edit, which the splitter would otherwise
# turn into overlapping annotations alongside any nearby word edit.
#
# Excludes line-break-like chars (LF U+000A, NEL U+0085, LINE SEPARATOR
# U+2028, PARAGRAPH SEPARATOR U+2029) — those carry paragraph-structure
# meaning and shouldn't be folded into spaces.
#
# Each entry is a single BMP code point replaced by another single BMP code
# point, so absolute positions are preserved 1:1 between the original and
# normalized source — annotation positions remain valid against the caller's
# original source string.
_NORMALIZABLE_SPACES = frozenset(
    {
        "\u0009",  # CHARACTER TABULATION (HT/Tab)
        "\u000b",  # LINE TABULATION (VT)
        "\u000c",  # FORM FEED (FF)
        "\u000d",  # CARRIAGE RETURN (CR)
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
        "\u202f",  # NARROW NO-BREAK SPACE
        "\u205f",  # MEDIUM MATHEMATICAL SPACE
        "\u3000",  # IDEOGRAPHIC SPACE
    }
)


def _normalize_source_spaces(source: str, target: str) -> str:
    # Per-char guard: if a normalizable char also appears anywhere in target,
    # leave source occurrences alone so we don't manufacture a fake diff
    # against a target that genuinely uses the exotic whitespace.
    chars_to_replace = _NORMALIZABLE_SPACES - set(target)
    if not chars_to_replace:
        return source
    return "".join(" " if c in chars_to_replace else c for c in source)


@dataclass
class ReplacementSegment:
    pass


@dataclass
class ReplaceSegment(ReplacementSegment):
    source: str
    target: str


@dataclass
class RetainSegment(ReplacementSegment):
    text: str


@dataclass
class TokenizedReplaceSegment:
    source_tokens: List[str]
    target_tokens: List[str]

    @property
    def source(self) -> str:
        return " ".join(self.source_tokens)

    @property
    def target(self) -> str:
        return " ".join(self.target_tokens)


def get_replacement_segments(source, target) -> List[ReplacementSegment]:
    """Split punctuation from core text when it's only at leading/trailing edges.

    Rules:
    - Source/target punctuation must be only leading OR only trailing (not both, not in middle)
    - Closing punctuation (. ? ! etc.) at leading position prevents split
    - Split into: leading punct segment + core segment + trailing punct segment
    - Shared whitespace between edges is preserved as RetainSegment
    """
    if not (source and target):
        return [ReplaceSegment(source=source, target=target)]

    EDGE = re.escape("".join(punctuation_marks | {" "}))

    def split_edges(text: str) -> Tuple[str, str, str]:
        lead = re.match(rf"^[{EDGE}]+", text)
        trail = re.search(rf"[{EDGE}]+$", text)
        l, r = lead.end() if lead else 0, trail.start() if trail else len(text)
        return (text[:l], text[l:r], text[r:]) if l < r else ("", text, "")

    (sl, sc, st), (tl, tc, tt) = split_edges(source), split_edges(target)

    bail = (
        (set(sc) | set(tc)) & punctuation_marks
        or (sl and st)
        or (tl and tt)
        or (set(sl) | set(tl)) & (closing_punctuation_marks | colon_punctuation_marks)
    )
    if bail:
        return [ReplaceSegment(source=source, target=target)]

    def extract_shared_ws_from_start(
        src_edge: str, tgt_edge: str
    ) -> Tuple[str, str, str]:
        ws = 0
        max_ws = min(len(src_edge), len(tgt_edge))
        while ws < max_ws:
            src_ch, tgt_ch = src_edge[ws], tgt_edge[ws]
            if src_ch in whitespace_and_newlines and src_ch == tgt_ch:
                ws += 1
            else:
                break
        if ws:
            return src_edge[:ws], src_edge[ws:], tgt_edge[ws:]
        return "", src_edge, tgt_edge

    def extract_shared_ws_from_end(
        src_edge: str, tgt_edge: str
    ) -> Tuple[str, str, str]:
        ws = 0
        max_ws = min(len(src_edge), len(tgt_edge))
        while ws < max_ws:
            src_ch, tgt_ch = src_edge[-(ws + 1)], tgt_edge[-(ws + 1)]
            if src_ch in whitespace_and_newlines and src_ch == tgt_ch:
                ws += 1
            else:
                break
        if ws:
            return src_edge[-ws:], src_edge[:-ws], tgt_edge[:-ws]
        return "", src_edge, tgt_edge

    segments = []

    if sl or tl:
        shared_start_ws, sl_mid, tl_mid = extract_shared_ws_from_start(sl, tl)
        if shared_start_ws:
            segments.append(RetainSegment(text=shared_start_ws))
        shared_end_ws, sl_stripped, tl_stripped = extract_shared_ws_from_end(
            sl_mid, tl_mid
        )
        if sl_stripped or tl_stripped:
            segments.append(ReplaceSegment(source=sl_stripped, target=tl_stripped))
        if shared_end_ws:
            segments.append(RetainSegment(text=shared_end_ws))

    segments.append(ReplaceSegment(source=sc, target=tc))

    if st or tt:
        shared_start_ws, st_mid, tt_mid = extract_shared_ws_from_start(st, tt)
        if shared_start_ws:
            segments.append(RetainSegment(text=shared_start_ws))
        shared_end_ws, st_stripped, tt_stripped = extract_shared_ws_from_end(
            st_mid, tt_mid
        )
        if st_stripped or tt_stripped:
            segments.append(ReplaceSegment(source=st_stripped, target=tt_stripped))
        if shared_end_ws:
            segments.append(RetainSegment(text=shared_end_ws))

    return segments


def get_replacement_segments_tokenized(
    source_tokens: List[str], target_tokens: List[str]
) -> List[TokenizedReplaceSegment]:
    if not (source_tokens and target_tokens):
        return [
            TokenizedReplaceSegment(
                source_tokens=list(source_tokens) if source_tokens else [],
                target_tokens=list(target_tokens) if target_tokens else [],
            )
        ]

    def is_pure_punct(tok: str) -> bool:
        return bool(tok) and all(c in punctuation_marks for c in tok)

    def is_closing_punct(tok: str) -> bool:
        return bool(tok) and all(c in closing_punctuation_marks for c in tok)

    def is_colon_punct(tok: str) -> bool:
        return bool(tok) and all(c in colon_punctuation_marks for c in tok)

    def has_punct(tok: str) -> bool:
        return any(c in punctuation_marks for c in tok)

    def split_edge_tokens(
        tokens: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        if not tokens:
            return [], [], []

        lead_end = 0
        while lead_end < len(tokens) and is_pure_punct(tokens[lead_end]):
            lead_end += 1

        trail_start = len(tokens)
        while trail_start > lead_end and is_pure_punct(tokens[trail_start - 1]):
            trail_start -= 1

        return tokens[:lead_end], tokens[lead_end:trail_start], tokens[trail_start:]

    sl, sc, st = split_edge_tokens(source_tokens)
    tl, tc, tt = split_edge_tokens(target_tokens)

    core_has_punct = any(is_pure_punct(t) for t in sc + tc)
    has_both_edges = (sl and st) or (tl and tt)
    leading_has_closing = any(is_closing_punct(t) for t in sl + tl) or any(
        is_colon_punct(t) for t in tl
    )
    source_too_short = len(source_tokens) <= 1

    if core_has_punct or has_both_edges or leading_has_closing or source_too_short:
        return [
            TokenizedReplaceSegment(
                source_tokens=list(source_tokens), target_tokens=list(target_tokens)
            )
        ]

    segments = []

    if sl or tl:
        segments.append(TokenizedReplaceSegment(source_tokens=sl, target_tokens=tl))

    segments.append(TokenizedReplaceSegment(source_tokens=sc, target_tokens=tc))

    if st or tt:
        segments.append(TokenizedReplaceSegment(source_tokens=st, target_tokens=tt))

    return segments


def _words_and_gaps(text: str) -> Tuple[List[str], List[str]]:
    """Split text into tokens and the whitespace/empty runs around them.

    Tokens come from METC's tokenizer, so edge punctuation (commas, parens,
    periods) is its own token while internal punctuation (hyphens, apostrophes)
    stays within a token; whitespace tokens are folded into the gaps. For
    punctuation-free text this matches a plain non-whitespace split. Splitting at
    edge punctuation lets the re-splitter below isolate an unchanged word from an
    adjacent punctuation edit (e.g. `yesterday` vs `yesterday,`), which a plain
    whitespace split would fuse into one annotation.

    Returns (words, gaps) where gaps has len(words) + 1 entries: gaps[i] is the
    run immediately before words[i], and gaps[-1] is the trailing run.
    Concatenating gaps[0] + words[0] + gaps[1] + ... + words[-1] + gaps[-1]
    reproduces `text` exactly, so absolute positions are preserved.
    """
    words: List[str] = []
    gaps: List[str] = []
    cur = ""
    prev_end = 0
    for _token, start, end in tokenize_text(text):
        cur += text[prev_end:start]
        prev_end = end
        piece = text[start:end]
        if piece.strip() == "":  # whitespace token -> fold into the gap
            cur += piece
        else:
            gaps.append(cur)
            words.append(piece)
            cur = ""
    cur += text[prev_end:]
    gaps.append(cur)
    return words, gaps


def _word_level_diff(
    source_words: List[str], target_words: List[str]
) -> List[Tuple[int, List[str]]]:
    """Word-granularity diff with each word treated as an atomic unit and no
    semantic cleanup. Returns a list of (op, words) where op is 0/1/-1 for
    equal/insert/delete. Mirrors the char-encoding approach used by
    compute_annotated_edits_result_tokenized.
    """
    dmp = dmp_module.diff_match_patch()
    word_array = [""]
    word_hash: Dict[str, int] = {}

    def encode(words: List[str]) -> str:
        chars = []
        for w in words:
            if w not in word_hash:
                word_array.append(w)
                word_hash[w] = len(word_array) - 1
            chars.append(chr(word_hash[w]))
        return "".join(chars)

    diffs = dmp.diff_main(encode(source_words), encode(target_words), False)
    return [(op, [word_array[ord(c)] for c in chunk]) for op, chunk in diffs]


def split_change_block(
    delete_text: str, insert_text: str
) -> List[Tuple[str, str, str]]:
    """Re-split a single change block (one accumulated delete/insert pair) at
    unchanged words.

    diff_match_patch's semantic cleanup fuses small unchanged words into the
    surrounding edit, producing one annotation that spans untouched words (e.g.
    `{ its a weird looking=>, it's a weird-looking}`). This re-splits the block
    at every unchanged word, so the example becomes
    `{ its=>, it's} a {weird looking=>weird-looking}`.

    Splitting also happens when an unchanged word borders an inserted or deleted
    word: `{curve will deviate from=>the curve deviates from a}` becomes
    `{=>the} curve {will deviate=>deviates} from {=>a}`. The unchanged word
    keeps the whitespace it shares with both sides; the inserted/deleted word —
    together with the whitespace that exists on only one side — becomes a
    (possibly zero-width) edit. Pure insertions already emit zero-width edits on
    this path, so this introduces no new annotation shape.

    The block is kept merged as a single edit when (a) content was moved/
    restructured -- a token appears on both the deleted and inserted side, e.g.
    a word-order rewrite -- since splitting would un-chunk an intentional
    restructuring merge; or (b) an unchanged word is genuinely ambiguous: a side
    bordering another unchanged word must have identical whitespace, and a side
    bordering a change must use only simple (<= one space) whitespace. Irregular
    spacing such as a stray double space keeps the block merged (e.g.
    `{realistinc  or more apporpuate scenatiorn=>realistic or a more appropriate
    scenario}`).

    Returns a list of segments: ("retain", text, None) for untouched runs and
    ("edit", source, target) for changed runs. The source side (text for retain,
    source for edit) concatenates back to `delete_text`; the target side to
    `insert_text`.
    """
    single = [("edit", delete_text, insert_text)]

    d_words, d_gaps = _words_and_gaps(delete_text)
    i_words, i_gaps = _words_and_gaps(insert_text)
    if not d_words or not i_words:
        return single

    word_diff = _word_level_diff(d_words, i_words)

    # A token appearing on BOTH the deleted and inserted side means content was
    # moved/restructured (a word-order rewrite), not locally edited. Re-splitting
    # such a block would un-chunk an intentional restructuring merge -- the
    # purpose of cleanupSemantic (see metc_diff_extension.diffByWords) -- so keep
    # it as one edit. Local edits have disjoint deleted/inserted token sets.
    deleted = {w for op, words in word_diff if op == -1 for w in words}
    inserted = {w for op, words in word_diff if op == 1 for w in words}
    if deleted & inserted:
        return single

    # Flatten the word diff into single-word ops, grouping consecutive changed
    # words into one edit segment and each unchanged word into its own retain.
    flat: List[Tuple[int, int, int, int]] = []  # (op, d_idx, i_idx, seg_id)
    seg_types: List[str] = []
    di = ii = 0
    in_edit = False
    for op, words in word_diff:
        for _ in words:
            if op == 0:
                seg_types.append("retain")
                flat.append((0, di, ii, len(seg_types) - 1))
                di += 1
                ii += 1
                in_edit = False
            else:
                if not in_edit:
                    seg_types.append("edit")
                    in_edit = True
                sid = len(seg_types) - 1
                if op == -1:
                    flat.append((-1, di, -1, sid))
                    di += 1
                else:
                    flat.append((1, -1, ii, sid))
                    ii += 1

    if not any(op == 0 for op, *_ in flat):
        return single

    # Keep the whole block merged if any unchanged word is genuinely ambiguous.
    # A side bordering another unchanged word / block boundary must match
    # exactly; a side bordering a change may differ but must be "simple" (a
    # clean insert/delete, not irregular spacing such as a double space).
    SIMPLE = {"", " "}
    for k, (op, dq, iq, _) in enumerate(flat):
        if op != 0:
            continue
        ld, li, td, ti = d_gaps[dq], i_gaps[iq], d_gaps[dq + 1], i_gaps[iq + 1]
        prev_change = k > 0 and flat[k - 1][0] != 0
        next_change = k < len(flat) - 1 and flat[k + 1][0] != 0
        if prev_change or k == 0:
            if ld not in SIMPLE or li not in SIMPLE:
                return single
        elif ld != li:
            return single
        if next_change or k == len(flat) - 1:
            if td not in SIMPLE or ti not in SIMPLE:
                return single
        elif td != ti:
            return single

    # An unchanged word "keeps" a surrounding gap (attaches it to its retain)
    # only when that gap is identical on the source and target sides; otherwise
    # the gap belongs to the adjacent edit.
    src_word_seg: Dict[int, Tuple[int, bool]] = {}
    tgt_word_seg: Dict[int, Tuple[int, bool]] = {}
    keep_lead_d: Dict[int, bool] = {}
    keep_trail_d: Dict[int, bool] = {}
    keep_lead_i: Dict[int, bool] = {}
    keep_trail_i: Dict[int, bool] = {}
    for op, dq, iq, sid in flat:
        if op in (0, -1):
            src_word_seg[dq] = (sid, op == 0)
        if op in (0, 1):
            tgt_word_seg[iq] = (sid, op == 0)
        if op == 0:
            keep_lead_d[dq] = keep_lead_i[iq] = d_gaps[dq] == i_gaps[iq]
            keep_trail_d[dq] = keep_trail_i[iq] = d_gaps[dq + 1] == i_gaps[iq + 1]

    nseg = len(seg_types)
    src_parts = [""] * nseg
    tgt_parts = [""] * nseg

    def assign(words, gaps, word_seg, keep_lead, keep_trail, parts):
        n = len(words)
        for j in range(n + 1):
            gap = gaps[j]
            if gap:
                left = word_seg.get(j - 1) if j > 0 else None
                right = word_seg.get(j) if j < n else None
                neighbour = left or right  # a gap always borders a word
                if neighbour is None:
                    continue
                if right and right[1] and keep_lead.get(j):
                    owner = right[0]  # retain keeps its leading gap
                elif left and left[1] and keep_trail.get(j - 1):
                    owner = left[0]  # retain keeps its trailing gap
                elif left and not left[1]:
                    owner = left[0]  # otherwise -> adjacent edit
                elif right and not right[1]:
                    owner = right[0]
                else:
                    owner = neighbour[0]
                parts[owner] += gap
            if j < n:
                parts[word_seg[j][0]] += words[j]

    assign(d_words, d_gaps, src_word_seg, keep_lead_d, keep_trail_d, src_parts)
    assign(i_words, i_gaps, tgt_word_seg, keep_lead_i, keep_trail_i, tgt_parts)

    segments: List[Tuple[str, str, str]] = []
    for sid in range(nseg):
        if seg_types[sid] == "retain":
            segments.append(("retain", src_parts[sid], None))
        else:
            segments.append(("edit", src_parts[sid], tgt_parts[sid]))

    # Defensive: a failed split would corrupt the absolute edit positions the
    # caller tracks, so fall back to the merged edit if reassembly is not exact.
    src = "".join(a for _, a, _ in segments)
    tgt = "".join(a if kind == "retain" else b for kind, a, b in segments)
    if src != delete_text or tgt != insert_text:
        return single

    return segments


def alter_render_punctuation(source, target):
    segments = get_replacement_segments(source, target)
    template = "{{{source}=>{target}}}"
    result = ""
    for segment in segments:
        if isinstance(segment, ReplaceSegment):
            result += template.format(source=segment.source, target=segment.target)
        elif isinstance(segment, RetainSegment):
            result += segment.text
    return result


def render_replacements(diffs: List[Tuple[int, str]]) -> str:
    render = ""
    current_add = ""
    current_del = ""
    for val, tok in diffs + [(0, "")]:
        if val == 0:
            if current_add or current_del:
                render += alter_render_punctuation(current_del, current_add)
            current_add = current_del = ""
            render += tok
        elif val == 1:
            current_add += tok
        else:
            current_del += tok
    return render


def compute_annotated_edits_result(
    source: str, target: str, cleanup_semantic: bool = False
) -> Dict[str, Any]:
    source = _normalize_source_spaces(source, target)
    diffs = diffByWords(source, target, diff_cleanup_semantic=cleanup_semantic)

    annotated_parts = []
    edits = []

    source_pos = 0
    current_delete = ""
    current_insert = ""

    for val, tok in diffs + [(0, "")]:
        if val == 0:
            if current_delete or current_insert:
                # With cleanup semantic, split the merged block at unchanged
                # words (see split_change_block); otherwise treat it as one edit.
                # A block-level retain (an unchanged word pulled out between
                # sub-edits) is routed through the same inner loop as a
                # RetainSegment, so positions stay aligned with the source — the
                # pieces concatenate back to current_delete exactly.
                if cleanup_semantic:
                    block_segments = split_change_block(current_delete, current_insert)
                else:
                    block_segments = [("edit", current_delete, current_insert)]

                for kind, block_delete, block_insert in block_segments:
                    if kind == "retain":
                        segments = [RetainSegment(text=block_delete)]
                    else:
                        segments = get_replacement_segments(block_delete, block_insert)
                        # On the cleanup path, don't accept a peel that orphans a
                        # pure insertion/deletion off a SINGLE-word change -- that
                        # would emit two edits in a row with no word to anchor the
                        # inserted punctuation (e.g. {Yall=>Y'all}{=>,}). Keep it
                        # grouped instead ({Yall=>Y'all,}, a Compound_Trivial-style
                        # edit). For a multi-word change, leave the peel alone: the
                        # punctuation stays a separate (free) Punctuation edit
                        # rather than being absorbed into the compound, which would
                        # promote its severity (Moderate -> Complex). A substitution
                        # peel (e.g. a quote change, both sides non-empty) is kept.
                        if (
                            cleanup_semantic
                            and len(block_delete.split()) <= 1
                            and any(
                                isinstance(s, ReplaceSegment)
                                and bool(s.source) != bool(s.target)
                                for s in segments
                            )
                        ):
                            segments = [
                                ReplaceSegment(source=block_delete, target=block_insert)
                            ]

                    for segment in segments:
                        if isinstance(segment, ReplaceSegment):
                            if segment.source == segment.target:
                                annotated_parts.append(segment.source)
                                source_pos += _utf16_len(segment.source)
                                continue

                            # `marker_text` (in the response `edit` field) stays
                            # `src=>tgt` for API compatibility. The annotated
                            # string sent to the classifier uses
                            # `[EDIT] src=>tgt [/EDIT]` (single outer space) —
                            # confirmed best for both seq-cls and NER heads on a
                            # 4-case sweep (incl. Polish diacritic). markActiveEdit
                            # on the Java side `.strip()`s the captured content so
                            # the seq-cls rebuild produces this same form.
                            marker_text = f"{segment.source}=>{segment.target}"
                            annotated_parts.append(
                                f"[EDIT] {segment.source}=>{segment.target} [/EDIT]"
                            )

                            start_pos = source_pos
                            end_pos = start_pos + _utf16_len(segment.source)

                            edits.append(
                                {
                                    "edit": marker_text,
                                    "start": start_pos,
                                    "end": end_pos,
                                    "source": segment.source,
                                    "target": segment.target,
                                }
                            )

                            source_pos += _utf16_len(segment.source)

                        elif isinstance(segment, RetainSegment):
                            annotated_parts.append(segment.text)
                            source_pos += _utf16_len(segment.text)

            current_delete = ""
            current_insert = ""

            annotated_parts.append(tok)
            source_pos += _utf16_len(tok)

        elif val == 1:
            current_insert += tok
        else:
            current_delete += tok

    return {"annotated": "".join(annotated_parts), "edits": edits}


def compute_annotated_edits_result_tokenized(
    source: str, target: str
) -> Dict[str, Any]:
    source_tokens = _java_split_space(source)
    target_tokens = _java_split_space(target)

    dmp = dmp_module.diff_match_patch()
    token_array = [""]
    token_hash = {}

    def tokens_to_chars(tokens):
        chars = []
        for token in tokens:
            if token in token_hash:
                chars.append(chr(token_hash[token]))
            else:
                token_array.append(token)
                token_hash[token] = len(token_array) - 1
                chars.append(chr(len(token_array) - 1))
        return "".join(chars)

    source_chars = tokens_to_chars(source_tokens)
    target_chars = tokens_to_chars(target_tokens)

    diffs = dmp.diff_main(source_chars, target_chars, False)

    annotated_parts = []
    edits = []
    token_pos = 0
    current_delete_tokens = []
    current_insert_tokens = []

    def flush_edit():
        nonlocal token_pos
        if current_delete_tokens or current_insert_tokens:
            if annotated_parts and not annotated_parts[-1].endswith(" "):
                annotated_parts.append(" ")

            segments = get_replacement_segments_tokenized(
                current_delete_tokens, current_insert_tokens
            )

            segment_token_pos = token_pos

            for segment in segments:
                num_segment_tokens = len(segment.source_tokens)

                if segment.source_tokens == segment.target_tokens:
                    if segment.source_tokens:
                        annotated_parts.append(segment.source)
                    segment_token_pos += num_segment_tokens
                    continue

                # See compute_annotated_edits_result for rationale.
                marker_text = f"{segment.source}=>{segment.target}"
                annotated_parts.append(
                    f"[EDIT] {segment.source}=>{segment.target} [/EDIT]"
                )

                edits.append(
                    {
                        "edit": marker_text,
                        "startToken": segment_token_pos,
                        "endToken": segment_token_pos + num_segment_tokens,
                        "source": segment.source,
                        "target": segment.target,
                    }
                )

                segment_token_pos += num_segment_tokens

            token_pos += len(current_delete_tokens)
            current_delete_tokens.clear()
            current_insert_tokens.clear()

    for op, encoded_text in diffs:
        decoded_tokens = [token_array[ord(c)] for c in encoded_text]

        if op == 0:
            flush_edit()
            if (
                annotated_parts
                and not annotated_parts[-1].endswith(" ")
                and decoded_tokens
            ):
                annotated_parts.append(" ")
            annotated_parts.append(" ".join(decoded_tokens))
            token_pos += len(decoded_tokens)
        elif op == -1:
            current_delete_tokens.extend(decoded_tokens)
        elif op == 1:
            current_insert_tokens.extend(decoded_tokens)

    flush_edit()

    annotated = re.sub(r" +", " ", "".join(annotated_parts)).strip()

    return {"annotated": annotated, "edits": edits}
