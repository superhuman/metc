from pathlib import Path
import re
from typing import Set

emoji_sequences_path = Path(__file__).with_name("emoji-sequences.txt")


def extract_emoji_related_chars() -> Set[str]:
    result = []
    with emoji_sequences_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # get the first part of the line that represents
            # the range of emoji, emoji sequence, or single emoji
            emoji_part = line.split(";")[0].strip()
            if ".." in emoji_part:
                # search for the number of emoji in a "list"
                # e.g. we're looking for [2] in this line
                # 231A..231B ... ... ... [2] (⌚..⌛)
                n_emoji = re.search(r"\[(\d+)\]", line)
                assert n_emoji
                n_emoji = int(n_emoji.group(1))

                start, end_incl = emoji_part.split("..")
                # check that the range is valid
                assert int(start, 16) + n_emoji - 1 == int(end_incl, 16)

                for i in range(n_emoji):
                    result.append(chr(int(start, 16) + i))
            else:
                for c in emoji_part.split(" "):
                    assert c, f"Invalid emoji sequence: {emoji_part}"
                    result.append(chr(int(c, 16)))

    # Filter out ASCII characters in the collected set
    result = set(c for c in result if not c.isascii())
    return result


EMOJI_CHARS: Set[str] = set(extract_emoji_related_chars())


def is_emoji_char(c: str) -> bool:
    # \u200d (zero-width joiner) is used to join emoji together
    # e.g. "🧑‍🧑‍🧒" == "🧑\u200d🧑\u200d🧒"
    is_zwj = c == "\u200d"
    return c in EMOJI_CHARS or is_zwj
