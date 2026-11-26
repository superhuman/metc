import pytest

from metc.offsets import py_to_utf16_index, utf16_len, utf16_slice, utf16_to_py_index


def test_utf16_roundtrip_with_non_bmp_characters():
    text = "🇨🇴 hello 😂"
    assert utf16_len(text) == 13
    for python_index in range(len(text) + 1):
        offset = py_to_utf16_index(text, python_index)
        assert utf16_to_py_index(text, offset) == python_index


def test_utf16_slice_uses_java_boundaries():
    text = "🇨🇴hello"
    assert utf16_slice(text, 4, 9) == "hello"


def test_mid_surrogate_offset_is_rejected():
    with pytest.raises(ValueError, match="surrogate"):
        utf16_to_py_index("😂", 1)
