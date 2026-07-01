import re

from client.longform import split_blocks


def _strip_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_multi_paragraph_splits_within_block_chars():
    paragraphs = [f"Paragraph {i}. " + ("word " * 20) for i in range(20)]
    text = "\n\n".join(paragraphs)
    blocks = split_blocks(text, block_chars=200)
    assert len(blocks) > 1
    assert all(len(b) <= 200 for b in blocks)


def test_single_giant_paragraph_sentence_split():
    # One paragraph, no blank lines, longer than block_chars -> must be
    # sentence-split (on '. ') into blocks each <= block_chars.
    sentence = "This is a reasonably long sentence about nothing in particular. "
    text = sentence * 30  # no blank lines anywhere
    assert "\n\n" not in text
    block_chars = 300
    blocks = split_blocks(text, block_chars=block_chars)
    assert len(blocks) > 1
    assert all(len(b) <= block_chars for b in blocks)


def test_content_preserved_ignoring_whitespace():
    text = (
        "First paragraph with some words here.\n\n"
        "Second paragraph, a bit longer than the first one, "
        "with multiple sentences. Here is another sentence! And one more?\n\n"
        "Third and final paragraph."
    )
    blocks = split_blocks(text, block_chars=60)
    assert len(blocks) >= 1
    joined = _strip_ws("".join(blocks))
    original = _strip_ws(text)
    assert joined == original


def test_drops_empty_and_whitespace_only_blocks():
    text = "\n\n\n   \n\nReal content here.\n\n   \n\n"
    blocks = split_blocks(text, block_chars=100)
    assert blocks == ["Real content here."]
    assert all(b.strip() for b in blocks)


def test_never_splits_mid_word_unless_word_exceeds_block_chars():
    long_sentence = " ".join(["word"] * 100) + "."
    blocks = split_blocks(long_sentence, block_chars=50)
    assert all(len(b) <= 50 for b in blocks)
    for b in blocks:
        for token in b.split(" "):
            assert token == "" or token == "word" or token == "word."


def test_single_word_exceeding_block_chars_is_hard_split():
    huge_word = "x" * 500
    blocks = split_blocks(huge_word, block_chars=100)
    assert all(len(b) <= 100 for b in blocks)
    assert _strip_ws("".join(blocks)) == huge_word
