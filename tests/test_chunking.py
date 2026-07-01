from chunking import split_sentences, pack_sentences

def test_split_basic_punctuation():
    assert split_sentences("Hello world. How are you? Fine!") == \
        ["Hello world.", "How are you?", "Fine!"]

def test_split_cjk_and_newlines():
    assert split_sentences("你好。今天天气不错！\nBye.") == \
        ["你好。", "今天天气不错！", "Bye."]

def test_split_hard_splits_overlong_sentence():
    long = "word " * 100  # 500 chars, no terminal punctuation
    out = split_sentences(long, max_chars=200)
    assert all(len(s) <= 200 for s in out)
    assert "".join(out).replace(" ", "") == long.replace(" ", "")

def test_split_drops_empty():
    assert split_sentences("  . .. \n\n") == []

def test_pack_greedy():
    sents = ["A.", "B.", "C.", "D."]  # 2 chars each
    assert pack_sentences(sents, max_chars=5) == ["A. B.", "C. D."]

def test_pack_single_overlong_passes_through():
    assert pack_sentences(["x" * 250], max_chars=200) == ["x" * 250]
