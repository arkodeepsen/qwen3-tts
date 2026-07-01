from srt import build_segments, segments_to_srt

def test_build_segments_no_gap():
    segs = build_segments(["A", "B"], [1.0, 2.0], gap=0.0)
    assert segs == [
        {"index": 1, "start": 0.0, "end": 1.0, "text": "A"},
        {"index": 2, "start": 1.0, "end": 3.0, "text": "B"},
    ]

def test_build_segments_with_gap():
    segs = build_segments(["A", "B"], [1.0, 1.0], gap=0.5)
    assert segs[1]["start"] == 1.5
    assert segs[1]["end"] == 2.5

def test_build_segments_length_mismatch():
    import pytest
    with pytest.raises(ValueError):
        build_segments(["A"], [1.0, 2.0])

def test_segments_to_srt_format():
    segs = [{"index": 1, "start": 0.0, "end": 1.25, "text": "Hi"}]
    out = segments_to_srt(segs)
    assert out == "1\n00:00:00,000 --> 00:00:01,250\nHi\n"
