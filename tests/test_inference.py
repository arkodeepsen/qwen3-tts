import base64, io
import numpy as np
import soundfile as sf
from inference import synthesize

class FakeModel:
    """Returns 1 sample-second of audio per call so durations are deterministic."""
    def __init__(self): self.calls = []; self.seeds = []
    def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
        import torch
        self.calls.append(text)
        self.seeds.append(torch.initial_seed())
        n = 100  # 100 samples @ sr=100 => 1.0s
        return [np.full(n, 0.2, dtype=np.float32)], 100

def _prompt():  # opaque to synthesize; just passed through
    return [object()]

def test_single_chunk_wav():
    m = FakeModel()
    out = synthesize(_prompt(), "Hello world.", "English", response_format="wav", model=m)
    assert out["format"] == "wav" and out["sample_rate"] == 100
    assert out["chunks"] == 1
    wav, sr = sf.read(io.BytesIO(base64.b64decode(out["audio_base64"])), dtype="float32")
    assert sr == 100 and len(wav) == 100
    assert out["srt"] is None and out["segments"] is None

def test_multi_sentence_packs_without_srt():
    m = FakeModel()
    out = synthesize(_prompt(), "A. B. C.", "English", return_srt=False, model=m)
    # short sentences pack into a single <=200 char unit -> 1 generate call
    assert len(m.calls) == 1 and out["chunks"] == 1

def test_srt_is_sentence_level():
    m = FakeModel()
    out = synthesize(_prompt(), "One. Two. Three.", "English", return_srt=True, model=m)
    assert len(m.calls) == 3            # one generate per sentence
    assert out["segments"][0]["text"] == "One."
    assert out["segments"][1]["start"] > 0
    assert out["srt"].startswith("1\n00:00:00,000 -->")

def test_seed_is_applied_each_chunk():
    m = FakeModel()
    synthesize(_prompt(), "One. Two.", "English", seed=123, return_srt=True, model=m)
    assert set(m.seeds) == {123}        # same fixed seed reset before each chunk
