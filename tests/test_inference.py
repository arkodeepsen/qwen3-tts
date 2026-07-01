import base64, io
import numpy as np
import soundfile as sf
from inference import synthesize

class FakeModel:
    """Returns 1 sample-second of audio per call so durations are deterministic.

    Records a fresh global-RNG draw per call so a test can *prove* the seed is
    reset before EVERY chunk: identical draws => reset each chunk; diverging
    draws => seeded only once (RNG advances between chunks)."""
    def __init__(self): self.calls = []; self.draws = []; self.last_kwargs = {}
    def generate_voice_clone(self, text, language, voice_clone_prompt, **kw):
        import torch
        self.calls.append(text)
        self.last_kwargs = kw
        self.draws.append(torch.rand(1).item())  # first draw after synthesize's manual_seed
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
    synthesize(_prompt(), "One. Two. Three.", "English", seed=123, return_srt=True, model=m)
    # Each chunk draws from the global RNG right after synthesize resets the seed.
    # Reset-before-every-chunk => all draws identical; seeded-once => draws diverge
    # as the RNG advances. This structurally proves per-chunk reset.
    assert len(m.draws) == 3
    assert len(set(m.draws)) == 1

def test_stability_params_forwarded():
    import config
    m = FakeModel()
    synthesize(_prompt(), "Hello.", "English", model=m)
    kw = m.last_kwargs
    assert kw["do_sample"] is True
    assert kw["repetition_penalty"] == config.REPETITION_PENALTY
    assert kw["top_p"] == config.TOP_P
    assert kw["temperature"] == config.TEMPERATURE
    assert kw["max_new_tokens"] == config.MAX_NEW_TOKENS

def test_per_request_params_override_defaults():
    import config
    m = FakeModel()
    synthesize(_prompt(), "Hello.", "English", temperature=0.5, repetition_penalty=1.3,
               top_p=0.95, top_k=20, max_new_tokens=256, model=m)
    kw = m.last_kwargs
    assert kw["temperature"] == 0.5
    assert kw["repetition_penalty"] == 1.3
    assert kw["top_p"] == 0.95
    assert kw["top_k"] == 20
    assert kw["max_new_tokens"] == 256

def test_unset_params_fall_back_to_config():
    import config
    m = FakeModel()
    synthesize(_prompt(), "Hello.", "English", temperature=0.5, model=m)  # only temperature set
    kw = m.last_kwargs
    assert kw["temperature"] == 0.5                        # override
    assert kw["repetition_penalty"] == config.REPETITION_PENALTY  # default
    assert kw["max_new_tokens"] == config.MAX_NEW_TOKENS          # default
