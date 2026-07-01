import io, shutil, base64
import numpy as np
import soundfile as sf
import pytest
from audio import concat_waveforms, encode_audio, b64

def _tone(n=8000):
    return (0.1 * np.sin(np.linspace(0, 20, n))).astype(np.float32)

def test_concat_inserts_silence():
    a, bwav = _tone(1000), _tone(1000)
    out = concat_waveforms([a, bwav], sr=1000, gap=0.5)  # 0.5s @1000hz = 500 samples
    assert len(out) == 1000 + 500 + 1000
    assert np.allclose(out[1000:1500], 0.0)

def test_concat_single():
    a = _tone(1000)
    assert len(concat_waveforms([a], sr=1000, gap=0.5)) == 1000

def test_encode_wav_roundtrip():
    wav = _tone()
    data = encode_audio(wav, 8000, "wav")
    back, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == 8000 and len(back) == len(wav)

def test_encode_flac_roundtrip():
    wav = _tone()
    data = encode_audio(wav, 8000, "flac")
    back, sr = sf.read(io.BytesIO(data), dtype="float32")
    assert sr == 8000

def test_encode_unsupported():
    with pytest.raises(ValueError):
        encode_audio(_tone(), 8000, "aiff")

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_mp3_nonempty():
    data = encode_audio(_tone(), 8000, "mp3")
    assert isinstance(data, bytes) and len(data) > 0

def test_b64_roundtrip():
    assert base64.b64decode(b64(b"abc")) == b"abc"
