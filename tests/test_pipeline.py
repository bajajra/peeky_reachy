from peeky_reachy.audio.io import FileAudioIO
from peeky_reachy.config import Config
from peeky_reachy.pipeline import Pipeline


def _run(wav_path, tmp_path):
    cfg = Config.from_env()
    cfg.assets_dir = str(tmp_path / "no_assets")   # force motion-only soothing
    audio = FileAudioIO(wav_path, cfg.sample_rate, cfg.frame_size)
    events = []
    pipe = Pipeline(cfg, audio, voice_client=None, on_soothe=events.append)
    pipe.run()
    return events


def test_sustained_cry_triggers_soothing(cry_wav, tmp_path):
    events = _run(cry_wav, tmp_path)
    assert len(events) >= 1
    assert events[0].decision.event.value == "baby_cry"
    assert events[0].used_clone is False   # no voice client -> fallback path


def test_silence_does_not_trigger(silence_wav, tmp_path):
    assert _run(silence_wav, tmp_path) == []


def test_speech_does_not_trigger_false_soothe(speech_wav, tmp_path):
    assert _run(speech_wav, tmp_path) == []
