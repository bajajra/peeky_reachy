import numpy as np

from peeky_reachy.audio.io import ArrayAudioIO
from peeky_reachy.config import Config
from peeky_reachy.pipeline import Pipeline
from tests.conftest import SR, baby_cry, silence


def _audio(sig):
    return (SR, (np.clip(sig, -1, 1) * 32767).astype(np.int16))


def test_array_audio_io_streams_and_records():
    io = ArrayAudioIO(baby_cry(1.0), SR, sample_rate=SR, frame_size=1536)
    io.start()
    frames = 0
    while io.read() is not None:
        frames += 1
    assert frames > 0
    io.play(np.zeros(100, dtype=np.float32), SR)
    assert io.played and io.played[0][1] == SR


def test_analyze_triggers_on_cry():
    from peeky_reachy.webapp import analyze

    audio = _audio(np.concatenate([silence(2.5), baby_cry(5.0)]))
    summary, soothe_audio, rows, _ = analyze(audio, 0.55, 3.0, 30.0, 3.0, 5, 0.5,
                                             False, False, "")
    assert "Soothing triggered" in summary
    assert len(rows) > 0


def test_analyze_silence_no_trigger():
    from peeky_reachy.webapp import analyze

    audio = _audio(silence(7.0))
    summary, _, _, _ = analyze(audio, 0.55, 3.0, 30.0, 3.0, 5, 0.5, False, False, "")
    assert "No soothing triggered" in summary


def test_pipeline_window_callback_and_event_audio(tmp_path):
    cfg = Config.from_env()
    cfg.assets_dir = str(tmp_path / "none")
    io = ArrayAudioIO(np.concatenate([silence(2.5), baby_cry(5.0)]), SR,
                      cfg.sample_rate, cfg.frame_size)
    windows = []
    events = []
    Pipeline(cfg, io, voice_client=None,
             on_window=lambda *a: windows.append(a),
             on_soothe=events.append).run()
    assert len(windows) > 0
    assert len(events) >= 1
    assert events[0].at_seconds > 0
