"""Command-line entry points: peeky (run), peeky-demo (file), peeky-enroll."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Optional

from .config import Config
from .pipeline import Pipeline, SootheEvent
from .voice.clone_client import VoiceCloneClient
from .voice.enroll import CONSENT_TEXT, enroll_from_wav, record_and_enroll
from .voice.store import EnrollmentStore


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _make_voice(cfg: Config, enabled: bool) -> Optional[VoiceCloneClient]:
    if not enabled:
        return None
    store = EnrollmentStore(cfg.enrollment_dir)
    return VoiceCloneClient(cfg.voice_clone_url, store, cfg.voice_clone_timeout_s)


def _print_soothe(event: SootheEvent) -> None:
    d = event.decision
    reason = f"{d.reason.value} (~{d.reason_confidence:.0%})" if d.reason and d.reason.value != "unknown" else "n/a"
    print(f"  >> SOOTHE @ score={d.score:.2f} event={d.event.value} reason={reason} "
          f"clone={event.used_clone} played={event.played}")
    print(f"     phrase: \"{event.phrase}\"")


def demo_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="peeky-demo",
                                     description="File-driven e2e demo (no robot/GPU needed).")
    parser.add_argument("--wav", required=True, help="input wav to stream through the pipeline")
    parser.add_argument("--voice", action="store_true", help="attempt voice-clone synth via GPU service")
    parser.add_argument("--output-dir", default="output", help="where captured soothing playback is saved")
    parser.add_argument("--reason", action="store_true", help="enable weak cry-reason hinting")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from .audio.io import FileAudioIO

    cfg = Config.from_env()
    if args.reason:
        cfg.reason_hint_enabled = True
    audio = FileAudioIO(args.wav, cfg.sample_rate, cfg.frame_size, output_dir=args.output_dir)

    events: list[SootheEvent] = []
    pipe = Pipeline(cfg, audio, voice_client=_make_voice(cfg, args.voice),
                    on_soothe=lambda e: (events.append(e), _print_soothe(e)))
    print(f"Streaming {args.wav} ...")
    pipe.run()
    print(f"Done. {len(events)} soothe event(s) over {pipe.clock:.1f}s of audio.")
    return 0


def enroll_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="peeky-enroll",
                                     description="Enroll a caregiver's consented voice sample.")
    parser.add_argument("--name", required=True, help="caregiver display name (e.g. 'Mom')")
    parser.add_argument("--transcript", required=True, help="exact words spoken in the sample")
    parser.add_argument("--language", default="en")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="path to an existing voice sample wav")
    src.add_argument("--record", type=float, metavar="SECONDS", help="record from the mic for N seconds")
    parser.add_argument("--i-consent", action="store_true",
                        help="confirm caregiver consent (required)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.i_consent:
        print("Consent required to enroll a voice. Re-run with --i-consent.\n")
        print(CONSENT_TEXT)
        return 2

    cfg = Config.from_env()
    store = EnrollmentStore(cfg.enrollment_dir)
    if args.wav:
        rec = enroll_from_wav(store, wav_path=args.wav, display_name=args.name,
                              transcript=args.transcript, language=args.language,
                              consent_given=True)
    else:
        from .audio.io import LocalAudioIO

        audio = LocalAudioIO(cfg.sample_rate, cfg.frame_size)
        print(f"Recording {args.record:.0f}s — please read the transcript aloud now...")
        rec = record_and_enroll(store, audio, seconds=args.record, display_name=args.name,
                                transcript=args.transcript, language=args.language,
                                consent_given=True)
    print(f"Enrolled '{rec.display_name}' as id '{rec.speaker_id}'.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="peeky", description="Peeky baby/pet monitor.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="live monitoring (laptop mic; optional sim/robot motion)")
    run_p.add_argument("--no-voice", action="store_true", help="disable voice-clone synth")
    run_p.add_argument("--robot", action="store_true", help="connect to a Reachy Mini for motion")
    run_p.add_argument("--reason", action="store_true", help="enable weak cry-reason hinting")
    run_p.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("enroll", help="enroll a caregiver voice (see peeky-enroll -h)")
    sub.add_parser("demo", help="file-driven demo (see peeky-demo -h)")

    args, rest = parser.parse_known_args(argv)
    if args.command == "enroll":
        return enroll_main(rest)
    if args.command == "demo":
        return demo_main(rest)

    _setup_logging(args.verbose)
    cfg = Config.from_env()
    if args.reason:
        cfg.reason_hint_enabled = True

    from .audio.io import LocalAudioIO

    mini = None
    if args.robot:
        from reachy_mini import ReachyMini

        mini = ReachyMini().__enter__()

    audio = LocalAudioIO(cfg.sample_rate, cfg.frame_size)
    pipe = Pipeline(cfg, audio, mini=mini, voice_client=_make_voice(cfg, not args.no_voice),
                    on_soothe=_print_soothe)
    stop = threading.Event()
    print("Peeky is listening. Ctrl-C to stop. (Companion only — not a safety/medical monitor.)")
    try:
        pipe.run(stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
