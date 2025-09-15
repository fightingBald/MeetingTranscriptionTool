#!/usr/bin/env python3
"""
Transcribe audio/video to text & SRT using OpenAI Whisper, with progress bar.

Usage:
  python transcribe_video.py video.mp4 --model small --language zh
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import whisper
from whisper.utils import get_writer
from tqdm import tqdm
import ffmpeg


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds using ffmpeg.probe."""
    try:
        probe = ffmpeg.probe(path)
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        print(f"[WARN] Unable to get audio duration: {e}")
        return 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Whisper Transcriber with progress bar")
    p.add_argument("input", help="Path to input media file (audio/video)")
    p.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium", "large"])
    p.add_argument("--language", default=None, help="Language code (zh, en, fr...)")
    p.add_argument("--output-dir", default="./transcripts")
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.input):
        sys.stderr.write(f"[ERR] File not found: {args.input}\n")
        sys.exit(1)

    # Device selection
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    fp16 = device == "cuda"

    print(f"[INFO] Loading model {args.model} on {device} (fp16={fp16}) …")
    model = whisper.load_model(args.model, device=device)

    # Get total duration (seconds) if available
    total_duration = get_audio_duration(args.input)

    # Create progress bar (if duration unknown, tqdm will be indeterminate)
    pbar = tqdm(total=total_duration if total_duration > 0 else None, unit="sec", desc="Transcribing", dynamic_ncols=True)

    # Note: the upstream whisper.transcribe call doesn't accept a per-segment callback in the
    # official repo, so we can't update the bar in real-time here without a custom streaming
    # transcription loop. We keep the bar for UX and mark it complete after transcription.
    result = model.transcribe(
        args.input,
        language=args.language,
        fp16=fp16,
        verbose=False
    )

    if total_duration > 0:
        pbar.n = total_duration
        pbar.refresh()
    pbar.close()

    # Output files
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = os.path.join(args.output_dir, f"{base_name}.{timestamp}")

    txt_path = f"{base_out}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip() + "\n")
    print(f"[OK] Wrote transcript: {txt_path}")

    srt_writer = get_writer("srt", args.output_dir)
    srt_writer(result, f"{base_out}")
    print(f"[OK] Wrote subtitles: {base_out}.srt")


if __name__ == "__main__":
    main()

