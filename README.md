MeetingTranscriptionTool
=======================

Lightweight CLI to transcribe audio/video to text and SRT using OpenAI Whisper.

Key features
- Transcribe media files with Whisper models (tiny/base/small/medium/large).
- Write plain-text transcript (.txt) and subtitles (.srt).
- Progress-bar UX (attempts to use media duration).

Prerequisites
- macOS / Linux / Windows with Python 3.9+ installed.
- ffmpeg installed and on PATH (install with `brew install ffmpeg` on macOS, or your platform package manager).
- (Optional) CUDA drivers + PyTorch GPU build to run larger models faster.

Install
1. Create and activate a virtual environment (recommended):
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate    # Windows

2. Install Python dependencies:
   pip install -r requirements.txt

Usage
- Basic (English audio):
  python transcribe_video.py /path/to/video.mp4 --model small --language en
- Basic (Chinese audio):
  python transcribe_video.py /path/to/video.mp4 --model small --language zh

- Outputs:
  - transcripts/<input_name>.<timestamp>.txt  (full transcript)
  - transcripts/<input_name>.<timestamp>.srt  (subtitles)

