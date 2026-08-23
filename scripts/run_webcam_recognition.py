"""Launch SilentVoice real-time webcam recognition."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.realtime.webcam_recognizer import WebcamRecognizer


def main() -> None:
    """Create the production recognizer and start its webcam loop."""
    print("===================================")
    print("SILENTVOICE REAL-TIME RECOGNITION")
    print("===================================")
    print("\nControls:\n\nQ / ESC -> Quit\nR       -> Reset sequence\n")
    print("Starting webcam...")
    recognizer = WebcamRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
