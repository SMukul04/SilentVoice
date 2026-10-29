from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.normalizer import LandmarkNormalizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/metadata_cache/include50_manifest_v1.csv"
VIDEO_ROOT = ROOT / "datasets/include50/videos"
OUTPUT_ROOT = ROOT / "datasets/include50/processed"
MODEL_PATH = ROOT / "models/mediapipe/hand_landmarker.task"

SEQUENCE_LENGTH = 32
FEATURE_COUNT = 126
SPLITS = ("train", "val", "test")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("include50")


def load_manifest() -> tuple[pd.DataFrame, list[str]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST, keep_default_na=False)
    required = {"split", "video_path", "normalized_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df = df[df["split"].isin(SPLITS)].copy()

    if df.empty:
        raise ValueError("Manifest contains no recognized split entries.")

    if df["normalized_path"].duplicated().any():
        duplicates = df.loc[
            df["normalized_path"].duplicated(keep=False), "normalized_path"
        ].head(10).tolist()
        raise ValueError(f"Duplicate normalized paths found: {duplicates}")

    labels = sorted(df["label"].unique().tolist())
    if len(labels) != 50:
        raise ValueError(f"Expected 50 classes, found {len(labels)}.")

    counts = df["split"].value_counts().to_dict()
    expected = {"train": 689, "val": 77, "test": 192}
    if counts != expected:
        raise ValueError(f"Unexpected split counts: {counts}; expected {expected}")

    return df, labels


def resolve_video(video_path: str) -> Path:
    # The manifest path is relative to the dataset's original directory tree.
    return VIDEO_ROOT / Path(video_path.replace("\\", "/"))


def validate_metadata(df: pd.DataFrame, labels: list[str]) -> int:
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print(f"Total videos: {len(df)}")
    print(f"Classes: {len(labels)}")
    print("Split counts:")
    for split in SPLITS:
        subset = df[df["split"] == split]
        print(f"  {split}: {len(subset)} videos, {subset['label'].nunique()} classes")

    existing = sum(resolve_video(p).is_file() for p in df["video_path"])
    print(f"Videos currently present: {existing}/{len(df)}")
    print("Metadata validation: PASSED")
    return existing


def extract_video_features(
    video_path: Path,
    detector: MediaPipeDetector,
    extractor: LandmarkExtractor,
    normalizer: LandmarkNormalizer,
    sampler: FrameSampler,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        # Sample 32 frame indices spread across the entire video.
        indices = sampler.sample(
            [Path(str(i)) for i in range(total_frames)]
        )
        frame_indices = [int(p.name) for p in indices]

        features = []
        last_timestamp = -1

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()

            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed reading frame {frame_index} from {video_path}"
                )

            # Keep MediaPipe video timestamps strictly increasing.
            source_ms = (
                int(frame_index * 1000 / fps)
                if fps > 0
                else frame_index
            )
            timestamp_ms = max(source_ms, last_timestamp + 1)
            detector._timestamp_ms = max(
                detector._timestamp_ms,
                timestamp_ms - 1,
            )
            last_timestamp = detector._timestamp_ms + 1

            detection = detector.detect(frame)
            frame_features = extractor.extract(detection)
            vector = normalizer.normalize(frame_features)

            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vector.shape != (FEATURE_COUNT,):
                raise ValueError(
                    f"Expected {FEATURE_COUNT} features, got {vector.shape}"
                )
            features.append(vector)

        result = np.asarray(features, dtype=np.float32)
        if result.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
            raise ValueError(f"Unexpected sequence shape: {result.shape}")
        return result

    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess INCLUDE-50 videos into landmark sequences."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and local video availability without processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum videos per split for a small smoke test.",
    )
    args = parser.parse_args()

    df, class_names = load_manifest()
    existing = validate_metadata(df, class_names)

    if args.validate_only:
        return 0

    missing = [
        str(resolve_video(path))
        for path in df["video_path"]
        if not resolve_video(path).is_file()
    ]
    if missing:
        print(f"\nCannot preprocess: {len(missing)} videos are missing.")
        print("Example missing paths:")
        for path in missing[:10]:
            print(f"  {path}")
        print(
            "\nPlace the selected videos under datasets/include50/videos/"
            "while preserving their manifest-relative folder structure."
        )
        return 2

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    class_to_index = {name: i for i, name in enumerate(class_names)}

    (OUTPUT_ROOT / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sampler = FrameSampler(sequence_length=SEQUENCE_LENGTH)
    extractor = LandmarkExtractor()
    normalizer = LandmarkNormalizer()

    report = {
        "dataset": "INCLUDE-50",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": FEATURE_COUNT,
        "class_count": len(class_names),
        "splits": {},
        "errors": [],
    }

    with MediaPipeDetector(model_path=MODEL_PATH) as detector:
        for split in SPLITS:
            subset = df[df["split"] == split].copy()
            if args.limit > 0:
                subset = subset.head(args.limit)

            X, y, paths = [], [], []
            logger.info("Processing %s: %d videos", split, len(subset))

            for row in subset.itertuples(index=False):
                video = resolve_video(row.video_path)
                try:
                    sequence = extract_video_features(
                        video, detector, extractor, normalizer, sampler
                    )
                    X.append(sequence)
                    y.append(class_to_index[row.label])
                    paths.append(row.video_path)
                except Exception as exc:
                    logger.exception("Failed: %s", row.video_path)
                    report["errors"].append({
                        "split": split,
                        "video_path": row.video_path,
                        "error": str(exc),
                    })

            if not X:
                raise RuntimeError(
                    f"No videos processed successfully for split '{split}'."
                )

            X_array = np.stack(X).astype(np.float32)
            y_array = np.asarray(y, dtype=np.int64)
            output_path = OUTPUT_ROOT / f"{split}.npz"

            # Refuse to overwrite prior outputs, including smoke-test results.
            if output_path.exists():
                raise FileExistsError(
                    f"Output already exists: {output_path}. "
                    "Move it aside or choose a fresh output directory first."
                )

            np.savez_compressed(
                output_path,
                X=X_array,
                y=y_array,
                video_paths=np.asarray(paths, dtype=str),
            )

            report["splits"][split] = {
                "samples_written": len(X_array),
                "X_shape": list(X_array.shape),
                "y_shape": list(y_array.shape),
                "output": str(output_path.relative_to(ROOT)),
            }
            logger.info(
                "Saved %s: X=%s, y=%s",
                output_path.name,
                X_array.shape,
                y_array.shape,
            )

    report_path = ROOT / "datasets/include50/reports/preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nReport saved: {report_path}")
    print(f"Failed videos: {len(report['errors'])}")
    print("Review the report before training.")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
