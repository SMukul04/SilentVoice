from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.landmark_extractor import LandmarkExtractor
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector
from backend.sign_recognition.normalizer import LandmarkNormalizer

MANIFEST = ROOT / "datasets/metadata_cache/include50_manifest_v1.csv"
VIDEO_ROOT = ROOT / "datasets/include50/videos"
OUTPUT_ROOT = ROOT / "datasets/include50/processed"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
REPORT_PATH = ROOT / "datasets/include50/reports/preprocessing_report.json"
MODEL_PATH = ROOT / "models/mediapipe/hand_landmarker.task"

SEQUENCE_LENGTH = 32
FEATURE_COUNT = 126

SPLITS = ("train", "val", "test")

EXPECTED_COUNTS = {
    "train": 689,
    "val": 77,
    "test": 192,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("include50")


# ---------------------------------------------------------
# ATOMIC FILE SAVING
# ---------------------------------------------------------


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".tmp")

    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    os.replace(temp_path, path)


# ---------------------------------------------------------
# MANIFEST LOADING AND VALIDATION
# ---------------------------------------------------------


def load_manifest() -> tuple[pd.DataFrame, list[str]]:

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(
        MANIFEST,
        keep_default_na=False,
    )

    required = {
        "video_path",
        "normalized_path",
        "split",
        "label",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df = df[df["split"].isin(SPLITS)].copy()

    if df.empty:
        raise ValueError("Manifest contains no recognized split entries.")

    if df["normalized_path"].duplicated().any():

        duplicates = (
            df.loc[
                df["normalized_path"].duplicated(keep=False),
                "normalized_path",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(f"Duplicate normalized paths found: {duplicates}")

    labels = sorted(df["label"].unique().tolist())

    if len(labels) != 50:
        raise ValueError(f"Expected 50 classes, found {len(labels)}.")

    counts = df["split"].value_counts().to_dict()

    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Unexpected split counts: {counts}; " f"expected {EXPECTED_COUNTS}"
        )

    return df, labels


def resolve_video(video_path: str) -> Path:

    return VIDEO_ROOT / Path(video_path.replace("\\", "/"))


def validate_metadata(
    df: pd.DataFrame,
    labels: list[str],
) -> None:

    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print(f"Total videos: {len(df)}")
    print(f"Classes: {len(labels)}")

    print("Split counts:")

    for split in SPLITS:

        subset = df[df["split"] == split]

        print(
            f"  {split}: {len(subset)} videos, " f"{subset['label'].nunique()} classes"
        )

    existing = sum(resolve_video(path).is_file() for path in df["video_path"])

    print(f"Videos currently present: {existing}/{len(df)}")

    print("Metadata validation: PASSED")


# ---------------------------------------------------------
# CHECKPOINT MANAGEMENT
# ---------------------------------------------------------


def checkpoint_path(
    split: str,
    video_path: str,
) -> Path:

    key = hashlib.sha256(video_path.encode("utf-8")).hexdigest()

    return CHECKPOINT_ROOT / split / f"{key}.npz"


def load_checkpoint(
    path: Path,
    expected_video_path: str,
    expected_label: int,
) -> np.ndarray | None:

    if not path.is_file():
        return None

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            sequence = np.asarray(
                data["X"],
                dtype=np.float32,
            )

            label = int(np.asarray(data["y"]).item())

            saved_path = str(np.asarray(data["video_path"]).item())

        if saved_path != expected_video_path:
            logger.warning(
                "Ignoring mismatched checkpoint: %s",
                path,
            )
            return None

        if label != expected_label:
            logger.warning(
                "Ignoring checkpoint with incorrect label: %s",
                path,
            )
            return None

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):
            logger.warning(
                "Ignoring checkpoint with invalid shape: %s",
                path,
            )
            return None

        if not np.isfinite(sequence).all():
            logger.warning(
                "Ignoring checkpoint containing invalid values: %s",
                path,
            )
            return None

        return sequence

    except Exception:

        logger.exception(
            "Could not read checkpoint. Regenerating: %s",
            path,
        )

        return None


def save_checkpoint(
    path: Path,
    sequence: np.ndarray,
    label: int,
    video_path: str,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:

        with os.fdopen(fd, "wb") as handle:

            np.savez_compressed(
                handle,
                X=np.asarray(
                    sequence,
                    dtype=np.float32,
                ),
                y=np.asarray(
                    label,
                    dtype=np.int64,
                ),
                video_path=np.asarray(
                    video_path,
                    dtype=str,
                ),
            )

        os.replace(
            temp_name,
            path,
        )

    except Exception:

        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass

        raise


# ---------------------------------------------------------
# VIDEO FEATURE EXTRACTION
# ---------------------------------------------------------


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

        # Sample 32 frames across the entire video.
        indices = sampler.sample([Path(str(i)) for i in range(total_frames)])

        frame_indices = [int(p.name) for p in indices]

        features = []

        last_timestamp = -1

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            ok, frame = cap.read()

            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed reading frame {frame_index} " f"from {video_path}"
                )

            source_ms = int(frame_index * 1000 / fps) if fps > 0 else frame_index

            timestamp_ms = max(
                source_ms,
                last_timestamp + 1,
            )

            detector._timestamp_ms = max(
                detector._timestamp_ms,
                timestamp_ms - 1,
            )

            last_timestamp = detector._timestamp_ms + 1

            detection = detector.detect(frame)

            frame_features = extractor.extract(detection)

            vector = np.asarray(
                normalizer.normalize(frame_features),
                dtype=np.float32,
            ).reshape(-1)

            if vector.shape != (FEATURE_COUNT,):
                raise ValueError(
                    f"Expected {FEATURE_COUNT} features, " f"got {vector.shape}"
                )

            features.append(vector)

        result = np.asarray(
            features,
            dtype=np.float32,
        )

        if result.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):
            raise ValueError(f"Unexpected sequence shape: {result.shape}")

        return result

    finally:

        cap.release()


# ---------------------------------------------------------
# FINAL NPZ GENERATION
# ---------------------------------------------------------


def write_split_npz(
    split: str,
    sequences: list[np.ndarray],
    labels: list[int],
    paths: list[str],
) -> Path:

    if not sequences:
        raise RuntimeError(f"No videos processed successfully for split '{split}'.")

    X_array = np.stack(sequences).astype(np.float32)

    y_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_ROOT / f"{split}.npz"

    fd, temp_name = tempfile.mkstemp(
        prefix=split + "_",
        suffix=".tmp",
        dir=str(OUTPUT_ROOT),
    )

    try:

        with os.fdopen(fd, "wb") as handle:

            np.savez_compressed(
                handle,
                X=X_array,
                y=y_array,
                video_paths=np.asarray(
                    paths,
                    dtype=str,
                ),
            )

        os.replace(
            temp_name,
            output_path,
        )

    except Exception:

        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass

        raise

    logger.info(
        "Saved %s: X=%s, y=%s",
        output_path.name,
        X_array.shape,
        y_array.shape,
    )

    return output_path


# ---------------------------------------------------------
# MAIN PREPROCESSING PIPELINE
# ---------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess INCLUDE-50 with " "per-video checkpoint/resume support."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=("Validate manifest and local video availability " "without processing."),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=("Optional maximum videos per split " "for a small smoke test."),
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or a positive integer.")

    df, class_names = load_manifest()

    validate_metadata(
        df,
        class_names,
    )

    if args.validate_only:
        return 0

    # Check that all expected videos exist.
    missing = [
        str(resolve_video(path))
        for path in df["video_path"]
        if not resolve_video(path).is_file()
    ]

    if missing:

        print(f"\nCannot preprocess: {len(missing)} videos are missing.")

        for path in missing[:10]:
            print(f"  {path}")

        return 2

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_to_index = {name: i for i, name in enumerate(class_names)}

    (OUTPUT_ROOT / "class_names.json").write_text(
        json.dumps(
            class_names,
            indent=2,
            ensure_ascii=False,
        ),
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
        "checkpointing": "per-video atomic NPZ",
        "splits": {},
        "errors": [],
    }

    atomic_write_json(
        REPORT_PATH,
        report,
    )

    with MediaPipeDetector(model_path=MODEL_PATH) as detector:

        for split in SPLITS:

            subset = df[df["split"] == split].copy()

            if args.limit > 0:
                subset = subset.head(args.limit)

            sequences = []
            y_values = []
            paths = []

            skipped = 0
            failed = 0

            logger.info(
                "Processing %s: %d videos. " "Existing checkpoints will be reused.",
                split,
                len(subset),
            )

            for position, row in enumerate(
                subset.itertuples(index=False),
                start=1,
            ):

                video_path = str(row.video_path)

                label_index = class_to_index[row.label]

                cp_path = checkpoint_path(
                    split,
                    video_path,
                )

                # Check whether this video was already processed.
                sequence = load_checkpoint(
                    cp_path,
                    video_path,
                    label_index,
                )

                if sequence is not None:

                    skipped += 1

                    logger.info(
                        "%s %d/%d | CHECKPOINT REUSED | %s",
                        split,
                        position,
                        len(subset),
                        video_path,
                    )

                else:

                    try:

                        sequence = extract_video_features(
                            resolve_video(video_path),
                            detector,
                            extractor,
                            normalizer,
                            sampler,
                        )

                        # Save immediately after successful extraction.
                        save_checkpoint(
                            cp_path,
                            sequence,
                            label_index,
                            video_path,
                        )

                        logger.info(
                            "%s %d/%d | CHECKPOINT SAVED | %s",
                            split,
                            position,
                            len(subset),
                            video_path,
                        )

                    except Exception as exc:

                        failed += 1

                        logger.exception(
                            "Failed: %s",
                            video_path,
                        )

                        report["errors"].append(
                            {
                                "split": split,
                                "video_path": video_path,
                                "error": str(exc),
                            }
                        )

                        report["splits"][split] = {
                            "requested": len(subset),
                            "completed": len(sequences),
                            "reused_checkpoints": skipped,
                            "failed": failed,
                            "status": "incomplete",
                        }

                        atomic_write_json(
                            REPORT_PATH,
                            report,
                        )

                        continue

                sequences.append(sequence)

                y_values.append(label_index)

                paths.append(video_path)

                # Save progress after every completed video.
                report["splits"][split] = {
                    "requested": len(subset),
                    "completed": len(sequences),
                    "reused_checkpoints": skipped,
                    "failed": failed,
                    "status": "in_progress",
                }

                atomic_write_json(
                    REPORT_PATH,
                    report,
                )

            # Do not generate a final NPZ if any video failed.
            if len(sequences) != len(subset):

                report["splits"][split] = {
                    "requested": len(subset),
                    "completed": len(sequences),
                    "reused_checkpoints": skipped,
                    "failed": failed,
                    "status": "incomplete",
                }

                atomic_write_json(
                    REPORT_PATH,
                    report,
                )

                logger.error(
                    "%s incomplete: %d/%d succeeded. "
                    "Fix failures and rerun. "
                    "Valid checkpoints will be reused.",
                    split,
                    len(sequences),
                    len(subset),
                )

                return 1

            # Generate final NPZ from all checkpoints.
            output_path = write_split_npz(
                split,
                sequences,
                y_values,
                paths,
            )

            report["splits"][split] = {
                "requested": len(subset),
                "samples_written": len(sequences),
                "X_shape": [
                    len(sequences),
                    SEQUENCE_LENGTH,
                    FEATURE_COUNT,
                ],
                "y_shape": [len(y_values)],
                "reused_checkpoints": skipped,
                "failed": failed,
                "status": "completed",
                "output": str(output_path.relative_to(ROOT)),
            }

            atomic_write_json(
                REPORT_PATH,
                report,
            )

    print(f"\nReport saved: {REPORT_PATH}")

    print(f"Failed videos: {len(report['errors'])}")

    print(f"Checkpoint directory: {CHECKPOINT_ROOT}")

    print("Review the report before training.")

    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
