from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from remotezip import RemoteZip

from backend.dataset.zenodo_client import ZenodoClient

RECORD_ID = 4010759
MANIFEST = ROOT / "datasets/metadata_cache/include50_manifest_v1.csv"
VIDEO_ROOT = ROOT / "datasets/include50/videos"
REPORT = ROOT / "datasets/include50/reports/staging_report.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("include50-staging")


def norm(path: str) -> str:
    p = path.replace("\\", "/").lstrip("./").casefold()
    suffix = PurePosixPath(p).suffix
    if suffix in {".mov", ".mp4", ".avi", ".m4v"}:
        p = p[:-len(suffix)] + ".video"
    return p


def safe_target(relative_path: str) -> Path:
    target = (VIDEO_ROOT / Path(relative_path.replace("\\", "/"))).resolve()
    if target != VIDEO_ROOT.resolve() and VIDEO_ROOT.resolve() not in target.parents:
        raise ValueError(f"Unsafe manifest path: {relative_path}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download matched videos after a complete, unambiguous scan.",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST, keep_default_na=False)
    required = {"video_path", "normalized_path", "split", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")

    targets = {}
    for row in df.itertuples(index=False):
        key = norm(row.video_path)
        if key in targets:
            raise ValueError(f"Duplicate manifest video path: {row.video_path}")
        targets[key] = row.video_path

    log.info("Manifest entries: %d", len(targets))
    log.info("Fetching Zenodo archive metadata...")

    client = ZenodoClient(RECORD_ID)
    client.fetch_metadata()
    raw_files = (client.metadata or {}).get("files", [])

    archives = []
    for info in raw_files:
        filename = str(info.get("key") or info.get("filename") or "")
        if not filename.casefold().endswith(".zip"):
            continue

        links = info.get("links") or {}
        url = links.get("download") or links.get("content")
        if not url:
            url = (
                "https://zenodo.org/records/"
                f"{RECORD_ID}/files/{quote(filename)}?download=1"
            )
        archives.append((filename, url))

    if not archives:
        raise RuntimeError("No ZIP archives found in Zenodo metadata.")

    log.info("ZIP archives to inspect: %d", len(archives))

    # Map each selected manifest path to its source archive and member.
    matches = defaultdict(list)

    for archive_name, url in archives:
        log.info("Scanning remote ZIP: %s", archive_name)
        try:
            with RemoteZip(url) as remote:
                for member in remote.namelist():
                    key = norm(member)
                    if key in targets:
                        matches[key].append((archive_name, url, member))
        except Exception as exc:
            raise RuntimeError(
                f"Could not inspect {archive_name}: {exc}"
            ) from exc

    missing = sorted(targets[key] for key in targets if key not in matches)
    ambiguous = {
        targets[key]: [entry[0] + "::" + entry[2] for entry in entries]
        for key, entries in matches.items()
        if len(entries) != 1
    }

    plan = {
        key: entries[0]
        for key, entries in matches.items()
        if len(entries) == 1
    }

    by_archive = defaultdict(list)
    for key, (archive_name, url, member) in plan.items():
        by_archive[archive_name].append((key, url, member))

    print("\n--- INCLUDE-50 staging scan ---")
    print(f"Manifest videos: {len(targets)}")
    print(f"Matched uniquely: {len(plan)}")
    print(f"Missing: {len(missing)}")
    print(f"Ambiguous: {len(ambiguous)}")
    print("\nSelected videos per archive:")
    for archive_name in sorted(by_archive):
        print(f"  {archive_name}: {len(by_archive[archive_name])}")

    report = {
        "record_id": RECORD_ID,
        "manifest_count": len(targets),
        "uniquely_matched": len(plan),
        "missing": missing,
        "ambiguous": ambiguous,
        "archives": {
            name: len(items) for name, items in sorted(by_archive.items())
        },
        "download_requested": args.download,
        "downloaded": 0,
        "already_present": 0,
        "download_errors": [],
    }

    if missing or ambiguous:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nDetails saved to: {REPORT}")
        print("No videos downloaded because the scan is incomplete or ambiguous.")
        return 2

    if not args.download:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("\nDry run complete. No video files downloaded.")
        print("Review the archive counts, then run with --download on the college PC.")
        return 0

    for archive_name in sorted(by_archive):
        items = by_archive[archive_name]
        url = items[0][1]
        log.info("Downloading %d selected videos from %s", len(items), archive_name)

        with RemoteZip(url) as remote:
            for key, _, member in items:
                target = safe_target(targets[key])
                if target.is_file() and target.stat().st_size > 0:
                    report["already_present"] += 1
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                try:
                    with remote.open(member) as source, temporary.open("wb") as dest:
                        shutil.copyfileobj(source, dest, length=1024 * 1024)

                    if temporary.stat().st_size == 0:
                        raise IOError("Downloaded file is empty.")

                    temporary.replace(target)
                    report["downloaded"] += 1
                except Exception as exc:
                    temporary.unlink(missing_ok=True)
                    report["download_errors"].append({
                        "video_path": targets[key],
                        "archive": archive_name,
                        "error": str(exc),
                    })
                    log.exception("Failed downloading %s", targets[key])

        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Download summary ---")
    print(f"Downloaded: {report['downloaded']}")
    print(f"Already present: {report['already_present']}")
    print(f"Errors: {len(report['download_errors'])}")
    print(f"Report: {REPORT}")
    return 0 if not report["download_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
