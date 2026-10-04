"""Download Roboflow Universe projects as YOLOv8 exports.

Roboflow serves exports only to logged-in accounts, so a free account's API key is
required (Settings -> API keys), passed as ``--api-key`` or ``ROBOFLOW_API_KEY``.

    python -m varroa_vision.data.roboflow hofer_bees
    python -m varroa_vision.data.roboflow --list

The export endpoint ``https://api.roboflow.com/<workspace>/<project>/<version>/yolov8``
returns a JSON document with a signed zip link; the zip unpacks to the Roboflow layout
(``{train,valid,test}/{images,labels}`` plus ``data.yaml``), the same as BEEHIVE, so
``varroa_vision.data.beehive``-style converters apply.

Candidate projects, surveyed 2026-10-04 (all CC BY 4.0 unless noted). Most "varroa"
projects on Universe are re-uploads of VarroaDataset's tunnel crops or sticky-board
photos; the ones below add something.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger

from varroa_vision.data.zenodo import download_file


@dataclass(frozen=True)
class RoboflowProject:
    key: str
    workspace: str
    project: str
    version: int
    note: str


PROJECTS: dict[str, RoboflowProject] = {
    "hofer_bees": RoboflowProject(
        "hofer_bees", "andrew-hofer-1qh7e", "bees-ytrmp", 4,
        "6,423 comb photos with many bees per image; classes bee, mite, pollen, queen, queen_cell, varroa. "
        "The best public stand-in for phone-on-comb footage for stage 1. Its varroa images are mostly "
        "VarroaDataset crops plus a few internet photos; filter by file name before using them for stage 2.",
    ),
    "sammy_var_sahi": RoboflowProject(
        "sammy_var_sahi", "sammy-crveu", "var_sahi", 6,
        "665 images, varroa only, mixed sticky-board tiles and internet comb photos.",
    ),
    "bolo_beehive": RoboflowProject(
        "bolo_beehive", "bolo-q0wr5", "beehive-detection", 2,
        "436 internet images, classes bees and varroa mites, Public Domain.",
    ),
    "beproj_internet": RoboflowProject(
        "beproj_internet", "beproj", "varroa-mites-detection-internet-images", 1,
        "62 internet photos of bees with mites.",
    ),
}

VARROA_DATASET_PATTERN = "-mp4-bee_id_"  # VarroaDataset crops re-uploaded to Roboflow keep this in their names


def export_link(p: RoboflowProject, api_key: str, fmt: str = "yolov8") -> str:
    url = f"https://api.roboflow.com/{p.workspace}/{p.project}/{p.version}/{fmt}"
    r = requests.get(url, params={"api_key": api_key}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"{url}: HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        return data["export"]["link"]
    except KeyError as exc:
        raise RuntimeError(f"unexpected response for {p.key}: {str(data)[:200]}") from exc


def download(p: RoboflowProject, api_key: str, dest_root: Path) -> Path:
    dest = dest_root / p.key
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{p.project}-v{p.version}-yolov8.zip"
    if not archive.exists():
        link = export_link(p, api_key)
        download_file(link, archive)
    if not (dest / "data.yaml").exists():
        logger.info("extracting {}", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keys", nargs="*", choices=list(PROJECTS), help="projects to fetch")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    parser.add_argument("--dest", type=Path, default=Path("data/raw/roboflow"))
    args = parser.parse_args(argv)
    if args.list or not args.keys:
        for p in PROJECTS.values():
            print(f"{p.key:16s} {p.workspace}/{p.project} v{p.version}\n    {p.note}")
        return
    if not args.api_key:
        raise SystemExit("set ROBOFLOW_API_KEY or pass --api-key (free account: app.roboflow.com -> Settings -> API)")
    for key in args.keys:
        logger.info("wrote {}", download(PROJECTS[key], args.api_key, args.dest))


if __name__ == "__main__":
    main()
