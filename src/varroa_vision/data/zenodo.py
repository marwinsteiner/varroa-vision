"""Download Zenodo records with resume support and MD5 verification.

Usage:
    python -m varroa_vision.data.zenodo varroa_dataset
    python -m varroa_vision.data.zenodo ev2 --dest data/raw
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    size: int
    md5: str


@dataclass(frozen=True)
class ZenodoRecord:
    key: str
    record_id: int
    title: str
    license: str
    files: tuple[ZenodoFile, ...]

    def url(self, name: str) -> str:
        return f"https://zenodo.org/api/records/{self.record_id}/files/{name}/content"


# Sizes and checksums from https://zenodo.org/api/records/<id>, fetched 2026-10-02.
RECORDS: dict[str, ZenodoRecord] = {
    "varroa_dataset": ZenodoRecord(
        key="varroa_dataset",
        record_id=4085044,
        title="VarroaDataset (Schurischuster and Kampel 2020)",
        license="CC-BY-4.0",
        files=(
            ZenodoFile("gt.csv", 1_226_427, "46889d2689eac833c6eb89a95085b8ef"),
            ZenodoFile("train.zip", 703_410_525, "87cb443aa28650ef34e83b2d1b3fa26d"),
            ZenodoFile("val.zip", 163_050_592, "631318f802b059e5274b0b26c4c0a593"),
            ZenodoFile("test.zip", 291_715_596, "a510921655dae21c10b5389fa0cfeef5"),
        ),
    ),
    "ev2": ZenodoRecord(
        key="ev2",
        record_id=13771384,
        title="EV2 Dataset (EdgeVision against Varroa, 2024)",
        license="CC-BY-4.0",
        files=(ZenodoFile("dataset.zip", 1_084_450_229, "c626a1f198cf7d0f41eae9c2660b0985"),),
    ),
}

DEFAULT_RAW_DIR = Path("data/raw")


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    md5: str | None = None,
    chunk_size: int = 1 << 20,
    timeout: int = 60,
) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial file if the server allows it.

    A ``.part`` file is used while downloading and renamed on success. If ``md5`` is
    given the finished file is verified and a mismatch raises ``ValueError``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if md5 is None or md5sum(dest) == md5:
            logger.info("already present: {}", dest)
            return dest
        logger.warning("checksum mismatch, re-downloading: {}", dest)
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        if offset and resp.status_code != 206:
            logger.warning("server ignored Range header, restarting {}", dest.name)
            offset = 0
        mode = "ab" if offset else "wb"
        total = expected_size
        if total is None and "Content-Length" in resp.headers:
            total = offset + int(resp.headers["Content-Length"])
        with (
            part.open(mode) as fh,
            tqdm(total=total, initial=offset, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))

    if expected_size is not None and part.stat().st_size != expected_size:
        raise ValueError(
            f"{dest.name}: size {part.stat().st_size} != expected {expected_size}; "
            "re-run to resume"
        )
    if md5 is not None:
        actual = md5sum(part)
        if actual != md5:
            part.unlink()
            raise ValueError(f"{dest.name}: md5 {actual} != expected {md5}")
    part.rename(dest)
    return dest


def download_record(
    record: ZenodoRecord,
    dest_dir: Path | None = None,
    files: list[str] | None = None,
    verify: bool = True,
) -> list[Path]:
    """Download every file of ``record`` (or the named subset) into ``dest_dir``."""
    dest_dir = (dest_dir or DEFAULT_RAW_DIR) / record.key
    wanted = set(files) if files else None
    logger.info("{} ({}), license {}", record.title, record.record_id, record.license)
    paths = []
    for f in record.files:
        if wanted is not None and f.name not in wanted:
            continue
        paths.append(
            download_file(
                record.url(f.name),
                dest_dir / f.name,
                expected_size=f.size,
                md5=f.md5 if verify else None,
            )
        )
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", choices=sorted(RECORDS))
    parser.add_argument("--dest", type=Path, default=DEFAULT_RAW_DIR, help="root of raw data dir")
    parser.add_argument("--files", nargs="*", help="subset of file names to fetch")
    parser.add_argument("--no-verify", action="store_true", help="skip MD5 verification")
    args = parser.parse_args(argv)
    paths = download_record(RECORDS[args.record], args.dest, args.files, verify=not args.no_verify)
    for p in paths:
        logger.info("ok {}", p)


if __name__ == "__main__":
    main()
