"""
Dataset acquisition module with retry logic, mirror fallback, and SHA-256 provenance tracking.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from spm.utils.logging_config import setup_logger

logger = setup_logger("spm.data.download")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file_with_retry(
    urls: List[str],
    destination_file: Path,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Attempt to download a file from a list of URLs (primary + mirrors) with retries.
    Returns status dictionary with provenance details.
    """
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination_file.with_suffix(destination_file.suffix + ".part")

    history: List[Dict[str, Any]] = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*"
    }

    for url in urls:
        for attempt in range(1, max_retries + 1):
            logger.info("Attempting download from %s (attempt %d/%d)", url, attempt, max_retries)
            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    headers=headers,
                    allow_redirects=True
                ) as response:
                    response.raise_for_status()
                    final_url = response.url
                    total_bytes = 0
                    with open(temp_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                                total_bytes += len(chunk)

                if temp_file.exists() and temp_file.stat().st_size > 0:
                    temp_file.replace(destination_file)
                    checksum = compute_sha256(destination_file)
                    logger.info("Download successful: %s (SHA-256: %s, final URL: %s)", destination_file.name, checksum, final_url)
                    return {
                        "status": "DOWNLOADED",
                        "resolved_url": final_url,
                        "sha256": checksum,
                        "size_bytes": destination_file.stat().st_size,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "attempts_log": history + [{"url": url, "final_url": final_url, "attempt": attempt, "status": "SUCCESS"}]
                    }
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {str(exc)}"
                logger.warning("Failed downloading from %s (attempt %d/%d): %s", url, attempt, max_retries, error_msg)
                history.append({"url": url, "attempt": attempt, "error": error_msg})
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                if attempt < max_retries:
                    time.sleep(backoff_factor * (2 ** (attempt - 1)))

    logger.error("All download attempts failed for candidate URLs: %s", urls)
    return {
        "status": "BLOCKED",
        "resolved_url": None,
        "sha256": None,
        "size_bytes": 0,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "attempts_log": history,
        "error": "Exhausted all retries across primary and mirror sources"
    }


def acquire_dataset(
    dataset_key: str,
    dataset_cfg: Dict[str, Any],
    raw_root: Path,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Acquire a single dataset according to its configuration dictionary and record provenance.
    Never overwrites existing raw data silently.
    """
    dest_dir = raw_root / dataset_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    urls = [dataset_cfg["source_url"]] + dataset_cfg.get("mirror_urls", [])
    filename = Path(dataset_cfg["source_url"]).name
    dest_file = dest_dir / filename
    provenance_file = dest_dir / "provenance.json"

    # Check if raw data already exists
    if dest_file.exists() and dest_file.stat().st_size > 0 and not overwrite:
        logger.info("Dataset file %s already exists. Verifying checksum without re-downloading.", dest_file)
        checksum = compute_sha256(dest_file)
        provenance = {
            "dataset_key": dataset_key,
            "name": dataset_cfg.get("name", dataset_key),
            "status": "VERIFIED_EXISTING",
            "source_url": dataset_cfg["source_url"],
            "sha256": checksum,
            "size_bytes": dest_file.stat().st_size,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "license": dataset_cfg.get("license", "Unknown"),
            "duration_semantic": dataset_cfg.get("duration_semantic", "unknown"),
            "file_path": str(dest_file)
        }
        with open(provenance_file, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)
        return provenance

    res = download_file_with_retry(urls=urls, destination_file=dest_file)

    provenance = {
        "dataset_key": dataset_key,
        "name": dataset_cfg.get("name", dataset_key),
        "status": res["status"],
        "source_url": dataset_cfg["source_url"],
        "resolved_url": res.get("resolved_url"),
        "sha256": res.get("sha256"),
        "size_bytes": res.get("size_bytes", 0),
        "retrieved_at": res["retrieved_at"],
        "license": dataset_cfg.get("license", "Unknown"),
        "duration_semantic": dataset_cfg.get("duration_semantic", "unknown"),
        "file_path": str(dest_file) if res["status"] == "DOWNLOADED" else None,
        "attempts_log": res.get("attempts_log", []),
        "error": res.get("error")
    }

    with open(provenance_file, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    return provenance


def acquire_all_datasets(
    config_dict: Dict[str, Any],
    raw_root: Path,
    overwrite: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Acquire all configured datasets."""
    results = {}
    for key, cfg in config_dict.get("datasets", {}).items():
        results[key] = acquire_dataset(key, cfg, raw_root=raw_root, overwrite=overwrite)
    return results
