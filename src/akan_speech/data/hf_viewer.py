from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError


BASE_URL = "https://datasets-server.huggingface.co"


def fetch_json(path: str, params: dict[str, str], *, retries: int = 8) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if attempt + 1 == retries:
                raise
            retry_after = int(exc.headers.get("Retry-After", "0") or 0)
            time.sleep(max(retry_after, min(60, 2**attempt)))
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"Unable to fetch {url}")


def first_audio_url(audio_cell: Any) -> str:
    if isinstance(audio_cell, list) and audio_cell:
        first = audio_cell[0] or {}
        return str(first.get("src") or "")
    if isinstance(audio_cell, dict):
        return str(audio_cell.get("src") or audio_cell.get("path") or "")
    return ""


def split_size(dataset: str, config: str, split: str) -> int:
    payload = fetch_json(
        "/rows",
        {"dataset": dataset, "config": config, "split": split, "offset": "0", "length": "1"},
    )
    return int(payload.get("num_rows_total") or 0)


def fetch_split_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
    workers: int = 2,
    cache_dir: str | Path = "data/processed/viewer_cache",
) -> list[dict[str, Any]]:
    total = split_size(dataset, config, split)
    offsets = list(range(0, total, page_size))

    def fetch_page(offset: int) -> list[dict[str, Any]]:
        cache_path = Path(cache_dir) / dataset.replace("/", "__") / config / split / f"{offset:08d}.json"
        if cache_path.exists():
            return list(json.loads(cache_path.read_text(encoding="utf-8")).get("rows", []))
        payload = fetch_json(
            "/rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": str(offset),
                "length": str(page_size),
            },
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return list(payload.get("rows", []))

    with ThreadPoolExecutor(max_workers=min(workers, len(offsets) or 1)) as executor:
        pages = list(executor.map(fetch_page, offsets))
    rows = [row for page in pages for row in page]
    rows.sort(key=lambda item: int(item.get("row_idx", 0)))
    if len(rows) != total:
        raise RuntimeError(f"Expected {total} {split} rows, received {len(rows)}")
    return rows
