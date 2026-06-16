from __future__ import annotations

import argparse
import json
from pathlib import Path

from akan_speech.data.manifest import read_jsonl
from akan_speech.data.quality import manifest_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Akan speech manifest.")
    parser.add_argument("manifest")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    report = manifest_quality_report(records)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

