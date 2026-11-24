"""Fetch a small reproducible candidate pool of native DIS5K train/validation pairs."""

import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlencode

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data/dis5k-wheels"
API = "https://api.kaggle.com/v1/datasets"
DATASET = "fineyouthpe/dis5k-dataset"


def fetch(url):
    return subprocess.check_output(
        ["curl.exe", "-fsSL", "--retry", "2", "--max-time", "70", url]
    )


def catalog():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "candidate-files.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    files = []
    args = {"pageSize": "1000"}
    while True:
        response = json.loads(fetch(f"{API}/list/{DATASET}?{urlencode(args)}"))
        files.extend(r["name"] for r in response["datasetFiles"])
        token = response.get("nextPageToken")
        if not token:
            break
        args["pageToken"] = token
    pairs = []
    for split, limit in (("DIS-TR", 45), ("DIS-VD", 12)):
        images = [
            f
            for f in files
            if f"/{split}/im/" in f
            and re.search(
                r"(?:Bicycle(?!Stand|Frame|Rack|Parking)|Tricycle|MountainBike|Unicycle|Rickshaw|RoadBike)",
                Path(f).name,
                re.IGNORECASE,
            )
        ]
        images.sort(key=lambda name: hashlib.sha256(name.encode()).hexdigest())
        for i, img in enumerate(images[:limit]):
            gt = img.replace("/im/", "/gt/").rsplit(".", 1)[0] + ".png"
            if gt not in files:
                raise ValueError(img)
            pairs.append(
                {
                    "id": f"{split[-2:].lower()}{i:02}",
                    "split": split,
                    "image": img,
                    "mask": gt,
                }
            )
    if not pairs:
        raise ValueError("No matched bicycle files")
    manifest.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    return pairs


def download(row):
    paths = []
    for role, ext in (("image", "jpg"), ("mask", "png")):
        path = ROOT / f"{row['id']}.{ext}"
        if not path.is_file():
            # Kaggle's single-file endpoint returns the original file for these members.
            payload = fetch(
                f"{API}/download/{DATASET}/{quote(row[role], safe='')}?datasetVersionNumber=1"
            )
            intermediate = path.with_suffix(f".{ext}.part")
            intermediate.write_bytes(payload)
            with Image.open(intermediate) as image:
                image.verify()
            intermediate.replace(path)
        paths.append(path)
    with Image.open(paths[0]) as rgb, Image.open(paths[1]) as mask:
        if rgb.size != mask.size:
            raise ValueError(row)
    return row["id"]


if __name__ == "__main__":
    pairs = catalog()
    print(f"{len(pairs)} candidate pairs", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key in pool.map(download, pairs):
            print(key, flush=True)
