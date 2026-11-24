"""Extract the official DAVIS 2017 TrainVal 480p archive and check CRCs on read."""

import argparse
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def extract(archive, dest):
    if (dest / "DAVIS").exists():
        raise FileExistsError("目标目录已存在，请另选输出位置。")
    with tempfile.TemporaryDirectory(dir=dest) as tmp, ZipFile(archive) as z:
        base = Path(tmp).resolve()
        for entry in z.infolist():
            path = (base / entry.filename).resolve()
            if not path.is_relative_to(base / "DAVIS"):
                raise ValueError(f"未知归档路径 {entry.filename}")
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        for split, n in (("train", 60), ("val", 30)):
            path = base / "DAVIS/ImageSets/2017" / (split + ".txt")
            if len(set(path.read_text().split())) != n:
                raise ValueError(f"{split} 划分数量不正确")
        (base / "DAVIS").rename(dest / "DAVIS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    extract(args.archive, ROOT / "data")
