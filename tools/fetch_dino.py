"""Resume verified ranged downloads when overseas artifact transfers reset."""

import hashlib
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = (
    "https://www.modelscope.cn/models/IDEA-Research/grounding-dino-tiny/resolve/master/"
)
FOLDER = ROOT / "models/grounding-dino-tiny"
PARTS = ROOT / ".install/dino-parts"
SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
SIZE = 689359096
BLOCK = 8 * 1024 * 1024


def get(offset):
    end = min(SIZE, offset + BLOCK) - 1
    final = PARTS / f"{offset:012d}"
    if final.exists() and final.stat().st_size == end - offset + 1:
        return final
    tmp = final.with_suffix(".part")
    for attempt in range(6):
        try:
            with requests.get(
                URL + "model.safetensors",
                params={"download": "true", "offset": offset},
                headers={
                    "Range": f"bytes={offset}-{end}",
                    "Accept-Encoding": "identity",
                },
                timeout=(12, 25),
                stream=True,
            ) as response:
                response.raise_for_status()
                if (
                    response.headers.get("Content-Range")
                    != f"bytes {offset}-{end}/{SIZE}"
                ):
                    raise ValueError("服务端未返回请求的字节范围")
                with tmp.open("wb") as f:
                    for data in response.iter_content(256 * 1024):
                        f.write(data)
            if tmp.stat().st_size != end - offset + 1:
                raise ValueError("下载片段不完整")
            tmp.replace(final)
            return final
        except (requests.RequestException, OSError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(8, 2**attempt))


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    dest = FOLDER / "model.safetensors"
    if dest.exists():
        with dest.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == SHA256:
                return
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        if not (FOLDER / name).exists():
            resp = requests.get(URL + name, timeout=30)
            resp.raise_for_status()
            resp.json()
            (FOLDER / name).write_bytes(resp.content)
    offsets = range(0, SIZE, BLOCK)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, _ in enumerate(pool.map(get, offsets), 1):
            if n % 10 == 0:
                print(f"{n}/{len(offsets)} blocks", flush=True)
    tmp = dest.with_suffix(".part")
    with tmp.open("wb") as f:
        for offset in offsets:
            with (PARTS / f"{offset:012d}").open("rb") as part:
                shutil.copyfileobj(part, f)
    with tmp.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != SHA256:
            raise ValueError("权重与原发布方的 SHA-256 不符")
    tmp.replace(dest)
    print(dest)


if __name__ == "__main__":
    main()
