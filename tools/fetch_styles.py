"""Install the two local style generators, verifying the exact bytes before publishing."""

import hashlib
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "models"
MODELS = {
    "whitebox.pth": (
        "77c8b7c275e2ff36d7fc718f3716cc381d08b82f2e2b01895c84b23a1368d76b",
        [
            "https://github.com/vinesmsuic/White-box-Cartoonization-PyTorch/raw/refs/heads/main/weights/sceneryonly.pth.tar",
            "https://raw.githubusercontent.com/vinesmsuic/White-box-Cartoonization-PyTorch/main/weights/sceneryonly.pth.tar",
        ],
    ),
    "lineart.pth": (
        "c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
        [
            "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
            "https://www.modelscope.cn/models/lllyasviel/Annotators/resolve/master/sk_model.pth",
            "https://modelscope.cn/models/AI-ModelScope/Annotators/resolve/master/sk_model.pth",
        ],
    ),
}


def fetch(name, expected, urls):
    target = ROOT / name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return
    for url in urls:
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as folder:
                candidate = Path(folder) / name
                with requests.get(url, timeout=(10, 90), stream=True) as response:
                    response.raise_for_status()
                    with candidate.open("wb") as file:
                        for block in response.iter_content(1 << 20):
                            file.write(block)
                if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                    raise ValueError("文件摘要不匹配")
                candidate.replace(target)
                return
        except (requests.RequestException, ValueError) as exc:
            print(f"{url}: {exc}")
    raise RuntimeError(f"未能取得已校验的 {name}")


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    for filename, (digest, sources) in MODELS.items():
        fetch(filename, digest, sources)
        print(filename, digest)
