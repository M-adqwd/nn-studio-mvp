"""从 hf-mirror 拉取 imagefolder 结构的数据集（无 Kaggle 凭据时的替代路径）。

用法：python download_labeled_folder.py <repo_id> <本地目录> <远端子目录>
远端子目录下的「类别目录/图片」会被还原成 <本地目录>/<类别>/<图片>。

镜像的 tree API 单次最多 1000 条且递归参数不生效，因此按两层逐个目录取。
"""
from __future__ import annotations

import io
import sys
import urllib.parse
from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com"
S = requests.Session()
S.headers.update({"User-Agent": "model-forge-bench/1.0"})


def list_dir(repo: str, path: str) -> list[dict]:
    url = f"{MIRROR}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"
    r = S.get(url, params={"limit": 1000}, timeout=90)
    r.raise_for_status()
    return r.json()


def fetch(repo: str, remote: str, local: Path) -> int:
    if local.exists() and local.stat().st_size > 0:
        return local.stat().st_size
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/datasets/{repo}/resolve/main/{urllib.parse.quote(remote)}"
    for attempt in range(3):
        try:
            with S.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                buf = io.BytesIO()
                for c in r.iter_content(65536):
                    buf.write(c)
            local.write_bytes(buf.getvalue())
            return len(buf.getvalue())
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
    return 0


def main() -> int:
    repo, dest, sub = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    classes = [x for x in list_dir(repo, sub) if x["type"] == "directory"]
    print(f"  类别目录 {len(classes)} 个: {[c['path'].split('/')[-1] for c in classes]}")
    total = n = 0
    for c in classes:
        cls = c["path"].split("/")[-1]
        files = [x for x in list_dir(repo, c["path"]) if x["type"] == "file"]
        if len(files) >= 1000:
            print(f"    ⚠ {cls} 命中 1000 条上限，可能未取全")
        for it in files:
            name = it["path"].split("/")[-1]
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            total += fetch(repo, it["path"], dest / cls / name)
            n += 1
        print(f"    {cls}: {len(files)} 张", flush=True)
    print(f"  完成 {n} 张 {total/1e6:.1f}MB → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
