"""Build the download packs for the Podcast Editor installer and write manifest.json.

Each pack is a .7z that the installer downloads only when its component is ticked, checks
against the SHA-256 in the manifest, and extracts into the install folder:

    cuda.7z                     -> {app}\\cuda                          cuBLAS + cuDNN 9 + CUDA runtime (NVIDIA)
    whisper-medium.7z           -> {app}\\models\\whisper-medium         faster-whisper medium
    whisper-large-v3-turbo.7z   -> {app}\\models\\whisper-large-v3-turbo faster-whisper large-v3-turbo

Sources are recorded exactly (Hugging Face commit, wheel file names) so a pack can be rebuilt.
Run on Linux with 7z available:  python build_packs.py --tag packs-1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

REPO = "grantsdale/gtpodcasteditor-packs"
WHISPER = {
    "whisper-medium": "Systran/faster-whisper-medium",
    "whisper-large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
WHISPER_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"]
# CTranslate2 4.x on Windows needs CUDA 12 cuBLAS and cuDNN 9 (same set the app was tested with).
CUDA_WHEELS = ["nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*", "nvidia-cuda-runtime-cu12"]


def log(m: str) -> None:
    print(f"[packs] {m}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def pack(name: str, src: Path, out: Path, dest: str, source: dict, level: int) -> dict:
    archive = out / f"{name}.7z"
    archive.unlink(missing_ok=True)
    # files at the archive root; non-solid so the installer can extract file by file
    if shutil.which("7z"):
        subprocess.run(["7z", "a", "-t7z", f"-mx={level}", "-ms=off", str(archive), "."], cwd=src, check=True,
                       stdout=subprocess.DEVNULL)
    else:                                   # no 7-Zip installed (local testing): pure-Python fallback
        import py7zr
        with py7zr.SevenZipFile(archive, "w", filters=[{"id": py7zr.FILTER_LZMA2, "preset": level}]) as z:
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(src).as_posix())
    info = {"file": archive.name, "url": f"https://github.com/{REPO}/releases/download/{{tag}}/{archive.name}",
            "sha256": sha256(archive), "size": archive.stat().st_size, "unpacked": folder_size(src),
            "dest": dest, "source": source}
    log(f"{archive.name}: {info['size'] / 2**20:.0f} MB (unpacked {info['unpacked'] / 2**20:.0f} MB)")
    return info


def build_whisper(name: str, repo: str, work: Path, out: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    rev = HfApi().model_info(repo).sha
    d = work / name
    snapshot_download(repo, revision=rev, local_dir=str(d), allow_patterns=WHISPER_FILES)
    shutil.rmtree(d / ".cache", ignore_errors=True)
    return pack(name, d, out, f"models\\{name}", {"huggingface": repo, "revision": rev}, level=1)


def build_cuda(work: Path, out: Path) -> dict:
    d = work / "cuda"
    d.mkdir(parents=True)
    wheels = work / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--platform", "win_amd64",
                    "--python-version", "3.12", "-d", str(wheels), *CUDA_WHEELS], check=True)
    names = []
    for whl in sorted(wheels.glob("*.whl")):
        names.append(whl.name)
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                if n.lower().endswith(".dll") and "/bin/" in n:
                    (d / Path(n).name).write_bytes(z.read(n))
    if not any(d.glob("cublas64_*.dll")) or not any(d.glob("cudnn*64_9.dll")):
        raise SystemExit("CUDA wheels didn't contain the expected DLLs")
    log(f"{len(list(d.glob('*.dll')))} CUDA DLLs from {', '.join(names)}")
    return pack("cuda", d, out, "cuda", {"wheels": names}, level=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        packs = {"cuda": build_cuda(work, out)}
        for name, repo in WHISPER.items():
            packs[name] = build_whisper(name, repo, work, out)
    for p in packs.values():
        p["url"] = p["url"].replace("{tag}", a.tag)
    manifest = {"schema": 1, "tag": a.tag, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "packs": packs}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("manifest.json written")


if __name__ == "__main__":
    main()
