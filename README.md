# Podcast Editor download packs

Large, rarely-changing pieces of the Podcast Editor Windows installer, published here so the installer
itself stays small. The installer downloads a pack only when its component is selected (and only if that
exact pack isn't already installed), verifies its SHA-256 against `manifest.json`, and extracts it.

| Pack | Contents | Installed to |
|---|---|---|
| `cuda.7z` | NVIDIA cuBLAS, cuDNN 9 and CUDA runtime DLLs (from NVIDIA's `nvidia-*-cu12` wheels) | `cuda\` |
| `whisper-medium.7z` | faster-whisper medium (`Systran/faster-whisper-medium`) | `models\whisper-medium\` |
| `whisper-large-v3-turbo.7z` | faster-whisper large-v3-turbo (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`) | `models\whisper-large-v3-turbo\` |

Exact sources (Hugging Face revisions, wheel file names) are recorded in each release's `manifest.json`.
The models are MIT-licensed (OpenAI Whisper weights, converted to CTranslate2); the NVIDIA DLLs are
redistributed under NVIDIA's CUDA/cuDNN redistribution terms.

This repository contains no application code. To publish new packs: **Actions → Publish packs → Run workflow**
with a new tag (e.g. `packs-2`).
