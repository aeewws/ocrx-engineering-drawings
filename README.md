# OCRX Engineering Drawings

[![Release](https://img.shields.io/github/v/release/aeewws/ocrx-engineering-drawings?display_name=tag)](https://github.com/aeewws/ocrx-engineering-drawings/releases/latest)
[![License](https://img.shields.io/github/license/aeewws/ocrx-engineering-drawings)](./LICENSE)
[![Windows](https://img.shields.io/badge/platform-Windows-0078D6)](https://github.com/aeewws/ocrx-engineering-drawings)
[![Codex](https://img.shields.io/badge/agent-Codex-black)](https://github.com/aeewws/ocrx-engineering-drawings)

面向工程图纸的 Windows-first OCR/CAD 工具包。它把真实生产里最有用的两条路线封成了可直接调用的主入口:

- `ocrsmart` / `ocrsmartbatch`: faster default for most drawing work
- `ocrbest` / `ocrbestbatch`: higher-accuracy default for harder pages and mixed formats

The project packages a practical workflow for engineering drawings, scanned PDFs, native vector PDFs, DXF, and DWG.

It is designed for Codex users who want one install step and a small set of commands that agents can reliably reuse.

## Download

- Latest release page:
  `https://github.com/aeewws/ocrx-engineering-drawings/releases/latest`
- Direct release ZIP:
  `https://github.com/aeewws/ocrx-engineering-drawings/releases/latest/download/ocrx-engineering-drawings-windows.zip`
- Source ZIP:
  `https://github.com/aeewws/ocrx-engineering-drawings/archive/refs/heads/main.zip`

If you just want the fastest path on Windows, use the release ZIP or the online installer below.

## Fast Install

Online bootstrap install:

```powershell
$tmp = Join-Path $env:TEMP "ocrx-install-online.ps1"
Invoke-WebRequest "https://raw.githubusercontent.com/aeewws/ocrx-engineering-drawings/main/install-online.ps1" -OutFile $tmp
powershell -ExecutionPolicy Bypass -File $tmp
```

Local release ZIP install:

1. Download `ocrx-engineering-drawings-windows.zip` from Releases.
2. Extract it anywhere.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex.ps1
```

## Why This Repo

- Native PDF parse first, OCR second
- DWG/DXF direct text extraction first, OCR fallback second
- PaddleOCR GPU as the main heavy-duty OCR engine
- RapidOCR path kept available for fast fallback and mixed workloads
- Codex-friendly wrapper commands with stable defaults
- Batch commands for real folders instead of single-file demos

## Modes

| Mode | Command | Goal | Default profile |
| --- | --- | --- | --- |
| Smart | `ocrsmart` / `ocrsmartbatch` | Speed, stability, broad compatibility | `--detail auto --dpi 300 --min-width 2200` |
| Best | `ocrbest` / `ocrbestbatch` | Accuracy, harder pages, lower miss rate | `--detail full --dpi 320 --min-width 2600` |

Both modes route inputs automatically:

| Input type | Strategy |
| --- | --- |
| Searchable or vector PDF | native parse first, OCR only when needed |
| Scanned PDF / image | drawing OCR pipeline |
| DXF | direct CAD text extraction, then OCR fallback |
| DWG | ODA convert to DXF, then direct extraction plus OCR fallback |

## What You Get

- `tools/ocrx.py`: the main all-in-one OCR/CAD script
- `scripts/install-codex.ps1`: installs the tool into a Codex-friendly Windows setup
- `ocrx`, `ocrsmart`, `ocrbest`, `ocrcad` command wrappers
- `.ocr.txt` and `.ocr.json` outputs for both agent use and manual review
- optional AGENTS hint so new Codex threads discover the two drawing modes automatically

## Quick Start

1. Clone this repo.
2. Open PowerShell in the repo.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex.ps1
```

This installer will:

- create or reuse `~/.codex/envs/paddleocr-clean310`
- install the pinned OCR/PDF/CAD Python dependencies
- copy `ocrx.py` into `~/.codex/tools/ocrx`
- generate wrapper commands in `%LOCALAPPDATA%\OpenAI\Codex\bin`
- reuse a local Tesseract install for searchable PDFs when available, and bootstrap `chi_sim` language data when it can
- optionally append a short routing hint to `~/.codex/AGENTS.md`

## Primary Commands

```powershell
ocrsmart       # smart default for day-to-day engineering drawings
ocrbest        # stronger default for harder pages and higher recall
ocrsmartbatch  # smart whole-folder processing
ocrbestbatch   # stronger whole-folder processing
ocrcad         # explicit CAD route for DWG/DXF
ocrx doctor    # environment health check
```

## Usage

Single file:

```powershell
ocrsmart path\to\drawing.pdf
ocrbest path\to\drawing.pdf
ocrcad path\to\layout.dxf
```

Whole folder:

```powershell
ocrsmartbatch path\to\folder
ocrbestbatch path\to\folder
ocrcadbatch path\to\cad-folder
```

Direct low-level entry:

```powershell
ocrx auto path\to\input.pdf --profile drawing
ocrx cad path\to\layout.dwg
ocrx doctor
```

## Output

By default the tool writes into `.\ocr_output` next to where you run the command.

Each input can produce:

- `*.ocr.txt`: merged human-readable text
- `*.ocr.json`: structured metadata, page summaries, and OCR source tags
- optional searchable PDF outputs when PDF OCR mode is requested

Searchable PDF notes:

- OCRX uses `ocrmypdf` plus local Tesseract for this optional path
- the installer writes a `tesseract` shim into the Codex bin when a local Tesseract install is detected
- if Simplified Chinese data is missing locally, the installer attempts a best-effort `chi_sim` bootstrap for Windows-first use

## Native PDF Parsing

If a PDF already contains embedded vector text, `ocrx` reads that text directly with PDF parsers before spending GPU time on OCR.

That means:

- much faster handling for exported design PDFs
- lower OCR noise on title blocks and small labels
- better agent-facing text for indexing, search, and extraction

## CAD Routing

DXF and DWG do not go through naive image OCR first.

The CAD route is:

1. load DXF directly, or convert DWG to DXF with ODA File Converter when available
2. extract `TEXT`, `MTEXT`, `ATTRIB`, `ATTDEF`, `DIMENSION`, and `MLEADER`
3. render the drawing to a clean image
4. OCR only the remaining visual content
5. merge direct CAD text and OCR text conservatively

This is the strongest practical local route for mixed engineering drawings on a Windows workstation without moving to a heavier remote stack.

## Tuning

These environment variables are useful when you want to trade throughput, memory, and pipeline parallelism:

- `OCRX_AUTO_ENGINE=rapid`: lower-VRAM fallback when you explicitly prefer RapidOCR over Paddle
- `OCRX_PADDLE_REC_BATCH_SIZE=8`: smaller Paddle recognition batches for tighter throughput control
- `OCRX_DRAWING_PREPARE_WORKERS=1`: reduce CPU-side page preparation fan-out
- `OCRX_DRAWING_QUEUE_SIZE=2`: keep less prepared work buffered ahead of OCR

For most engineering drawing work, keep the defaults first and only tune when the workstation is under memory or concurrency pressure.

## ODA Note

This repo does not bundle ODA File Converter.

- DXF works without ODA
- DWG works best when ODA File Converter is installed locally
- the installer will detect ODA if it is already present

## Privacy

- No sample drawings are included in the repo
- No user-specific absolute paths are committed
- No telemetry or cloud upload is built into the tool
- Your files stay local unless you choose to sync the outputs elsewhere

## Release Notes

For version history, see [CHANGELOG.md](./CHANGELOG.md).

## Agent Hint

If you want new Codex threads to discover the two drawing presets automatically, the installer can maintain this short hint in `~/.codex/AGENTS.md`:

```md
工程图纸分两档：`智能扫描` 用 `ocrsmart` / `ocrsmartbatch`，默认优先速度、稳定性和通用性；`最强扫描` 用 `ocrbest` / `ocrbestbatch`，默认优先精度和兼容性。两档都会自动按格式分流：PDF/图片先解析再OCR，DWG/DXF先文字提取再OCR补扫。用户明确说“智能扫描”就走智能档；说“最强扫描/直接扫/用最强的”就走最强档。
```

## Support Matrix

- Windows: first-class target
- NVIDIA GPU: recommended for PaddleOCR-heavy workloads
- Codex CLI / Codex app shell use: supported
- Pure shell use without Codex: also supported after install

## Repository Layout

- `tools/ocrx.py`: OCR/PDF/CAD entrypoint
- `scripts/install-codex.ps1`: local installer into Codex paths
- `install-online.ps1`: download latest release and run installer
- `templates/AGENTS.snippet.md`: short routing hint for fresh threads
- `docs/PIPELINES.md`: high-level route explanation

## License

This repo is released under the MIT License. Dependency licenses remain their own.
