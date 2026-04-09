# Changelog

## Unreleased

- fixed Windows installer Paddle dependency pinning so the CUDA 12.9 path installs cleanly again
- fixed installer AGENTS hint encoding so Chinese routing text is written as UTF-8 instead of mojibake
- improved executable discovery so `ocrx doctor` and searchable PDF generation prefer the current Python env and can detect local Tesseract installs outside PATH
- taught the installer to reuse local Tesseract, create a Codex bin shim, and bootstrap `chi_sim` language data best-effort for searchable PDFs

## v0.1.0

- packaged the local engineering-drawing OCR workflow into a public repo
- added `ocrsmart` and `ocrbest` presets for practical day-to-day use
- added native PDF parse-first routing
- added DWG/DXF direct text extraction with OCR fallback
- added Codex-friendly installer and wrapper commands
- added online bootstrap install and release ZIP support
