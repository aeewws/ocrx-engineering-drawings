# Contributing

## Scope

Contributions are welcome for:

- better engineering drawing OCR defaults
- CAD extraction improvements
- Windows install reliability
- batch workflow polish
- documentation and reproducible benchmarks

## Ground Rules

- keep the public repo free of private drawings
- avoid hardcoded user paths
- prefer stable defaults over clever but fragile tuning
- keep command names simple for agents and shell users

## Local Dev

1. Clone the repo.
2. Run `scripts/install-codex.ps1`.
3. Test with your own local samples.
4. Verify `ocrx doctor` before submitting changes.
