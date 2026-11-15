# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-11-15

### Changed
- Docs: training-grade README and walkthrough. The README now has a plain-English problem statement, a pipeline
  diagram, key concepts, "What you will learn", a clearer synthetic label and an updated layout. New
  `docs/walkthrough.md`.
- README quick start: the `--monthly m.csv --daily d.csv` placeholders are replaced by the CSVs that
  `scripts/make_synthetic_data.py` writes, so every command runs as written.

## [1.1.0] - 2026-11-15

### Added
- `original/`: the original research code (read-only).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.

## [1.0.0] - 2026-11-04

### Added
- First public release: the maintained code, documentation, tests and CI.
