# Changelog

## 0.4.0 — 2026-10-05

- Added a self-contained Windows desktop package: extract and launch without Python, an installed .NET runtime or build tools.
- Embedded the compiled generic .NET tensor interpreter in each native TIFF/GIF alongside the operator graph, model and current state. The bootstrap accepts only its compiled approved module hash and loads the interpreter from image bytes.
- Added native model/program mutation, exported-image continuation, hostile-input and independent numerical parity tests. The isolated player runs without Python files, authoring sidecars or a loose interpreter DLL.
- Added Execute, Run/Pause, Open, Reload and TIFF/GIF Export controls; exercised the live timer, keyboard handlers and export callbacks in package verification.
- Added a distinct NNFIELD3 / nnf-native/1 profile with bounded decimal-string model/constant values for exact cross-language packet canonicalization.
- Fixed numeric comparison tensors so subsequent arithmetic uses 0/1 values consistently in NumPy, scalar Python and .NET.
- Retained GPL corresponding source, official bundled-runtime licenses/notices, per-file manifests and explicit bootstrap/CPU boundaries.
- Disabled Git-revision assembly-version suffixes and verified identical interpreter bytes after rebuilding a fresh Git checkout at a different path.

## 0.3.0 — 2026-10-05

- Replaced recipe-only execution with image-carried operator graphs, model weights/bias, binary field state, tick, and step count.
- Added a generic bounded NumPy graph interpreter and independent Python scalar interpreter with STATE, PARAM, CONST, GATHER, DOT, ADD, MUL, SIGMOID, GE, and RETURN primitives.
- Runtime inference now loads its program, model and current state from pixels; it neither retrains a model nor selects a fixed host field rule.
- Refreshed TIFF/GIF outputs carry the accepted current state and advance it when executed again.
- Added TIFF/GIF behavior proofs: modifying only model data changes OR to AND, and modifying only graph offsets changes propagation to center-only behavior.
- Extended checkpoint integrity to include executable program content and hash.
- Increased bounded raster payloads to 16 KiB using one-pixel cells; old recipe-only carriers are rejected by this profile.
- Fixed inherited Windows output permissions when publishing staged directories.
- Regenerated examples, receipts, MSSL evidence, and the 21-test suite for the new execution model.

## 0.2.0 — 2026-10-05

- Split the neural demonstration into a standalone project with a default neural CLI and no sibling-folder dependencies.
- Added validated checkpoint continuation with bounded JSON, TIFF integrity checks, strict binary state validation, and model truth-table admission.
- Replaced unbounded helper assumptions with explicit state, parameter, model, probability, and resource checks.
- Used a bounded sigmoid to prevent overflow for extreme finite model parameters.
- Validated every carrier frame and rejected conflicting later-frame recipes and damaged cells.
- Retained exact source bytes, used relative receipt paths, and published successful runs from a staging directory without overwriting existing output.
- Added 20 focused tests and fresh demonstration/evidence artifacts.
- Applied GPL-3.0-only licensing to project source, documentation, tests, and generated examples.

## 0.1.0-research

- Initial learned local OR example in the shared TIFF/GIF Compute Lab research prototype.
