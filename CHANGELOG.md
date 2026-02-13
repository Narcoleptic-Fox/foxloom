# Changelog

## [0.2.1] - 2026-02-13

### Added
- Added generic `StoreManager` API for reusable multi-scope retrieval orchestration:
  - `StoreManager`
  - `StoreManagerConfig`
  - `ScopeQuery`
  - `RetrievalCandidate`
  - `MergeStats`
- Added deterministic multi-scope merge behavior:
  - scope precedence (`workspace > session > user > global`)
  - normalized-entity conflict collapse
  - stable tie-break by `memory_id`
  - explainable `selection_reason` values
- Added StoreManager unit coverage for scope leakage protection, precedence, and deterministic ordering.

## [0.2.0] - 2026-02-13

### Added
- Added deterministic decay scoring API via `scoring` module:
  - `DecayConfig`
  - `decayed_importance(...)`
- Added active context builder via `context_builder` module:
  - budgeted context assembly with strict word caps
  - deterministic ordering (scope -> score -> memory_id)
  - optional explainability tags
- Added ONNX embedder integration path behind `onnx-embedder` feature flag.
- Added standalone crate packaging metadata for crates.io/docs.rs.

### Changed
- Split `foxloom` into a standalone repository and crate (`v0.2.0`).
- Promoted `updated_at` to first-class field on `MemoryRecord` to support portable decay logic.
- Hardened foxstash adapter retrieval behavior with deterministic tie handling and bounded over-fetch.

### Reliability
- Expanded test coverage for:
  - deterministic retrieval ordering
  - budget enforcement
  - decay behavior
  - adapter rebuild atomicity
  - filtered over-fetch retrieval recovery

[0.2.1]: https://github.com/Narcoleptic-Fox/foxloom/releases/tag/v0.2.1
[0.2.0]: https://github.com/Narcoleptic-Fox/foxloom/releases/tag/v0.2.0
