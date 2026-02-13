# foxloom

Hybrid memory layer with mem0-style semantics on top of `foxstash-core`.

## What it provides

- Canonical memory model (`MemoryRecord`, scopes, memory types, ops)
- Deterministic merge semantics for add/update/supersede/noop decisions
- Decay scoring (`decayed_importance`) with configurable half-life/floor behavior
- Active context builder with strict word budgeting and deterministic ordering
- `foxstash-core` adapter for embedding upsert/search/delete and snapshot rebuild
- Optional ONNX embedder integration behind the `onnx-embedder` feature

## Install

```toml
[dependencies]
foxloom = "0.2"
```

Optional ONNX support:

```toml
[dependencies]
foxloom = { version = "0.2", features = ["onnx-embedder"] }
```

## Local ONNX assets

ONNX model/tokenizer files are intentionally not shipped in this crate.
For local development, store them under `.models/` (ignored by git).

## License

MIT
