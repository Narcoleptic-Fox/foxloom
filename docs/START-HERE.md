# Start Here

This guide gets you productive in under 15 minutes.

## What Foxloom Is

`foxloom` is a Rust memory library that provides:

- A canonical memory model (`MemoryRecord`, `MemoryOp`).
- Merge semantics for add/update/supersede/noop decisions.
- Decay scoring (`decayed_importance`) for recency-aware ranking.
- Deterministic active context builder under strict budget.
- A `foxstash-core` adapter for vector upsert/search/delete.

## Quick Install

```toml
[dependencies]
foxloom = "0.2"
```

Optional ONNX embedder support:

```toml
[dependencies]
foxloom = { version = "0.2", features = ["onnx-embedder"] }
```

## Minimal End-to-End Example

```rust
use foxloom::{
    build_active_context, merge_candidate, ContextBudget, ContextBuildConfig, ContextItem,
    DecayConfig, MemoryRecord, MemoryScope, MemoryType, decayed_importance,
};
use uuid::Uuid;

let mut record = MemoryRecord::new(
    Uuid::new_v4(),
    MemoryScope::Session,
    MemoryType::Episodic,
    "Primary region is us-east-1".to_string(),
);
record.importance = 0.9;

let effective = decayed_importance(
    record.importance,
    record.decay_half_life_hours,
    24.0,
    &DecayConfig::default(),
);

let candidate = MemoryRecord::new(
    Uuid::new_v4(),
    MemoryScope::Session,
    MemoryType::Episodic,
    "Primary region is us-east-1".to_string(),
);
let _op = merge_candidate(Some(&record), &candidate);

let built = build_active_context(
    &[ContextItem {
        memory_id: record.memory_id,
        text: record.text.clone(),
        memory_type: record.memory_type.clone(),
        scope: record.scope.clone(),
        similarity: 0.92,
        confidence: record.confidence,
        importance: effective,
        score: 0.98,
        source: "vector_search".to_string(),
    }],
    &ContextBudget { max_words: 220, reserve_words: 40 },
    &ContextBuildConfig { include_headers: true, include_why: false },
);

assert!(!built.prompt_prefix.is_empty());
```

## Onboarding Flow

```mermaid
flowchart LR
    A[Read START-HERE] --> B[Understand Memory Model]
    B --> C[Understand Retrieval and Context]
    C --> D[Run Tests]
    D --> E[Implement Integration]
```
