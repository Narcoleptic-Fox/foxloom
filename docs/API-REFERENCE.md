# API Reference

This page summarizes the public API from `src/lib.rs`.

## Re-exports

- `FoxstashAdapter`
- `DeterministicEmbedder`, `FoxstashCoreAdapter`, `TextEmbedder`
- `PersistentFoxstashCoreAdapter`, `PersistentConfig`
- `OnnxEmbedder` (feature: `onnx-embedder`)
- `MemoryOp`, `MemoryRecord`, `MemoryScope`, `MemoryStatus`, `MemoryType`
- `DecayConfig`, `decayed_importance`
- `ContextBudget`, `ContextBuildConfig`, `ContextItem`, `BuiltContext`
- `BudgetEstimator`, `WordBudgetEstimator`
- `build_active_context`, `build_active_context_with_estimator`
- `MergeStats`, `RetrievalCandidate`, `ScopeQuery`, `StoreManager`, `StoreManagerConfig`

## Merge Helper

- `merge_candidate(existing: Option<&MemoryRecord>, candidate: &MemoryRecord) -> MemoryOp`

## Adapter Contract

```rust
pub trait FoxstashAdapter: Send + Sync {
    fn upsert_embedding(&self, key: &str, text: &str, metadata: serde_json::Value) -> Result<(), String>;
    fn delete_embedding(&self, key: &str) -> Result<(), String>;
    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<serde_json::Value>,
    ) -> Result<Vec<MemoryRecord>, String>;

    fn batch_upsert_embeddings(&self, items: &[(String, String, Value)]) -> Result<(), String>;
}
```

## Persistent Adapter

`PersistentFoxstashCoreAdapter` provides WAL-backed persistence and recovery.

```rust
impl PersistentFoxstashCoreAdapter {
    pub fn new(dim: usize, path: impl AsRef<Path>, config: PersistentConfig) -> Result<Self, String>;
    pub fn sync(&self) -> Result<(), String>;
    pub fn force_checkpoint(&self) -> Result<(), String>;
    pub fn storage_stats(&self) -> StorageStats;
}
```

## Stability Notes

- `model`, `scoring`, and `context_builder` are intended for direct integration.
- `foxstash_adapter` is the reference implementation over `foxstash-core`.
- `onnx-embedder` is optional and should be feature-gated in downstream crates.
