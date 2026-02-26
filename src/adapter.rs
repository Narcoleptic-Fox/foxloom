use serde_json::Value;

use crate::MemoryRecord;

/// Storage adapter trait for vector-backed memory persistence.
///
/// Implementations handle embedding storage, deletion, and similarity retrieval.
/// The trait is intentionally minimal — 3 methods — to keep backend implementations
/// lightweight. See [`crate::FoxstashCoreAdapter`] for the default HNSW-backed impl.
pub trait FoxstashAdapter: Send + Sync {
    /// Insert or replace an embedding keyed by `key`.
    ///
    /// If a record with the same `key` already exists, it is overwritten.
    /// The `metadata` value is stored alongside the embedding for filtering during search.
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), String>;

    /// Soft-delete an embedding by `key`.
    ///
    /// The record is hidden from subsequent searches but may remain in the index
    /// until compaction (e.g. via `rebuild_from_records`). Re-upserting the same
    /// key clears the deletion marker.
    fn delete_embedding(&self, key: &str) -> Result<(), String>;

    /// Return the `top_k` most similar records to `query`, optionally filtered by metadata.
    ///
    /// `metadata_filter` performs recursive equality matching: every key/value in the
    /// filter must be present and equal in the record's metadata. Nested objects are
    /// matched recursively. Pass `None` to skip filtering.
    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, String>;

    fn batch_upsert_embeddings(&self, items: &[(String, String, Value)]) -> Result<(), String> {
        for (key, text, metadata) in items {
            self.upsert_embedding(key, text, metadata.clone())?;
        }
        Ok(())
    }
}
