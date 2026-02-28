# foxstash-db Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace foxloom's manual WAL/tombstone/persistence layer with foxstash-db v0.4.0, introducing proper error types and optional hybrid search capabilities.

**Architecture:** Feature-gated integration. `foxstash-db` is an optional dependency behind a `db` feature flag. A new `FoxstashDbAdapter` wraps `foxstash_db::Collection` + `Arc<dyn TextEmbedder>`. The `FoxstashAdapter` trait gains optional `search_hybrid`/`search_text` methods with fallback defaults. `PersistentFoxstashCoreAdapter` is removed entirely. All `Result<_, String>` errors become `Result<_, foxloom::Error>`.

**Tech Stack:** Rust, foxstash-core 0.4, foxstash-db 0.4, serde_json, uuid, chrono, parking_lot

---

## Task 1: Add Error Type

**Files:**
- Create: `src/error.rs`
- Modify: `src/lib.rs:1-24` (add module + re-export)
- Modify: `Cargo.toml` (no change yet, just prep)

**Step 1: Write failing test for error Display**

Add to `src/error.rs`:

```rust
use std::fmt;

use serde_json;

/// Errors produced by foxloom operations.
#[derive(Debug)]
pub enum Error {
    /// Embedding failed (model load, inference, dimension mismatch).
    Embedding(String),

    /// foxstash-core index operation failed.
    Core(foxstash_core::RagError),

    /// foxstash-db operation failed (feature-gated).
    #[cfg(feature = "db")]
    Db(foxstash_db::DbError),

    /// Serialization or deserialization failure.
    Serde(serde_json::Error),

    /// Record not found or invalid state.
    NotFound(String),

    /// Validation failure (bad input, missing required field).
    Validation(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Embedding(msg) => write!(f, "embedding error: {msg}"),
            Error::Core(err) => write!(f, "foxstash-core error: {err}"),
            #[cfg(feature = "db")]
            Error::Db(err) => write!(f, "foxstash-db error: {err}"),
            Error::Serde(err) => write!(f, "serialization error: {err}"),
            Error::NotFound(msg) => write!(f, "not found: {msg}"),
            Error::Validation(msg) => write!(f, "validation error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Core(err) => Some(err),
            #[cfg(feature = "db")]
            Error::Db(err) => Some(err),
            Error::Serde(err) => Some(err),
            _ => None,
        }
    }
}

impl From<foxstash_core::RagError> for Error {
    fn from(err: foxstash_core::RagError) -> Self {
        Error::Core(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serde(err)
    }
}

#[cfg(feature = "db")]
impl From<foxstash_db::DbError> for Error {
    fn from(err: foxstash_db::DbError) -> Self {
        Error::Db(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_embedding_error() {
        let err = Error::Embedding("dimension mismatch".to_string());
        assert_eq!(err.to_string(), "embedding error: dimension mismatch");
    }

    #[test]
    fn display_not_found_error() {
        let err = Error::NotFound("record abc".to_string());
        assert_eq!(err.to_string(), "not found: record abc");
    }

    #[test]
    fn display_validation_error() {
        let err = Error::Validation("empty text".to_string());
        assert_eq!(err.to_string(), "validation error: empty text");
    }

    #[test]
    fn serde_error_converts() {
        let raw = serde_json::from_str::<serde_json::Value>("not json");
        let err: Error = raw.unwrap_err().into();
        assert!(err.to_string().starts_with("serialization error:"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }
}
```

**Step 2: Register the module in `src/lib.rs`**

Add `pub mod error;` and `pub use error::Error;` to the module declarations.

**Step 3: Run tests to verify**

Run: `cargo test error::tests -v`
Expected: All 5 tests PASS

**Step 4: Commit**

```
git add src/error.rs src/lib.rs
git commit -m "feat: add foxloom::Error type with variant-per-source design"
```

---

## Task 2: Bump foxstash-core to 0.4 and Add foxstash-db Optional Dep

**Files:**
- Modify: `Cargo.toml`

**Step 1: Update dependencies**

In `Cargo.toml`, change:
- `foxstash-core = "0.3.1"` to `foxstash-core = "0.4"`
- Add `foxstash-db = { version = "0.4", optional = true }`
- Add feature: `db = ["dep:foxstash-db"]`

The `[features]` section becomes:
```toml
[features]
default = []
onnx-embedder = ["foxstash-core/onnx"]
db = ["dep:foxstash-db"]
```

The `[dependencies]` section becomes:
```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
uuid = { version = "1", features = ["v4", "v5", "serde"] }
chrono = { version = "0.4", features = ["serde", "clock"] }
foxstash-core = "0.4"
foxstash-db = { version = "0.4", optional = true }
parking_lot = "0.12"
```

**Step 2: Run `cargo check` (no features)**

Run: `cargo check`
Expected: PASS (may need minor foxstash-core 0.4 API adjustments if any broke)

**Step 3: Run `cargo check --features db`**

Run: `cargo check --features db`
Expected: PASS (foxstash-db dep resolves)

**Step 4: Run full test suite**

Run: `cargo test`
Expected: All 26 tests PASS (25 unit + 1 doc-test)

NOTE: If foxstash-core 0.4 has breaking API changes vs 0.3.1, fix them here. Known areas to check:
- `HNSWIndex::with_defaults` signature
- `IncrementalStorage::new` / `IncrementalConfig` shape
- `Document` / `SearchResult` field names
- `RagError` variants

**Step 5: Commit**

```
git add Cargo.toml
git commit -m "build: bump foxstash-core to 0.4, add optional foxstash-db dep"
```

---

## Task 3: Migrate FoxstashAdapter Trait to Error Type

**Files:**
- Modify: `src/adapter.rs` (trait signatures)
- Modify: `src/foxstash_adapter.rs` (FoxstashCoreAdapter impl, all `.map_err(|e| format!(...))` → proper Error variants)

**Step 1: Update trait signatures in `src/adapter.rs`**

Change all `Result<_, String>` to `Result<_, crate::Error>`. The full file becomes:

```rust
use serde_json::Value;

use crate::error::Error;
use crate::MemoryRecord;

/// Storage adapter trait for vector-backed memory persistence.
///
/// Implementations handle embedding storage, deletion, and similarity retrieval.
/// The trait is intentionally minimal — 3 required methods — to keep backend
/// implementations lightweight.
pub trait FoxstashAdapter: Send + Sync {
    /// Insert or replace an embedding keyed by `key`.
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), Error>;

    /// Soft-delete an embedding by `key`.
    fn delete_embedding(&self, key: &str) -> Result<(), Error>;

    /// Return the `top_k` most similar records to `query`, optionally filtered by metadata.
    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, Error>;

    /// Batch insert/replace embeddings. Default loops over `upsert_embedding`.
    fn batch_upsert_embeddings(&self, items: &[(String, String, Value)]) -> Result<(), Error> {
        for (key, text, metadata) in items {
            self.upsert_embedding(key, text, metadata.clone())?;
        }
        Ok(())
    }

    /// Hybrid vector + keyword search. Default falls back to `similarity_search`.
    fn search_hybrid(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, Error> {
        self.similarity_search(query, top_k, metadata_filter)
    }

    /// Keyword-only (BM25) search. Default falls back to `similarity_search`.
    fn search_text(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, Error> {
        self.similarity_search(query, top_k, metadata_filter)
    }

    /// Whether this adapter supports hybrid search natively.
    fn supports_hybrid(&self) -> bool {
        false
    }

    /// Whether this adapter supports text-only (BM25) search natively.
    fn supports_text_search(&self) -> bool {
        false
    }
}
```

**Step 2: Update FoxstashCoreAdapter impl in `src/foxstash_adapter.rs`**

Change all error mappings in `FoxstashCoreAdapter`:
- `self.embedder.embed(text)?` returns `Result<_, String>` — wrap with `Error::Embedding`:
  `self.embedder.embed(text).map_err(Error::Embedding)?`
- `.map_err(|e| format!("foxstash add failed: {e}"))` → `.map_err(Error::Core)?` (since `RagError` has `From`)
- `.map_err(|e| format!("foxstash search failed: {e}"))` → `.map_err(Error::Core)?`
- `rebuild_from_documents` error: `.map_err(Error::Core)?`

Also update `TextEmbedder::embed` return type consideration: The `TextEmbedder` trait still returns `Result<_, String>` — this is fine, the adapter maps it to `Error::Embedding` at the call site. Changing `TextEmbedder` would be a larger breaking change for a separate PR.

**Step 3: Run tests**

Run: `cargo test`
Expected: All tests PASS. Existing tests use `.expect()` so the error type change is transparent.

**Step 4: Commit**

```
git add src/adapter.rs src/foxstash_adapter.rs
git commit -m "feat!: migrate FoxstashAdapter trait from String to foxloom::Error"
```

---

## Task 4: Extract Metadata Conversion to Shared Module

**Files:**
- Create: `src/metadata.rs`
- Modify: `src/foxstash_adapter.rs` (remove duplicated code, import from metadata)
- Modify: `src/lib.rs` (add `mod metadata;` — not `pub`)

**Step 1: Create `src/metadata.rs`**

Extract these functions from `src/foxstash_adapter.rs` into `src/metadata.rs`:
- `FALLBACK_UUID_NAMESPACE` constant
- `RESERVED_METADATA_KEYS` constant
- `normalize_document_metadata(key, metadata) -> Value`
- `memory_record_from_parts(doc_id, content, metadata, default_confidence, embedding_ref) -> MemoryRecord`
- `result_to_memory_record(result) -> MemoryRecord` (rename to `search_result_to_record`)
- `document_to_memory_record(document) -> MemoryRecord` (rename to `document_to_record`)
- `metadata_from_record(record) -> Value` (rename to `record_to_metadata`)
- `extract_json_fields(meta_obj) -> Value`
- `metadata_matches(metadata, filter) -> bool`
- All parse helpers: `value_as_uuid`, `value_as_str`, `value_as_f32`, `value_as_u32`, `parse_scope`, `parse_memory_type`, `parse_status`

All items should be `pub(crate)`.

**Step 2: Update `src/foxstash_adapter.rs`**

Replace all moved functions with imports:
```rust
use crate::metadata::{
    normalize_document_metadata, search_result_to_record, document_to_record,
    record_to_metadata, metadata_matches,
};
```

Remove the moved function bodies and constants. Keep only:
- `TextEmbedder` trait + impls (`DeterministicEmbedder`, `OnnxEmbedder`)
- `FoxstashCoreAdapter` struct + impl
- `normalize_vector` and `fnv1a64` (embedding-specific, stay here)
- `OVERFETCH_CAP_MULTIPLIER` constant

**Step 3: Add `mod metadata;` to `src/lib.rs`**

Add `mod metadata;` (not `pub`) alongside the other module declarations.

**Step 4: Run tests**

Run: `cargo test`
Expected: All tests PASS — pure refactor, no behavior change.

**Step 5: Commit**

```
git add src/metadata.rs src/foxstash_adapter.rs src/lib.rs
git commit -m "refactor: extract metadata conversion to shared module"
```

---

## Task 5: Remove PersistentFoxstashCoreAdapter

**Files:**
- Modify: `src/foxstash_adapter.rs` (delete ~250 lines: lines 650-997)
- Modify: `src/lib.rs` (remove re-exports of `PersistentConfig`, `PersistentFoxstashCoreAdapter`)

**Step 1: Delete PersistentFoxstashCoreAdapter and related code**

Remove from `src/foxstash_adapter.rs`:
- `PersistentConfig` struct and `Default` impl (lines 650-663)
- `PersistedDocument` struct and impls (lines 665-705)
- `encode_wal_content` function (lines 707-714)
- `decode_wal_document` function (lines 716-732)
- `PersistentFoxstashCoreAdapter` struct and all impls (lines 733-997)
- Test functions `persistent_adapter_recovers_from_wal_reopen` and `persistent_adapter_exposes_checkpoint_stats` (lines 1382-1449)

Also remove now-unused imports at the top of `foxstash_adapter.rs`:
- `std::path::Path`
- `std::collections::HashMap`
- `foxstash_core::storage::incremental::*` (IncrementalConfig, IncrementalStorage, IndexMetadata, RecoveryHelper, StorageStats, WalOperation)
- `serde::{Deserialize, Serialize}` (if no longer needed — check `DeterministicEmbedder`)

**Step 2: Update `src/lib.rs` re-exports**

Remove from `pub use foxstash_adapter`:
- `PersistentConfig`
- `PersistentFoxstashCoreAdapter`

The export line becomes:
```rust
pub use foxstash_adapter::{
    DeterministicEmbedder, FoxstashCoreAdapter, TextEmbedder,
};
```

**Step 3: Run tests**

Run: `cargo test`
Expected: 23 tests PASS (lost 2 persistent adapter tests). Doc-test still passes.

**Step 4: Commit**

```
git add src/foxstash_adapter.rs src/lib.rs
git commit -m "feat!: remove PersistentFoxstashCoreAdapter (superseded by foxstash-db)"
```

---

## Task 6: Add json_to_filter Conversion (Feature-Gated)

**Files:**
- Modify: `src/metadata.rs` (add `json_to_filter` function, behind `#[cfg(feature = "db")]`)

**Step 1: Write failing test**

Add to `src/metadata.rs` tests (behind `#[cfg(feature = "db")]`):

```rust
#[cfg(feature = "db")]
mod db_tests {
    use super::*;
    use foxstash_db::Filter;

    #[test]
    fn json_to_filter_flat_object() {
        let json = serde_json::json!({"scope": "session", "status": "active"});
        let filter = json_to_filter(&json).expect("should produce filter");
        // We can't easily inspect Filter internals, but we can verify it was created
        // The filter should be an And of two Eq filters
        let _ = filter; // type-checks that it returns Filter
    }

    #[test]
    fn json_to_filter_none_for_null() {
        let json = serde_json::Value::Null;
        assert!(json_to_filter(&json).is_none());
    }

    #[test]
    fn json_to_filter_none_for_empty_object() {
        let json = serde_json::json!({});
        assert!(json_to_filter(&json).is_none());
    }

    #[test]
    fn json_to_filter_nested_uses_dot_notation() {
        let json = serde_json::json!({"json_fields": {"entity": "deploy_window"}});
        let filter = json_to_filter(&json).expect("should produce filter");
        let _ = filter;
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --features db metadata::db_tests -v`
Expected: FAIL — `json_to_filter` not defined

**Step 3: Implement `json_to_filter`**

Add to `src/metadata.rs`:

```rust
/// Convert a recursive-equality JSON filter into a foxstash-db `Filter`.
///
/// The JSON format mirrors the existing `metadata_matches` semantics:
/// `{"scope": "session", "status": "active"}` becomes
/// `Filter::And([Filter::Eq("scope", "session"), Filter::Eq("status", "active")])`.
///
/// Nested objects use dot-notation:
/// `{"json_fields": {"entity": "x"}}` becomes `Filter::Eq("json_fields.entity", "x")`.
///
/// Returns `None` for null or empty objects.
#[cfg(feature = "db")]
pub(crate) fn json_to_filter(value: &serde_json::Value) -> Option<foxstash_db::Filter> {
    fn collect_filters(prefix: &str, value: &serde_json::Value, out: &mut Vec<foxstash_db::Filter>) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    let field = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    collect_filters(&field, v, out);
                }
            }
            _ => {
                out.push(foxstash_db::Filter::eq(&prefix, value.clone()));
            }
        }
    }

    if value.is_null() {
        return None;
    }
    let mut filters = Vec::new();
    collect_filters("", value, &mut filters);
    if filters.is_empty() {
        return None;
    }
    if filters.len() == 1 {
        return Some(filters.remove(0));
    }
    Some(foxstash_db::Filter::and(filters))
}
```

**Step 4: Run tests**

Run: `cargo test --features db metadata -v`
Expected: All metadata tests PASS (both default and db-gated)

**Step 5: Commit**

```
git add src/metadata.rs
git commit -m "feat: add json_to_filter conversion for foxstash-db integration"
```

---

## Task 7: Implement FoxstashDbAdapter

**Files:**
- Create: `src/foxstash_db_adapter.rs`
- Modify: `src/lib.rs` (add feature-gated module + re-exports)

**Step 1: Write the adapter with tests**

Create `src/foxstash_db_adapter.rs`:

```rust
//! foxstash-db backed adapter for persistent vector storage with hybrid search.
//!
//! Requires the `db` feature flag.

use std::sync::Arc;

use foxstash_db::Collection;
use serde_json::Value;

use crate::error::Error;
use crate::foxstash_adapter::TextEmbedder;
use crate::metadata::{
    json_to_filter, normalize_document_metadata, search_result_to_record, record_to_metadata,
};
use crate::{FoxstashAdapter, MemoryRecord};

/// Adapter backed by a [`foxstash_db::Collection`] for WAL-persistent vector storage
/// with hybrid search and rich metadata filtering.
///
/// Constructed with an injected `Collection` and `TextEmbedder`. The adapter owns
/// the embedding step; the `Collection` owns persistence, indexing, and search.
pub struct FoxstashDbAdapter {
    collection: Arc<Collection>,
    embedder: Arc<dyn TextEmbedder>,
}

impl FoxstashDbAdapter {
    /// Create a new adapter with an existing collection and embedder.
    pub fn new(collection: Arc<Collection>, embedder: Arc<dyn TextEmbedder>) -> Self {
        Self {
            collection,
            embedder,
        }
    }

    /// Convenience constructor using a `DeterministicEmbedder` for testing.
    pub fn with_deterministic_embedder(collection: Arc<Collection>, dim: usize) -> Self {
        use crate::foxstash_adapter::DeterministicEmbedder;
        Self::new(collection, Arc::new(DeterministicEmbedder::new(dim)))
    }

    /// Direct access to the underlying collection.
    pub fn collection(&self) -> &Arc<Collection> {
        &self.collection
    }

    /// Flush WAL to disk.
    pub fn flush(&self) -> Result<(), Error> {
        self.collection.flush().map_err(Error::Db)
    }

    /// Compact the collection (rebuild index from live documents only).
    pub fn compact(&self) -> Result<(), Error> {
        self.collection.compact().map_err(Error::Db)
    }

    /// Number of documents in the collection.
    pub fn len(&self) -> usize {
        self.collection.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.collection.is_empty()
    }

    /// Search with a typed `foxstash_db::Filter` instead of JSON metadata.
    pub fn search_with_filter(
        &self,
        query: &str,
        top_k: usize,
        filter: &foxstash_db::Filter,
    ) -> Result<Vec<MemoryRecord>, Error> {
        let embedding = self.embedder.embed(query).map_err(Error::Embedding)?;
        let results = self
            .collection
            .search(&embedding, top_k, Some(filter))
            .map_err(Error::Db)?;
        Ok(results.into_iter().map(search_result_to_record).collect())
    }
}

impl FoxstashAdapter for FoxstashDbAdapter {
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), Error> {
        let embedding = self.embedder.embed(text).map_err(Error::Embedding)?;
        let metadata = normalize_document_metadata(key, metadata);
        self.collection
            .upsert(key.to_string(), text.to_string(), embedding, Some(metadata))
            .map_err(Error::Db)
    }

    fn delete_embedding(&self, key: &str) -> Result<(), Error> {
        // foxstash-db returns false if not found — not an error for us
        self.collection.delete(key).map_err(Error::Db)?;
        Ok(())
    }

    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, Error> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let embedding = self.embedder.embed(query).map_err(Error::Embedding)?;
        let filter = metadata_filter.as_ref().and_then(json_to_filter);
        let results = self
            .collection
            .search(&embedding, top_k, filter.as_ref())
            .map_err(Error::Db)?;
        Ok(results.into_iter().map(search_result_to_record).collect())
    }

    fn search_hybrid(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, Error> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let embedding = self.embedder.embed(query).map_err(Error::Embedding)?;
        let filter = metadata_filter.as_ref().and_then(json_to_filter);
        let results = self
            .collection
            .search_hybrid(&embedding, query, top_k, filter.as_ref(), None)
            .map_err(Error::Db)?;
        Ok(results.into_iter().map(search_result_to_record).collect())
    }

    fn search_text(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, Error> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let filter = metadata_filter.as_ref().and_then(json_to_filter);
        let results = self
            .collection
            .search_text(query, top_k, filter.as_ref())
            .map_err(Error::Db)?;
        Ok(results.into_iter().map(search_result_to_record).collect())
    }

    fn supports_hybrid(&self) -> bool {
        true
    }

    fn supports_text_search(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foxstash_db::{DbConfig, Filter, VectorStore};
    use tempfile::TempDir;
    use uuid::Uuid;

    fn test_adapter(dir: &std::path::Path, dim: usize) -> FoxstashDbAdapter {
        let store = VectorStore::open(
            dir,
            DbConfig::default().with_embedding_dim(dim),
        )
        .expect("open store");
        let collection = store
            .get_or_create_collection("test")
            .expect("create collection");
        FoxstashDbAdapter::with_deterministic_embedder(collection, dim)
    }

    #[test]
    fn upsert_and_search_round_trip() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);
        let key = Uuid::new_v4().to_string();

        adapter
            .upsert_embedding(
                &key,
                "deploy window is saturday",
                serde_json::json!({"scope": "session", "status": "active"}),
            )
            .expect("upsert");

        let results = adapter
            .similarity_search("deploy window", 5, None)
            .expect("search");
        assert!(!results.is_empty());
        assert!(results[0].text.contains("deploy window"));
    }

    #[test]
    fn delete_hides_from_search() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);
        let key = Uuid::new_v4().to_string();

        adapter
            .upsert_embedding(
                &key,
                "secret fact",
                serde_json::json!({"scope": "session"}),
            )
            .expect("upsert");

        adapter.delete_embedding(&key).expect("delete");

        let results = adapter
            .similarity_search("secret fact", 5, None)
            .expect("search");
        assert!(results.is_empty(), "deleted record should be hidden");
    }

    #[test]
    fn metadata_filter_works() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);

        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "workspace policy alpha",
                serde_json::json!({"scope": "workspace", "status": "active"}),
            )
            .expect("upsert");
        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "session note alpha",
                serde_json::json!({"scope": "session", "status": "active"}),
            )
            .expect("upsert");

        let results = adapter
            .similarity_search(
                "alpha",
                5,
                Some(serde_json::json!({"scope": "workspace"})),
            )
            .expect("search");
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("workspace"));
    }

    #[test]
    fn hybrid_search_returns_results() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);

        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "gateway service running on port 8080",
                serde_json::json!({"scope": "session"}),
            )
            .expect("upsert");

        let results = adapter
            .search_hybrid("gateway port", 5, None)
            .expect("hybrid search");
        assert!(!results.is_empty());
    }

    #[test]
    fn text_search_returns_results() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);

        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "kubernetes pod scaling policy",
                serde_json::json!({"scope": "session"}),
            )
            .expect("upsert");

        let results = adapter
            .search_text("kubernetes scaling", 5, None)
            .expect("text search");
        assert!(!results.is_empty());
    }

    #[test]
    fn supports_flags_are_true() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);
        assert!(adapter.supports_hybrid());
        assert!(adapter.supports_text_search());
    }

    #[test]
    fn typed_filter_search() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);

        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "infra alert critical",
                serde_json::json!({"scope": "workspace", "status": "active"}),
            )
            .expect("upsert");

        let filter = Filter::eq("scope", "workspace");
        let results = adapter
            .search_with_filter("infra alert", 5, &filter)
            .expect("filtered search");
        assert!(!results.is_empty());
    }

    #[test]
    fn flush_and_compact_succeed() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);

        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "compaction test",
                serde_json::json!({"scope": "session"}),
            )
            .expect("upsert");

        adapter.flush().expect("flush");
        adapter.compact().expect("compact");
        assert_eq!(adapter.len(), 1);
    }

    #[test]
    fn batch_upsert_works() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);

        let items: Vec<(String, String, Value)> = (0..5)
            .map(|i| {
                (
                    Uuid::new_v4().to_string(),
                    format!("batch item {i}"),
                    serde_json::json!({"scope": "session", "status": "active"}),
                )
            })
            .collect();

        adapter.batch_upsert_embeddings(&items).expect("batch upsert");
        assert_eq!(adapter.len(), 5);
    }

    #[test]
    fn empty_top_k_returns_empty() {
        let dir = TempDir::new().expect("tempdir");
        let adapter = test_adapter(dir.path(), 64);
        let results = adapter.similarity_search("anything", 0, None).expect("search");
        assert!(results.is_empty());
        let results = adapter.search_hybrid("anything", 0, None).expect("hybrid");
        assert!(results.is_empty());
        let results = adapter.search_text("anything", 0, None).expect("text");
        assert!(results.is_empty());
    }
}
```

**Step 2: Register module and re-exports in `src/lib.rs`**

Add feature-gated module and re-exports:

```rust
#[cfg(feature = "db")]
pub mod foxstash_db_adapter;

#[cfg(feature = "db")]
pub use foxstash_db_adapter::FoxstashDbAdapter;

#[cfg(feature = "db")]
pub use foxstash_db::{DbConfig, Filter, HybridConfig, MergeStrategy};
```

**Step 3: Run tests**

Run: `cargo test --features db -v`
Expected: All tests PASS (existing + new db adapter tests)

**Step 4: Commit**

```
git add src/foxstash_db_adapter.rs src/lib.rs
git commit -m "feat: add FoxstashDbAdapter with hybrid search and rich filtering"
```

---

## Task 8: Update lib.rs Doc-Test and Public API

**Files:**
- Modify: `src/lib.rs` (update doc-test example, clean up exports)

**Step 1: Update the doc-test on `merge_candidate`**

The existing doc-test uses the old API. Ensure the example still compiles with the new `Error` type. The doc-test doesn't call adapter methods so it should be fine — just verify.

**Step 2: Run doc-tests**

Run: `cargo test --doc`
Expected: 1 doc-test PASS

**Step 3: Run full test suite with all feature combos**

Run these sequentially:
```bash
cargo test                           # no features
cargo test --features db             # db only
cargo test --features onnx-embedder  # onnx only (may skip if no model files)
cargo test --features db,onnx-embedder  # both (may skip if no model files)
```

Expected: All pass (onnx tests may be skipped if model files absent)

**Step 4: Commit (if any changes needed)**

```
git add src/lib.rs
git commit -m "docs: update public API surface for v0.3.0"
```

---

## Task 9: Bump Version to 0.3.0

**Files:**
- Modify: `Cargo.toml` (version field)

**Step 1: Update version**

Change `version = "0.2.1"` to `version = "0.3.0"` in `Cargo.toml`.

**Step 2: Final verification**

Run: `cargo test --features db`
Run: `cargo doc --features db --no-deps`

Expected: All pass, docs generate cleanly

**Step 3: Commit**

```
git add Cargo.toml
git commit -m "chore: bump foxloom to v0.3.0"
```

---

## Summary of Breaking Changes (v0.2.1 → v0.3.0)

1. **`FoxstashAdapter` trait**: All methods return `Result<_, foxloom::Error>` instead of `Result<_, String>`
2. **`PersistentFoxstashCoreAdapter` removed**: Use `FoxstashDbAdapter` (behind `db` feature) instead
3. **`PersistentConfig` removed**: Use `foxstash_db::DbConfig` instead
4. **foxstash-core bumped**: 0.3.1 → 0.4 (type compatibility)
5. **New trait methods** (non-breaking due to defaults): `search_hybrid`, `search_text`, `supports_hybrid`, `supports_text_search`

## Migration Guide (for consumers)

```rust
// Before (v0.2.x)
use foxloom::{PersistentFoxstashCoreAdapter, PersistentConfig};
let adapter = PersistentFoxstashCoreAdapter::new(384, "./data", PersistentConfig::default())?;

// After (v0.3.0)
use foxloom::{FoxstashDbAdapter, DbConfig};
use foxstash_db::VectorStore;
let store = VectorStore::open("./data", DbConfig::default().with_embedding_dim(384))?;
let collection = store.get_or_create_collection("memories")?;
let adapter = FoxstashDbAdapter::with_deterministic_embedder(collection, 384);
// Or with a real embedder:
// let adapter = FoxstashDbAdapter::new(collection, Arc::new(my_embedder));
```
