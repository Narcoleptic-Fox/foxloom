use std::sync::Arc;

use foxstash_core::index::HNSWIndex;
use foxstash_core::Document;
use foxstash_core::SearchResult;
use parking_lot::Mutex;
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{FoxstashAdapter, MemoryRecord, MemoryScope, MemoryStatus, MemoryType};

const FALLBACK_UUID_NAMESPACE: Uuid = Uuid::from_bytes([
    0x84, 0x34, 0x5a, 0x26, 0x3b, 0xa7, 0x4f, 0xb0, 0x88, 0x26, 0x95, 0x5f, 0x6d, 0x65, 0x60, 0xcf,
]);

const RESERVED_METADATA_KEYS: [&str; 12] = [
    "memory_id",
    "workspace_id",
    "user_id",
    "session_id",
    "scope",
    "memory_type",
    "status",
    "confidence",
    "importance",
    "decay_half_life_hours",
    "source_run_id",
    "json_fields",
];

pub trait TextEmbedder: Send + Sync {
    fn dimension(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
}

#[derive(Debug, Clone)]
pub struct DeterministicEmbedder {
    embedding_dim: usize,
}

impl DeterministicEmbedder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl TextEmbedder for DeterministicEmbedder {
    fn dimension(&self) -> usize {
        self.embedding_dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        if self.embedding_dim == 0 {
            return Err("embedding_dim must be > 0".to_string());
        }

        let mut out = vec![0.0f32; self.embedding_dim];
        if text.trim().is_empty() {
            return Ok(out);
        }

        for (position, token) in text.split_whitespace().enumerate() {
            let hash = fnv1a64(token.as_bytes());
            let idx = (hash as usize) % self.embedding_dim;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            let weight = 1.0 + ((position % 7) as f32 * 0.03);
            out[idx] += sign * weight;
        }

        for (i, window) in text.as_bytes().windows(2).enumerate() {
            let hash = fnv1a64(window);
            let idx = (hash as usize) % self.embedding_dim;
            let sign = if (hash >> 1) & 1 == 0 { 1.0 } else { -1.0 };
            let weight = 0.2 + ((i % 5) as f32 * 0.01);
            out[idx] += sign * weight;
        }

        normalize_vector(&mut out);
        Ok(out)
    }
}

#[cfg(feature = "onnx-embedder")]
pub struct OnnxEmbedder {
    inner: Mutex<foxstash_core::embedding::OnnxEmbedder>,
}

#[cfg(feature = "onnx-embedder")]
impl OnnxEmbedder {
    pub fn from_files(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        ort_dylib_path: Option<std::path::PathBuf>,
    ) -> Result<Self, String> {
        if let Some(path) = ort_dylib_path {
            foxstash_core::embedding::OnnxEmbedder::init_from(path)
                .map_err(|e| format!("failed to initialize ONNX runtime: {e}"))?;
        }

        let inner = foxstash_core::embedding::OnnxEmbedder::new(model_path, tokenizer_path)
            .map_err(|e| format!("failed to create ONNX embedder: {e}"))?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }
}

#[cfg(feature = "onnx-embedder")]
impl TextEmbedder for OnnxEmbedder {
    fn dimension(&self) -> usize {
        self.inner.lock().embedding_dim()
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut vec = self
            .inner
            .lock()
            .embed(text)
            .map_err(|e| format!("onnx embedder inference failed: {e}"))?;
        normalize_vector(&mut vec);
        Ok(vec)
    }
}

#[derive(Clone)]
pub struct FoxstashCoreAdapter {
    index: Arc<Mutex<HNSWIndex>>,
    embedder: Arc<dyn TextEmbedder>,
}

impl FoxstashCoreAdapter {
    pub fn new(embedding_dim: usize) -> Self {
        Self::with_embedder(Arc::new(DeterministicEmbedder::new(embedding_dim)))
    }

    pub fn with_embedder(embedder: Arc<dyn TextEmbedder>) -> Self {
        let embedding_dim = embedder.dimension();
        Self {
            index: Arc::new(Mutex::new(HNSWIndex::with_defaults(embedding_dim))),
            embedder,
        }
    }

    pub fn len(&self) -> usize {
        self.index.lock().len()
    }

    pub fn snapshot_documents(&self) -> Vec<Document> {
        self.index.lock().get_all_documents()
    }

    pub fn rebuild_from_documents(&self, documents: &[Document]) -> Result<(), String> {
        let mut index = self.index.lock();
        index.clear();
        for doc in documents {
            index
                .add(doc.clone())
                .map_err(|e| format!("foxstash restore failed: {e}"))?;
        }
        Ok(())
    }

    pub fn snapshot_records(&self) -> Vec<MemoryRecord> {
        self.snapshot_documents()
            .into_iter()
            .map(document_to_memory_record)
            .collect()
    }

    pub fn rebuild_from_records(&self, records: &[MemoryRecord]) -> Result<(), String> {
        {
            self.index.lock().clear();
        }
        for record in records {
            self.upsert_embedding(
                &record.memory_id.to_string(),
                &record.text,
                metadata_from_record(record),
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "onnx-embedder")]
    pub fn try_from_onnx_files(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
        ort_dylib_path: Option<std::path::PathBuf>,
    ) -> Result<Self, String> {
        let embedder = OnnxEmbedder::from_files(model_path, tokenizer_path, ort_dylib_path)?;
        Ok(Self::with_embedder(Arc::new(embedder)))
    }

    #[cfg(not(feature = "onnx-embedder"))]
    pub fn try_from_onnx_files(
        _model_path: impl AsRef<std::path::Path>,
        _tokenizer_path: impl AsRef<std::path::Path>,
        _ort_dylib_path: Option<std::path::PathBuf>,
    ) -> Result<Self, String> {
        Err("foxloom built without `onnx-embedder` feature".to_string())
    }
}

impl FoxstashAdapter for FoxstashCoreAdapter {
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), String> {
        let embedding = self.embedder.embed(text)?;
        let metadata = normalize_document_metadata(key, metadata);
        let doc = Document {
            id: key.to_string(),
            content: text.to_string(),
            embedding,
            metadata: Some(metadata),
        };

        self.index
            .lock()
            .add(doc)
            .map_err(|e| format!("foxstash add failed: {e}"))
    }

    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, String> {
        let query_embedding = self.embedder.embed(query)?;
        let results = self
            .index
            .lock()
            .search(&query_embedding, top_k)
            .map_err(|e| format!("foxstash search failed: {e}"))?;

        let mut out = Vec::new();
        for result in results {
            let metadata_value = result
                .metadata
                .clone()
                .unwrap_or_else(|| Value::Object(Map::new()));

            if let Some(filter) = &metadata_filter {
                if !metadata_matches(&metadata_value, filter) {
                    continue;
                }
            }

            out.push(result_to_memory_record(result));
        }

        Ok(out)
    }
}

fn normalize_vector(vec: &mut [f32]) {
    let norm = vec
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for v in vec {
            if v.is_finite() {
                *v /= norm;
            } else {
                *v = 0.0;
            }
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn normalize_document_metadata(key: &str, metadata: Value) -> Value {
    let mut map = match metadata {
        Value::Object(map) => map,
        other => {
            let mut map = Map::new();
            map.insert("json_fields".to_string(), other);
            map
        }
    };

    map.entry("memory_id".to_string())
        .or_insert_with(|| Value::String(key.to_string()));
    Value::Object(map)
}

fn result_to_memory_record(result: SearchResult) -> MemoryRecord {
    memory_record_from_parts(
        &result.id,
        &result.content,
        result.metadata.as_ref(),
        result.score,
        Some(result.id.as_str()),
    )
}

fn document_to_memory_record(document: Document) -> MemoryRecord {
    memory_record_from_parts(
        &document.id,
        &document.content,
        document.metadata.as_ref(),
        0.7,
        Some(document.id.as_str()),
    )
}

fn memory_record_from_parts(
    doc_id: &str,
    content: &str,
    metadata: Option<&Value>,
    default_confidence: f32,
    embedding_ref: Option<&str>,
) -> MemoryRecord {
    let metadata = metadata
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let meta_obj = metadata.as_object();
    let status_raw = meta_obj
        .and_then(|m| m.get("status"))
        .and_then(value_as_str);

    let memory_id = meta_obj
        .and_then(|m| m.get("memory_id"))
        .and_then(value_as_uuid)
        .or_else(|| Uuid::parse_str(doc_id).ok())
        .unwrap_or_else(|| Uuid::new_v5(&FALLBACK_UUID_NAMESPACE, doc_id.as_bytes()));

    let scope = meta_obj
        .and_then(|m| m.get("scope"))
        .and_then(value_as_str)
        .and_then(parse_scope)
        .unwrap_or(MemoryScope::Session);
    let memory_type = meta_obj
        .and_then(|m| m.get("memory_type"))
        .and_then(value_as_str)
        .and_then(parse_memory_type)
        .unwrap_or(MemoryType::Episodic);
    let status = match status_raw {
        Some(raw) => parse_status(raw).unwrap_or(MemoryStatus::Quarantined),
        None => MemoryStatus::Active,
    };
    let confidence = meta_obj
        .and_then(|m| m.get("confidence"))
        .and_then(value_as_f32)
        .unwrap_or(default_confidence)
        .clamp(0.0, 1.0);
    let importance = meta_obj
        .and_then(|m| m.get("importance"))
        .and_then(value_as_f32)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let decay_half_life_hours = meta_obj
        .and_then(|m| m.get("decay_half_life_hours"))
        .and_then(value_as_u32);
    let source_run_id = meta_obj
        .and_then(|m| m.get("source_run_id"))
        .and_then(value_as_uuid);
    let json_fields = extract_json_fields(meta_obj);

    MemoryRecord {
        memory_id,
        workspace_id: meta_obj
            .and_then(|m| m.get("workspace_id"))
            .and_then(value_as_str)
            .map(str::to_string),
        user_id: meta_obj
            .and_then(|m| m.get("user_id"))
            .and_then(value_as_str)
            .map(str::to_string),
        session_id: meta_obj
            .and_then(|m| m.get("session_id"))
            .and_then(value_as_str)
            .map(str::to_string),
        scope,
        memory_type,
        text: content.to_string(),
        json_fields,
        embedding_ref: embedding_ref.map(str::to_string),
        confidence,
        importance,
        decay_half_life_hours,
        status,
        source_run_id,
    }
}

fn metadata_from_record(record: &MemoryRecord) -> Value {
    Value::Object(Map::from_iter([
        (
            "memory_id".to_string(),
            Value::String(record.memory_id.to_string()),
        ),
        (
            "workspace_id".to_string(),
            record
                .workspace_id
                .as_ref()
                .map(|v| Value::String(v.clone()))
                .unwrap_or(Value::Null),
        ),
        (
            "user_id".to_string(),
            record
                .user_id
                .as_ref()
                .map(|v| Value::String(v.clone()))
                .unwrap_or(Value::Null),
        ),
        (
            "session_id".to_string(),
            record
                .session_id
                .as_ref()
                .map(|v| Value::String(v.clone()))
                .unwrap_or(Value::Null),
        ),
        (
            "scope".to_string(),
            Value::String(
                match record.scope {
                    MemoryScope::User => "user",
                    MemoryScope::Session => "session",
                    MemoryScope::Workspace => "workspace",
                    MemoryScope::Global => "global",
                }
                .to_string(),
            ),
        ),
        (
            "memory_type".to_string(),
            Value::String(
                match record.memory_type {
                    MemoryType::Profile => "profile",
                    MemoryType::Episodic => "episodic",
                    MemoryType::Policy => "policy",
                    MemoryType::ArtifactSummary => "artifact_summary",
                }
                .to_string(),
            ),
        ),
        (
            "status".to_string(),
            Value::String(
                match record.status {
                    MemoryStatus::Active => "active",
                    MemoryStatus::Superseded => "superseded",
                    MemoryStatus::Quarantined => "quarantined",
                    MemoryStatus::Deleted => "deleted",
                }
                .to_string(),
            ),
        ),
        (
            "confidence".to_string(),
            Value::from(f64::from(record.confidence)),
        ),
        (
            "importance".to_string(),
            Value::from(f64::from(record.importance)),
        ),
        (
            "decay_half_life_hours".to_string(),
            record
                .decay_half_life_hours
                .map(|v| Value::from(u64::from(v)))
                .unwrap_or(Value::Null),
        ),
        (
            "source_run_id".to_string(),
            record
                .source_run_id
                .map(|v| Value::String(v.to_string()))
                .unwrap_or(Value::Null),
        ),
        ("json_fields".to_string(), record.json_fields.clone()),
    ]))
}

fn extract_json_fields(meta_obj: Option<&Map<String, Value>>) -> Value {
    let Some(meta_obj) = meta_obj else {
        return Value::Null;
    };

    if let Some(v) = meta_obj.get("json_fields") {
        return v.clone();
    }

    let mut custom = Map::new();
    for (k, v) in meta_obj {
        if RESERVED_METADATA_KEYS.contains(&k.as_str()) {
            continue;
        }
        custom.insert(k.clone(), v.clone());
    }
    if custom.is_empty() {
        Value::Null
    } else {
        Value::Object(custom)
    }
}

fn metadata_matches(metadata: &Value, filter: &Value) -> bool {
    match (metadata, filter) {
        (Value::Object(meta_obj), Value::Object(filter_obj)) => filter_obj.iter().all(|(k, v)| {
            meta_obj
                .get(k)
                .map(|meta_value| metadata_matches(meta_value, v))
                .unwrap_or(false)
        }),
        _ => metadata == filter,
    }
}

fn value_as_uuid(v: &Value) -> Option<Uuid> {
    v.as_str().and_then(|s| Uuid::parse_str(s).ok())
}

fn value_as_str(v: &Value) -> Option<&str> {
    v.as_str().map(str::trim).filter(|s| !s.is_empty())
}

fn value_as_f32(v: &Value) -> Option<f32> {
    v.as_f64().map(|n| n as f32)
}

fn value_as_u32(v: &Value) -> Option<u32> {
    v.as_u64().and_then(|n| u32::try_from(n).ok())
}

fn parse_scope(v: &str) -> Option<MemoryScope> {
    match v {
        "user" => Some(MemoryScope::User),
        "session" => Some(MemoryScope::Session),
        "workspace" => Some(MemoryScope::Workspace),
        "global" => Some(MemoryScope::Global),
        _ => None,
    }
}

fn parse_memory_type(v: &str) -> Option<MemoryType> {
    match v {
        "profile" => Some(MemoryType::Profile),
        "episodic" => Some(MemoryType::Episodic),
        "policy" => Some(MemoryType::Policy),
        "artifact_summary" => Some(MemoryType::ArtifactSummary),
        _ => None,
    }
}

fn parse_status(v: &str) -> Option<MemoryStatus> {
    match v {
        "active" => Some(MemoryStatus::Active),
        "superseded" => Some(MemoryStatus::Superseded),
        "quarantined" => Some(MemoryStatus::Quarantined),
        "deleted" => Some(MemoryStatus::Deleted),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn adapter_search_returns_inserted_content() {
        let adapter = FoxstashCoreAdapter::new(64);
        let key = Uuid::new_v4().to_string();

        adapter
            .upsert_embedding(
                &key,
                "den keeps run traces",
                serde_json::json!({"scope":"session"}),
            )
            .expect("upsert");

        let out = adapter
            .similarity_search("run traces", 3, None)
            .expect("search");
        assert!(!out.is_empty());
    }

    #[test]
    fn invalid_doc_id_maps_to_stable_generated_memory_id() {
        let adapter = FoxstashCoreAdapter::new(32);

        adapter
            .upsert_embedding(
                "not-a-uuid",
                "stable fallback id",
                serde_json::json!({"scope":"session"}),
            )
            .expect("upsert");

        let first = adapter
            .similarity_search("stable fallback id", 1, None)
            .expect("first search");
        let second = adapter
            .similarity_search("stable fallback id", 1, None)
            .expect("second search");

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert_eq!(first[0].memory_id, second[0].memory_id);
    }

    #[test]
    fn metadata_maps_into_memory_record_fields() {
        let adapter = FoxstashCoreAdapter::new(64);
        let memory_id = Uuid::new_v4();
        let run_id = Uuid::new_v4();

        adapter
            .upsert_embedding(
                &memory_id.to_string(),
                "remember this profile detail",
                serde_json::json!({
                    "workspace_id": "w-1",
                    "user_id": "u-7",
                    "session_id": "s-2",
                    "scope": "user",
                    "memory_type": "profile",
                    "status": "superseded",
                    "confidence": 0.91,
                    "importance": 0.88,
                    "decay_half_life_hours": 72,
                    "source_run_id": run_id.to_string(),
                    "json_fields": {"origin":"unit_test"}
                }),
            )
            .expect("upsert");

        let out = adapter
            .similarity_search("profile detail", 3, None)
            .expect("search");
        assert!(!out.is_empty());
        let record = &out[0];

        assert_eq!(record.memory_id, memory_id);
        assert_eq!(record.workspace_id.as_deref(), Some("w-1"));
        assert_eq!(record.user_id.as_deref(), Some("u-7"));
        assert_eq!(record.session_id.as_deref(), Some("s-2"));
        assert_eq!(record.scope, MemoryScope::User);
        assert_eq!(record.memory_type, MemoryType::Profile);
        assert_eq!(record.status, crate::MemoryStatus::Superseded);
        assert_eq!(record.source_run_id, Some(run_id));
        assert_eq!(record.decay_half_life_hours, Some(72));
        assert_eq!(
            record.json_fields.get("origin").and_then(|v| v.as_str()),
            Some("unit_test")
        );
    }

    #[test]
    fn deterministic_embedder_is_stable() {
        let embedder = DeterministicEmbedder::new(32);
        let a = embedder
            .embed("foxloom deterministic vectors")
            .expect("embed");
        let b = embedder
            .embed("foxloom deterministic vectors")
            .expect("embed");
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_status_is_quarantined_on_read() {
        let adapter = FoxstashCoreAdapter::new(64);
        let id = Uuid::new_v4().to_string();
        adapter
            .upsert_embedding(
                &id,
                "suspicious fact",
                serde_json::json!({
                    "scope": "session",
                    "status": "nonsense"
                }),
            )
            .expect("upsert");

        let out = adapter
            .similarity_search("suspicious", 3, None)
            .expect("search");
        assert!(!out.is_empty());
        assert_eq!(out[0].status, MemoryStatus::Quarantined);
    }

    #[test]
    fn metadata_filter_nested_object_matches() {
        let adapter = FoxstashCoreAdapter::new(64);
        let id = Uuid::new_v4().to_string();
        adapter
            .upsert_embedding(
                &id,
                "nested metadata",
                serde_json::json!({
                    "scope": "session",
                    "json_fields": {"origin": "unit_test", "kind": {"tier": "gold"}}
                }),
            )
            .expect("upsert");

        let out = adapter
            .similarity_search(
                "nested metadata",
                3,
                Some(serde_json::json!({"json_fields": {"kind": {"tier": "gold"}}})),
            )
            .expect("search");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn snapshot_and_rebuild_records_round_trip() {
        let adapter = FoxstashCoreAdapter::new(64);
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        adapter
            .upsert_embedding(
                &id_a.to_string(),
                "primary owner is atlas",
                serde_json::json!({"scope":"session","memory_type":"policy","status":"active"}),
            )
            .expect("upsert a");
        adapter
            .upsert_embedding(
                &id_b.to_string(),
                "backup owner is apollo",
                serde_json::json!({"scope":"session","memory_type":"policy","status":"superseded"}),
            )
            .expect("upsert b");

        let snapshot = adapter.snapshot_records();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(adapter.len(), 2);

        adapter.rebuild_from_records(&snapshot).expect("rebuild");
        assert_eq!(adapter.len(), 2);

        let out = adapter
            .similarity_search("owner", 5, Some(serde_json::json!({"scope":"session"})))
            .expect("search");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn concurrent_upsert_search_and_rebuild_soak() {
        let adapter = Arc::new(FoxstashCoreAdapter::new(64));
        let threads = 6usize;
        let docs_per_thread = 80usize;

        let mut joins = Vec::new();
        for t in 0..threads {
            let adapter = adapter.clone();
            joins.push(thread::spawn(move || {
                for i in 0..docs_per_thread {
                    let id = Uuid::new_v4().to_string();
                    let text = format!("soak memory t{} i{} owner atlas", t, i);
                    adapter
                        .upsert_embedding(
                            &id,
                            &text,
                            serde_json::json!({
                                "scope":"session",
                                "memory_type":"episodic",
                                "status":"active",
                                "thread": t,
                            }),
                        )
                        .expect("upsert");
                    if i % 10 == 0 {
                        let _ = adapter
                            .similarity_search(
                                "owner atlas",
                                5,
                                Some(serde_json::json!({"scope":"session","status":"active"})),
                            )
                            .expect("search");
                    }
                }
            }));
        }

        for join in joins {
            join.join().expect("join soak thread");
        }

        let expected = threads * docs_per_thread;
        assert_eq!(adapter.len(), expected);

        let snapshot = adapter.snapshot_records();
        assert_eq!(snapshot.len(), expected);

        adapter.rebuild_from_records(&snapshot).expect("rebuild");
        assert_eq!(adapter.len(), expected);

        let out = adapter
            .similarity_search(
                "owner atlas",
                10,
                Some(serde_json::json!({"scope":"session","status":"active"})),
            )
            .expect("search after rebuild");
        assert!(!out.is_empty());
    }
}
