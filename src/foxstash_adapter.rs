use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

use foxstash_core::index::HNSWIndex;
use foxstash_core::storage::incremental::{
    IncrementalConfig, IncrementalStorage, IndexMetadata, RecoveryHelper, StorageStats,
    WalOperation,
};
use foxstash_core::Document;
use foxstash_core::SearchResult;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{FoxstashAdapter, MemoryRecord, MemoryScope, MemoryStatus, MemoryType};

const FALLBACK_UUID_NAMESPACE: Uuid = Uuid::from_bytes([
    0x84, 0x34, 0x5a, 0x26, 0x3b, 0xa7, 0x4f, 0xb0, 0x88, 0x26, 0x95, 0x5f, 0x6d, 0x65, 0x60, 0xcf,
]);
const OVERFETCH_CAP_MULTIPLIER: usize = 8;

const RESERVED_METADATA_KEYS: [&str; 13] = [
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
    "updated_at",
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
    tombstones: Arc<Mutex<HashSet<String>>>,
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
            tombstones: Arc::new(Mutex::new(HashSet::new())),
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
        let embedding_dim = self.embedder.dimension();
        let mut rebuilt = HNSWIndex::with_defaults(embedding_dim);
        for doc in documents {
            rebuilt
                .add(doc.clone())
                .map_err(|e| format!("foxstash restore failed: {e}"))?;
        }
        {
            let mut index = self.index.lock();
            *index = rebuilt;
        }
        self.tombstones.lock().clear();
        Ok(())
    }

    pub fn snapshot_records(&self) -> Vec<MemoryRecord> {
        self.snapshot_documents()
            .into_iter()
            .map(document_to_memory_record)
            .collect()
    }

    pub fn rebuild_from_records(&self, records: &[MemoryRecord]) -> Result<(), String> {
        let mut docs = Vec::with_capacity(records.len());
        for record in records {
            let embedding = self.embedder.embed(&record.text)?;
            docs.push(Document {
                id: record.memory_id.to_string(),
                content: record.text.clone(),
                embedding,
                metadata: Some(normalize_document_metadata(
                    &record.memory_id.to_string(),
                    metadata_from_record(record),
                )),
            });
        }
        self.rebuild_from_documents(&docs)
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
            .map_err(|e| format!("foxstash add failed: {e}"))?;
        self.tombstones.lock().remove(key);
        Ok(())
    }

    fn delete_embedding(&self, key: &str) -> Result<(), String> {
        self.tombstones.lock().insert(key.to_string());
        Ok(())
    }

    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, String> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let query_embedding = self.embedder.embed(query)?;
        let total = self.index.lock().len();
        if total == 0 {
            return Ok(vec![]);
        }
        let tombstones = self.tombstones.lock().clone();
        let max_fetch = top_k
            .saturating_mul(OVERFETCH_CAP_MULTIPLIER)
            .max(1)
            .min(total);

        let mut fetch_k = top_k.max(1).min(max_fetch);
        let mut out = Vec::new();
        loop {
            let mut results = self
                .index
                .lock()
                .search(&query_embedding, fetch_k)
                .map_err(|e| format!("foxstash search failed: {e}"))?;
            results.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.id.cmp(&right.id))
            });
            out.clear();
            let mut seen_ids = HashSet::new();
            for result in results {
                if tombstones.contains(&result.id) {
                    continue;
                }
                if !seen_ids.insert(result.id.clone()) {
                    continue;
                }
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
                if out.len() >= top_k {
                    break;
                }
            }
            if out.len() >= top_k || fetch_k >= max_fetch {
                break;
            }
            let next_fetch = (fetch_k.saturating_mul(2)).min(max_fetch);
            if next_fetch == fetch_k {
                break;
            }
            fetch_k = next_fetch;
        }
        out.truncate(top_k);

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
    let updated_at = meta_obj
        .and_then(|m| m.get("updated_at"))
        .and_then(value_as_str)
        .and_then(|v| chrono::DateTime::parse_from_rfc3339(v).ok())
        .map(|v| v.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now);
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
        updated_at,
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
        (
            "updated_at".to_string(),
            Value::String(record.updated_at.to_rfc3339()),
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

#[derive(Debug, Clone)]
pub struct PersistentConfig {
    pub incremental: IncrementalConfig,
    pub delete_compaction_threshold: usize,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        Self {
            incremental: IncrementalConfig::default(),
            delete_compaction_threshold: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedDocument {
    id: String,
    content: String,
    embedding: Vec<f32>,
    metadata_json: Option<String>,
}

impl PersistedDocument {
    fn from_document(doc: &Document) -> Result<Self, String> {
        let metadata_json = match &doc.metadata {
            Some(meta) => Some(
                serde_json::to_string(meta)
                    .map_err(|e| format!("serialize metadata for checkpoint failed: {e}"))?,
            ),
            None => None,
        };
        Ok(Self {
            id: doc.id.clone(),
            content: doc.content.clone(),
            embedding: doc.embedding.clone(),
            metadata_json,
        })
    }

    fn into_document(self) -> Result<Document, String> {
        let metadata = match self.metadata_json {
            Some(raw) => Some(
                serde_json::from_str(&raw)
                    .map_err(|e| format!("parse checkpoint metadata failed: {e}"))?,
            ),
            None => None,
        };
        Ok(Document {
            id: self.id,
            content: self.content,
            embedding: self.embedding,
            metadata,
        })
    }
}

fn encode_wal_content(text: &str, metadata: Option<&Value>) -> Result<String, String> {
    let payload = serde_json::json!({
        "v": 1,
        "text": text,
        "metadata": metadata.cloned().unwrap_or(Value::Null),
    });
    serde_json::to_string(&payload).map_err(|e| format!("encode WAL payload failed: {e}"))
}

fn decode_wal_document(mut doc: Document) -> Result<Document, String> {
    let parsed: Value = serde_json::from_str(&doc.content)
        .map_err(|e| format!("decode WAL payload failed: {e}"))?;
    if parsed.get("v").and_then(Value::as_u64) != Some(1) {
        return Err("unsupported WAL payload version".to_string());
    }
    if let Some(text) = parsed.get("text").and_then(Value::as_str) {
        doc.content = text.to_string();
    }
    let metadata = parsed.get("metadata").cloned().unwrap_or(Value::Null);
    doc.metadata = if metadata.is_null() {
        None
    } else {
        Some(metadata)
    };
    Ok(doc)
}
#[derive(Clone)]
pub struct PersistentFoxstashCoreAdapter {
    inner: FoxstashCoreAdapter,
    storage: Arc<Mutex<IncrementalStorage>>,
    active_docs: Arc<Mutex<HashMap<String, Document>>>,
    delete_count_since_compaction: Arc<Mutex<usize>>,
    config: PersistentConfig,
}

impl PersistentFoxstashCoreAdapter {
    pub fn new(
        embedding_dim: usize,
        base_path: impl AsRef<Path>,
        config: PersistentConfig,
    ) -> Result<Self, String> {
        Self::with_embedder_and_storage(
            Arc::new(DeterministicEmbedder::new(embedding_dim)),
            base_path,
            config,
        )
    }

    pub fn with_embedder_and_storage(
        embedder: Arc<dyn TextEmbedder>,
        base_path: impl AsRef<Path>,
        config: PersistentConfig,
    ) -> Result<Self, String> {
        let inner = FoxstashCoreAdapter::with_embedder(embedder);
        let storage = IncrementalStorage::new(base_path, config.incremental.clone())
            .map_err(|e| format!("foxstash incremental open failed: {e}"))?;
        let adapter = Self {
            inner,
            storage: Arc::new(Mutex::new(storage)),
            active_docs: Arc::new(Mutex::new(HashMap::new())),
            delete_count_since_compaction: Arc::new(Mutex::new(0)),
            config,
        };
        adapter.recover()?;
        Ok(adapter)
    }

    #[cfg(feature = "onnx-embedder")]
    pub fn try_from_onnx_files(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        ort_dylib_path: Option<std::path::PathBuf>,
        base_path: impl AsRef<Path>,
        config: PersistentConfig,
    ) -> Result<Self, String> {
        let embedder = Arc::new(OnnxEmbedder::from_files(
            model_path,
            tokenizer_path,
            ort_dylib_path,
        )?);
        Self::with_embedder_and_storage(embedder, base_path, config)
    }

    fn recover(&self) -> Result<(), String> {
        let mut docs_by_id = HashMap::<String, Document>::new();
        {
            let storage = self.storage.lock();
            if let Some((checkpoint_docs, _meta)) = storage
                .load_checkpoint::<Vec<PersistedDocument>>()
                .map_err(|e| format!("foxstash load checkpoint failed: {e}"))?
            {
                for persisted in checkpoint_docs {
                    let doc = persisted.into_document()?;
                    docs_by_id.insert(doc.id.clone(), doc);
                }
            }

            let helper = RecoveryHelper::new(&storage);
            helper
                .replay_wal(|op| {
                    match op {
                        WalOperation::Add(doc) => {
                            let decoded = decode_wal_document(doc.clone())
                                .map_err(foxstash_core::RagError::StorageError)?;
                            docs_by_id.insert(decoded.id.clone(), decoded);
                        }
                        WalOperation::Remove(id) => {
                            docs_by_id.remove(id);
                        }
                        WalOperation::Clear => {
                            docs_by_id.clear();
                        }
                        WalOperation::Checkpoint { .. } => {}
                    }
                    Ok(())
                })
                .map_err(|e| format!("foxstash WAL replay failed: {e}"))?;
        }

        let docs = docs_by_id.values().cloned().collect::<Vec<_>>();
        self.inner.rebuild_from_documents(&docs)?;
        *self.active_docs.lock() = docs_by_id;
        *self.delete_count_since_compaction.lock() = 0;
        Ok(())
    }

    pub fn storage_stats(&self) -> StorageStats {
        self.storage.lock().stats()
    }

    pub fn force_checkpoint(&self) -> Result<(), String> {
        self.checkpoint_if_needed(true)
    }

    pub fn sync(&self) -> Result<(), String> {
        self.storage
            .lock()
            .sync()
            .map_err(|e| format!("foxstash WAL sync failed: {e}"))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn checkpoint_if_needed(&self, force: bool) -> Result<(), String> {
        let docs = self
            .active_docs
            .lock()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        let persisted = docs
            .iter()
            .map(PersistedDocument::from_document)
            .collect::<Result<Vec<_>, _>>()?;
        let mut storage = self.storage.lock();
        if !force && !storage.needs_checkpoint() {
            return Ok(());
        }
        let meta = IndexMetadata {
            document_count: docs.len(),
            embedding_dim: self.inner.embedder.dimension(),
            index_type: "hnsw".to_string(),
        };
        storage
            .checkpoint(&persisted, meta)
            .map_err(|e| format!("foxstash checkpoint failed: {e}"))?;
        Ok(())
    }

    fn compact_if_needed(&self) -> Result<(), String> {
        let delete_count = *self.delete_count_since_compaction.lock();
        if delete_count < self.config.delete_compaction_threshold {
            return Ok(());
        }

        let docs = self
            .active_docs
            .lock()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        self.inner.rebuild_from_documents(&docs)?;
        *self.delete_count_since_compaction.lock() = 0;
        self.checkpoint_if_needed(true)
    }
}

impl FoxstashAdapter for PersistentFoxstashCoreAdapter {
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), String> {
        let embedding = self.inner.embedder.embed(text)?;
        let normalized = normalize_document_metadata(key, metadata);
        let doc = Document {
            id: key.to_string(),
            content: text.to_string(),
            embedding,
            metadata: Some(normalized),
        };
        let wal_doc = Document {
            id: doc.id.clone(),
            content: encode_wal_content(&doc.content, doc.metadata.as_ref())?,
            embedding: doc.embedding.clone(),
            metadata: None,
        };

        {
            self.storage
                .lock()
                .log_add(&wal_doc)
                .map_err(|e| format!("foxstash WAL add failed: {e}"))?;
        }

        {
            self.inner
                .index
                .lock()
                .add(doc.clone())
                .map_err(|e| format!("foxstash add failed: {e}"))?;
            self.inner.tombstones.lock().remove(key);
        }
        self.active_docs.lock().insert(key.to_string(), doc);
        self.checkpoint_if_needed(false)
    }

    fn delete_embedding(&self, key: &str) -> Result<(), String> {
        {
            self.storage
                .lock()
                .log_remove(key)
                .map_err(|e| format!("foxstash WAL remove failed: {e}"))?;
        }
        self.inner.tombstones.lock().insert(key.to_string());
        self.active_docs.lock().remove(key);
        *self.delete_count_since_compaction.lock() += 1;
        self.compact_if_needed()?;
        self.checkpoint_if_needed(false)
    }

    fn similarity_search(
        &self,
        query: &str,
        top_k: usize,
        metadata_filter: Option<Value>,
    ) -> Result<Vec<MemoryRecord>, String> {
        self.inner.similarity_search(query, top_k, metadata_filter)
    }

    fn batch_upsert_embeddings(&self, items: &[(String, String, Value)]) -> Result<(), String> {
        let mut docs = Vec::with_capacity(items.len());
        for (key, text, metadata) in items {
            let embedding = self.inner.embedder.embed(text)?;
            let normalized = normalize_document_metadata(key, metadata.clone());
            docs.push(Document {
                id: key.clone(),
                content: text.clone(),
                embedding,
                metadata: Some(normalized),
            });
        }

        {
            let mut storage = self.storage.lock();
            for doc in &docs {
                let wal_doc = Document {
                    id: doc.id.clone(),
                    content: encode_wal_content(&doc.content, doc.metadata.as_ref())?,
                    embedding: doc.embedding.clone(),
                    metadata: None,
                };
                storage
                    .log_add(&wal_doc)
                    .map_err(|e| format!("foxstash WAL add failed: {e}"))?;
            }
        }

        {
            let mut index = self.inner.index.lock();
            let mut tombstones = self.inner.tombstones.lock();
            let mut active = self.active_docs.lock();
            for doc in docs {
                index
                    .add(doc.clone())
                    .map_err(|e| format!("foxstash add failed: {e}"))?;
                tombstones.remove(&doc.id);
                active.insert(doc.id.clone(), doc);
            }
        }
        self.checkpoint_if_needed(false)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::thread;
    use tempfile::TempDir;

    struct FlakyEmbedder;

    impl TextEmbedder for FlakyEmbedder {
        fn dimension(&self) -> usize {
            8
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
            if text.contains("FAIL_EMBED") {
                return Err("forced embed failure".to_string());
            }
            Ok(vec![1.0; 8])
        }
    }

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

    #[test]
    fn delete_embedding_tombstones_results_and_upsert_clears_tombstone() {
        let adapter = FoxstashCoreAdapter::new(64);
        let id = Uuid::new_v4().to_string();
        adapter
            .upsert_embedding(
                &id,
                "team atlas owns service",
                serde_json::json!({"scope":"session","status":"active"}),
            )
            .expect("upsert");

        let before = adapter
            .similarity_search(
                "owns service",
                5,
                Some(serde_json::json!({"scope":"session","status":"active"})),
            )
            .expect("search before delete");
        assert_eq!(before.len(), 1);

        adapter.delete_embedding(&id).expect("delete embedding");
        let deleted = adapter
            .similarity_search(
                "owns service",
                5,
                Some(serde_json::json!({"scope":"session","status":"active"})),
            )
            .expect("search after delete");
        assert!(deleted.is_empty(), "tombstoned embeddings must be hidden");

        adapter
            .upsert_embedding(
                &id,
                "team atlas owns service",
                serde_json::json!({"scope":"session","status":"active"}),
            )
            .expect("re-upsert");
        let restored = adapter
            .similarity_search(
                "owns service",
                5,
                Some(serde_json::json!({"scope":"session","status":"active"})),
            )
            .expect("search after re-upsert");
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn search_overfetches_past_filtered_top_k() {
        let adapter = FoxstashCoreAdapter::new(64);
        for i in 0..40 {
            let id = Uuid::new_v4().to_string();
            adapter
                .upsert_embedding(
                    &id,
                    "shared query corpus owner",
                    serde_json::json!({"scope":"session","status":"active","bucket":"all"}),
                )
                .expect("upsert");
            if i < 35 {
                adapter.delete_embedding(&id).expect("tombstone");
            }
        }

        let out = adapter
            .similarity_search(
                "owner",
                5,
                Some(serde_json::json!({"scope":"session","status":"active"})),
            )
            .expect("search");
        assert_eq!(
            out.len(),
            5,
            "search should widen beyond initial top_k to find non-filtered records"
        );
    }

    #[test]
    fn rebuild_from_records_is_atomic_on_embedding_failure() {
        let adapter = FoxstashCoreAdapter::with_embedder(Arc::new(FlakyEmbedder));
        adapter
            .upsert_embedding(
                &Uuid::new_v4().to_string(),
                "stable baseline record",
                serde_json::json!({"scope":"session","status":"active"}),
            )
            .expect("seed baseline");
        let baseline = adapter.snapshot_records();
        assert_eq!(baseline.len(), 1);

        let mut bad_records = baseline.clone();
        bad_records.push(MemoryRecord {
            memory_id: Uuid::new_v4(),
            workspace_id: None,
            user_id: None,
            session_id: Some("s".to_string()),
            scope: MemoryScope::Session,
            memory_type: MemoryType::Episodic,
            text: "FAIL_EMBED trigger".to_string(),
            json_fields: Value::Null,
            embedding_ref: None,
            confidence: 0.7,
            importance: 0.5,
            decay_half_life_hours: None,
            status: MemoryStatus::Active,
            source_run_id: None,
            updated_at: Utc::now(),
        });

        let err = adapter.rebuild_from_records(&bad_records);
        assert!(err.is_err(), "expected forced embedding failure");
        assert_eq!(
            adapter.len(),
            1,
            "failed rebuild must not clear or partially replace existing index"
        );
    }
    #[test]
    fn persistent_adapter_recovers_from_wal_reopen() {
        let dir = TempDir::new().expect("tempdir");
        let cfg = PersistentConfig {
            incremental: IncrementalConfig::default().with_checkpoint_threshold(10_000),
            delete_compaction_threshold: 10_000,
        };
        {
            let adapter = PersistentFoxstashCoreAdapter::new(64, dir.path(), cfg.clone())
                .expect("persistent new");
            let a = Uuid::new_v4().to_string();
            let b = Uuid::new_v4().to_string();
            adapter
                .upsert_embedding(
                    &a,
                    "owner is atlas",
                    serde_json::json!({"scope":"session","status":"active"}),
                )
                .expect("upsert a");
            adapter
                .upsert_embedding(
                    &b,
                    "owner is zeus",
                    serde_json::json!({"scope":"session","status":"active"}),
                )
                .expect("upsert b");
            adapter.delete_embedding(&a).expect("delete a");
            adapter.sync().expect("sync");
        }

        let reopened = PersistentFoxstashCoreAdapter::new(64, dir.path(), cfg).expect("reopen");
        let out = reopened
            .similarity_search(
                "owner",
                5,
                Some(serde_json::json!({"scope":"session","status":"active"})),
            )
            .expect("search after reopen");
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("zeus"));
    }

    #[test]
    fn persistent_adapter_exposes_checkpoint_stats() {
        let dir = TempDir::new().expect("tempdir");
        let cfg = PersistentConfig {
            incremental: IncrementalConfig::default().with_checkpoint_threshold(2),
            delete_compaction_threshold: 10_000,
        };
        let adapter =
            PersistentFoxstashCoreAdapter::new(32, dir.path(), cfg).expect("persistent new");

        for _ in 0..3 {
            adapter
                .upsert_embedding(
                    &Uuid::new_v4().to_string(),
                    "checkpoint me",
                    serde_json::json!({"scope":"session","status":"active"}),
                )
                .expect("upsert");
        }

        let stats = adapter.storage_stats();
        assert!(
            stats.checkpoint_id.is_some(),
            "expected checkpoint after threshold"
        );
        assert!(stats.total_documents >= 2);
    }
}
