use foxstash_core::index::HNSWIndex;
use foxstash_core::Document;
use parking_lot::Mutex;
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

use crate::{FoxstashAdapter, MemoryRecord, MemoryScope, MemoryType};

#[derive(Clone)]
pub struct FoxstashCoreAdapter {
    index: Arc<Mutex<HNSWIndex>>,
    embedding_dim: usize,
}

impl FoxstashCoreAdapter {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            index: Arc::new(Mutex::new(HNSWIndex::with_defaults(embedding_dim))),
            embedding_dim,
        }
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.embedding_dim];
        if text.is_empty() {
            return vec;
        }

        for (i, b) in text.bytes().enumerate() {
            let idx = i % self.embedding_dim;
            let value = (b as f32) / 255.0;
            vec[idx] += value;
        }

        // L2-normalize for cosine-style retrieval
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }
}

impl FoxstashAdapter for FoxstashCoreAdapter {
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), String> {
        let doc = Document {
            id: key.to_string(),
            content: text.to_string(),
            embedding: self.embed_text(text),
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
        let query_embedding = self.embed_text(query);
        let results = self
            .index
            .lock()
            .search(&query_embedding, top_k)
            .map_err(|e| format!("foxstash search failed: {e}"))?;

        let mut out = Vec::new();
        for result in results {
            if let Some(filter) = &metadata_filter {
                if let Some(metadata) = &result.metadata {
                    if !metadata_matches(metadata, filter) {
                        continue;
                    }
                }
            }

            let memory_id = Uuid::parse_str(&result.id).unwrap_or_else(|_| Uuid::new_v4());
            let mut memory = MemoryRecord::new(
                memory_id,
                MemoryScope::Session,
                MemoryType::Episodic,
                result.content,
            );
            memory.embedding_ref = Some(result.id);
            memory.confidence = result.score.clamp(0.0, 1.0);
            memory.importance = 0.5;

            if let Some(metadata) = result.metadata {
                memory.json_fields = metadata;
            }

            out.push(memory);
        }

        Ok(out)
    }
}

fn metadata_matches(metadata: &Value, filter: &Value) -> bool {
    let Some(filter_obj) = filter.as_object() else {
        return true;
    };
    let Some(meta_obj) = metadata.as_object() else {
        return false;
    };

    filter_obj
        .iter()
        .all(|(k, v)| meta_obj.get(k).map(|mv| mv == v).unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
