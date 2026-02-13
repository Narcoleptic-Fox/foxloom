use serde_json::Value;

use crate::MemoryRecord;

pub trait FoxstashAdapter: Send + Sync {
    fn upsert_embedding(&self, key: &str, text: &str, metadata: Value) -> Result<(), String>;
    fn delete_embedding(&self, key: &str) -> Result<(), String>;

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
