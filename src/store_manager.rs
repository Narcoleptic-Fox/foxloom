use std::{cmp::Ordering, collections::HashMap};

use uuid::Uuid;

use crate::{MemoryRecord, MemoryScope};

#[derive(Debug, Clone)]
pub struct StoreManagerConfig {
    pub top_k: usize,
    /// When `true`, attempt to extract entity keys from record text (splitting on
    /// `" is "` or `":"`) when `json_fields.entity` is not set. This heuristic can
    /// produce false positives (e.g. `"This is a good policy"` → entity `"This"`),
    /// so it is disabled by default. Prefer setting `json_fields.entity` explicitly.
    pub text_entity_extraction: bool,
}

impl Default for StoreManagerConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            text_entity_extraction: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoreManager {
    config: StoreManagerConfig,
}

#[derive(Debug, Clone)]
pub struct ScopeQuery {
    pub session_id: String,
    pub workspace_id: Option<String>,
    pub user_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RetrievalCandidate {
    pub record: MemoryRecord,
    pub score: f32,
    pub similarity: f32,
    pub source: String,
    pub selected_from_scope: MemoryScope,
    pub selection_reason: String,
}

#[derive(Debug, Clone, Default)]
pub struct MergeStats {
    pub filtered_candidates: usize,
    pub kept_candidates: usize,
}

impl StoreManager {
    pub fn new(config: StoreManagerConfig) -> Self {
        Self { config }
    }

    pub fn top_k(&self) -> usize {
        self.config.top_k.max(1)
    }

    pub fn scope_allows(&self, scope: &ScopeQuery, record: &MemoryRecord) -> bool {
        let session_match = record.session_id.as_deref() == Some(scope.session_id.as_str());
        let workspace_match = scope
            .workspace_id
            .as_deref()
            .is_some_and(|workspace_id| record.workspace_id.as_deref() == Some(workspace_id));
        let user_match = scope
            .user_id
            .as_deref()
            .is_some_and(|user_id| record.user_id.as_deref() == Some(user_id));
        session_match || workspace_match || user_match || record.scope == MemoryScope::Global
    }

    pub fn merge_scoped_candidates(
        &self,
        candidates: Vec<RetrievalCandidate>,
    ) -> (Vec<RetrievalCandidate>, MergeStats) {
        let mut by_id = HashMap::<Uuid, RetrievalCandidate>::new();
        for candidate in candidates {
            by_id.entry(candidate.record.memory_id).or_insert(candidate);
        }

        let mut by_entity = HashMap::<String, Vec<RetrievalCandidate>>::new();
        let mut kept = Vec::new();
        let mut filtered = 0usize;

        for candidate in by_id.into_values() {
            if let Some(entity) = normalized_entity_key(&candidate.record, self.config.text_entity_extraction) {
                by_entity.entry(entity).or_default().push(candidate);
            } else {
                kept.push(mark_non_conflicting(candidate));
            }
        }

        for (_, mut conflicts) in by_entity {
            if conflicts.is_empty() {
                continue;
            }
            conflicts.sort_by(compare_scope_then_score);
            let mut winner = conflicts.remove(0);
            if let Some(second) = conflicts.first() {
                winner.selection_reason = selection_reason(second, &winner);
            } else {
                winner.selection_reason = "non_conflicting".to_string();
            }
            filtered = filtered.saturating_add(conflicts.len());
            kept.push(winner);
        }

        apply_scope_precedence(&mut kept);
        let top_k = self.top_k();
        if kept.len() > top_k {
            filtered = filtered.saturating_add(kept.len() - top_k);
            kept.truncate(top_k);
        }
        let kept_count = kept.len();
        (
            kept,
            MergeStats {
                filtered_candidates: filtered,
                kept_candidates: kept_count,
            },
        )
    }

    pub fn scope_rank(scope: &MemoryScope) -> usize {
        scope_rank(scope)
    }
}

fn apply_scope_precedence(items: &mut [RetrievalCandidate]) {
    items.sort_by(compare_scope_then_score);
}

fn compare_scope_then_score(left: &RetrievalCandidate, right: &RetrievalCandidate) -> Ordering {
    let left_rank = scope_rank(&left.record.scope);
    let right_rank = scope_rank(&right.record.scope);
    left_rank
        .cmp(&right_rank)
        .then_with(|| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| {
            left.record
                .memory_id
                .as_u128()
                .cmp(&right.record.memory_id.as_u128())
        })
}

fn selection_reason(existing: &RetrievalCandidate, replacement: &RetrievalCandidate) -> String {
    let existing_rank = scope_rank(&existing.record.scope);
    let replacement_rank = scope_rank(&replacement.record.scope);
    if replacement_rank < existing_rank {
        "higher_precedence".to_string()
    } else {
        "higher_score".to_string()
    }
}

fn mark_non_conflicting(mut candidate: RetrievalCandidate) -> RetrievalCandidate {
    candidate.selection_reason = "non_conflicting".to_string();
    candidate
}

fn scope_rank(scope: &MemoryScope) -> usize {
    match scope {
        MemoryScope::Workspace => 0,
        MemoryScope::Session => 1,
        MemoryScope::User => 2,
        MemoryScope::Global => 3,
    }
}

fn normalized_entity_key(record: &MemoryRecord, text_extraction: bool) -> Option<String> {
    if let Some(entity) = record
        .json_fields
        .get("entity")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return Some(normalize_entity_text(entity));
    }

    if text_extraction {
        return extract_entity_from_text(&record.text).map(normalize_entity_text);
    }

    None
}

fn extract_entity_from_text(text: &str) -> Option<&str> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((lhs, _)) = trimmed.split_once(" is ") {
        let lhs = lhs.trim();
        if !lhs.is_empty() {
            return Some(lhs);
        }
    }
    if let Some((lhs, _)) = trimmed.split_once(':') {
        let lhs = lhs.trim();
        if !lhs.is_empty() {
            return Some(lhs);
        }
    }
    None
}

fn normalize_entity_text(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut prev_space = false;
    for c in value.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_space = false;
        } else if !prev_space {
            out.push(' ');
            prev_space = true;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::MemoryType;

    use super::*;

    fn record(id: u128, scope: MemoryScope, text: &str, entity: Option<&str>) -> MemoryRecord {
        let mut r = MemoryRecord::new(Uuid::from_u128(id), scope, MemoryType::Policy, text.to_string());
        r.json_fields = entity.map(|e| json!({"entity": e})).unwrap_or(serde_json::Value::Null);
        r
    }

    fn candidate(record: MemoryRecord, score: f32) -> RetrievalCandidate {
        RetrievalCandidate {
            selected_from_scope: record.scope.clone(),
            selection_reason: "non_conflicting".to_string(),
            similarity: 0.8,
            source: "test".to_string(),
            score,
            record,
        }
    }

    #[test]
    fn merge_prefers_workspace_over_global_for_same_entity() {
        let mgr = StoreManager::new(StoreManagerConfig { top_k: 8, ..Default::default() });
        let global = candidate(record(1, MemoryScope::Global, "owner is team atlas", Some("owner")), 0.95);
        let workspace = candidate(record(2, MemoryScope::Workspace, "owner is team zeus", Some("owner")), 0.80);

        let (merged, stats) = mgr.merge_scoped_candidates(vec![global, workspace]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].record.memory_id, Uuid::from_u128(2));
        assert_eq!(merged[0].selection_reason, "higher_precedence");
        assert_eq!(stats.filtered_candidates, 1);
    }

    #[test]
    fn merge_uses_uuid_tiebreaker_deterministically() {
        let mgr = StoreManager::new(StoreManagerConfig { top_k: 8, ..Default::default() });
        let a = candidate(record(1, MemoryScope::Session, "a", None), 1.0);
        let b = candidate(record(2, MemoryScope::Session, "b", None), 1.0);
        let (merged, _) = mgr.merge_scoped_candidates(vec![b, a]);
        assert_eq!(merged[0].record.memory_id, Uuid::from_u128(1));
        assert_eq!(merged[1].record.memory_id, Uuid::from_u128(2));
    }

    #[test]
    fn scope_filter_blocks_cross_workspace_leakage() {
        let mgr = StoreManager::new(StoreManagerConfig { top_k: 8, ..Default::default() });
        let scope = ScopeQuery {
            session_id: "s1".to_string(),
            workspace_id: Some("wa".to_string()),
            user_id: Some("u1".to_string()),
        };
        let mut other_ws = record(11, MemoryScope::Workspace, "owner is team zeus", Some("owner"));
        other_ws.workspace_id = Some("wb".to_string());
        assert!(!mgr.scope_allows(&scope, &other_ws));

        let mut global = record(12, MemoryScope::Global, "timezone is UTC", Some("timezone"));
        global.workspace_id = None;
        assert!(mgr.scope_allows(&scope, &global));
    }
}


