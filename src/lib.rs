pub mod adapter;
pub mod context_builder;
pub mod foxstash_adapter;
pub mod model;
pub mod scoring;
pub mod store_manager;

pub use adapter::FoxstashAdapter;
pub use context_builder::{
    build_active_context, build_active_context_with_estimator, BudgetEstimator, BuiltContext,
    ContextBudget, ContextBuildConfig, ContextItem, WordBudgetEstimator,
};
#[cfg(feature = "onnx-embedder")]
pub use foxstash_adapter::OnnxEmbedder;
pub use foxstash_adapter::{
    DeterministicEmbedder, FoxstashCoreAdapter, PersistentConfig, PersistentFoxstashCoreAdapter,
    TextEmbedder,
};
pub use model::{MemoryOp, MemoryRecord, MemoryScope, MemoryStatus, MemoryType};
pub use scoring::{decayed_importance, DecayConfig};
pub use store_manager::{
    MergeStats, RetrievalCandidate, ScopeQuery, StoreManager, StoreManagerConfig,
};

// MergeConfig is defined in this file (lib.rs) and exported directly.

/// Configuration for merge behavior.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Minimum confidence gap required to reject a candidate via `Noop`.
    ///
    /// A candidate is rejected when `candidate.confidence + supersede_threshold < current.confidence`.
    /// Lower values make supersede easier; higher values protect established memories.
    ///
    /// Default: `0.05`
    pub supersede_threshold: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            supersede_threshold: 0.05,
        }
    }
}

/// Determine the merge operation for a candidate memory against an optional existing record.
///
/// Uses default [`MergeConfig`]. See [`merge_candidate_with_config`] for custom thresholds.
///
/// # Decision tree
///
/// 1. **No existing record** → `Add`
/// 2. **Exact text match** → `Update` metadata (confidence/importance) or `Noop` if identical
/// 3. **Scope or type mismatch** → `Add` (different category, no conflict)
/// 4. **Existing not active** → `Add` (don't conflict with archived memories)
/// 5. **Candidate not active** → `Noop` (don't overwrite active with non-active)
/// 6. **Different entity** → `Add` (no conflict between unrelated facts)
/// 7. **Same entity, candidate confidence too low** → `Noop`
/// 8. **Same entity, sufficient confidence** → `Supersede`
///
/// # Entity detection
///
/// Entities are matched via `json_fields.entity`. If neither record has an entity set,
/// they are treated as non-conflicting (step 6 → `Add`).
///
/// # Examples
///
/// ```
/// use foxloom::{merge_candidate, MemoryRecord, MemoryScope, MemoryType, MemoryOp};
/// use uuid::Uuid;
///
/// // Adding a new memory (no existing)
/// let candidate = MemoryRecord::new(
///     Uuid::new_v4(), MemoryScope::Session, MemoryType::Episodic,
///     "deploy succeeded".to_string(),
/// );
/// let op = merge_candidate(None, &candidate);
/// assert!(matches!(op, MemoryOp::Add { .. }));
/// ```
pub fn merge_candidate(existing: Option<&MemoryRecord>, candidate: &MemoryRecord) -> MemoryOp {
    merge_candidate_with_config(existing, candidate, &MergeConfig::default())
}

/// Determine the merge operation with a custom [`MergeConfig`].
///
/// See [`merge_candidate`] for the decision tree and semantics.
pub fn merge_candidate_with_config(
    existing: Option<&MemoryRecord>,
    candidate: &MemoryRecord,
    config: &MergeConfig,
) -> MemoryOp {
    match existing {
        None => MemoryOp::Add {
            record: candidate.clone(),
            reason: "no_similar_memory".to_string(),
        },
        Some(current) => {
            if current.text == candidate.text {
                let mut patch = serde_json::Map::new();
                if (current.confidence - candidate.confidence).abs() > f32::EPSILON {
                    patch.insert(
                        "confidence".to_string(),
                        serde_json::json!(candidate.confidence),
                    );
                }
                if (current.importance - candidate.importance).abs() > f32::EPSILON {
                    patch.insert(
                        "importance".to_string(),
                        serde_json::json!(candidate.importance),
                    );
                }
                if patch.is_empty() {
                    return MemoryOp::Noop {
                        memory_id: current.memory_id,
                        reason: "duplicate_text".to_string(),
                    };
                }
                return MemoryOp::Update {
                    memory_id: current.memory_id,
                    patch: serde_json::Value::Object(patch),
                    reason: "duplicate_text_metadata_update".to_string(),
                };
            }

            if current.scope != candidate.scope || current.memory_type != candidate.memory_type {
                return MemoryOp::Add {
                    record: candidate.clone(),
                    reason: "new_memory_scope_or_type_mismatch".to_string(),
                };
            }

            if current.status != MemoryStatus::Active {
                return MemoryOp::Add {
                    record: candidate.clone(),
                    reason: "existing_memory_not_active".to_string(),
                };
            }
            if candidate.status != MemoryStatus::Active {
                return MemoryOp::Noop {
                    memory_id: current.memory_id,
                    reason: "candidate_not_active".to_string(),
                };
            }

            if !same_entity(current, candidate) {
                return MemoryOp::Add {
                    record: candidate.clone(),
                    reason: "new_memory_non_conflicting".to_string(),
                };
            }

            if candidate.confidence + config.supersede_threshold < current.confidence {
                return MemoryOp::Noop {
                    memory_id: current.memory_id,
                    reason: "lower_confidence_candidate".to_string(),
                };
            }

            MemoryOp::Supersede {
                old_memory_id: current.memory_id,
                new_record: candidate.clone(),
                reason: "higher_confidence_or_newer_fact".to_string(),
            }
        }
    }
}

fn same_entity(a: &MemoryRecord, b: &MemoryRecord) -> bool {
    let entity_a = a
        .json_fields
        .get("entity")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty());
    let entity_b = b
        .json_fields
        .get("entity")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|v| !v.is_empty());
    matches!((entity_a, entity_b), (Some(left), Some(right)) if left.eq_ignore_ascii_case(right))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn merge_emits_add_when_missing() {
        let candidate = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Episodic,
            "task succeeded".to_string(),
        );

        let op = merge_candidate(None, &candidate);
        match op {
            MemoryOp::Add { record, .. } => assert_eq!(record.text, "task succeeded"),
            _ => panic!("expected add"),
        }
    }

    #[test]
    fn merge_avoids_supersede_for_unrelated_text() {
        let mut current = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Episodic,
            "deploy window is saturday".to_string(),
        );
        current.confidence = 0.9;
        current.json_fields = serde_json::json!({"entity":"deploy_window"});

        let mut candidate = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Episodic,
            "pager escalation starts with primary".to_string(),
        );
        candidate.confidence = 0.8;
        candidate.json_fields = serde_json::json!({"entity":"pager_policy"});

        let op = merge_candidate(Some(&current), &candidate);
        match op {
            MemoryOp::Add { reason, .. } => assert_eq!(reason, "new_memory_non_conflicting"),
            _ => panic!("expected add"),
        }
    }

    #[test]
    fn merge_blocks_lower_confidence_supersede() {
        let mut current = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Policy,
            "service owner is team atlas".to_string(),
        );
        current.confidence = 0.95;
        current.json_fields = serde_json::json!({"entity":"service_owner"});

        let mut candidate = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Policy,
            "service owner is team apollo".to_string(),
        );
        candidate.confidence = 0.70;
        candidate.json_fields = serde_json::json!({"entity":"service_owner"});

        let op = merge_candidate(Some(&current), &candidate);
        match op {
            MemoryOp::Noop { reason, .. } => assert_eq!(reason, "lower_confidence_candidate"),
            _ => panic!("expected noop"),
        }
    }

    #[test]
    fn merge_supersedes_same_entity_with_sufficient_confidence() {
        let mut current = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Policy,
            "service owner is team atlas".to_string(),
        );
        current.confidence = 0.70;
        current.json_fields = serde_json::json!({"entity":"service_owner"});

        let mut candidate = MemoryRecord::new(
            Uuid::new_v4(),
            MemoryScope::Session,
            MemoryType::Policy,
            "service owner is team apollo".to_string(),
        );
        candidate.confidence = 0.78;
        candidate.json_fields = serde_json::json!({"entity":"service_owner"});

        let op = merge_candidate(Some(&current), &candidate);
        match op {
            MemoryOp::Supersede { old_memory_id, .. } => {
                assert_eq!(old_memory_id, current.memory_id)
            }
            _ => panic!("expected supersede"),
        }
    }
}
