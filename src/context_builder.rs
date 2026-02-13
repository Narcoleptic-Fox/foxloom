use std::collections::BTreeMap;

use uuid::Uuid;

use crate::{MemoryScope, MemoryType};

#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub max_words: usize,
    pub reserve_words: usize,
}

#[derive(Debug, Clone)]
pub struct ContextBuildConfig {
    pub include_headers: bool,
    pub include_why: bool,
}

#[derive(Debug, Clone)]
pub struct ContextItem {
    pub memory_id: Uuid,
    pub text: String,
    pub memory_type: MemoryType,
    pub scope: MemoryScope,
    pub similarity: f32,
    pub confidence: f32,
    pub importance: f32,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct BuiltContext {
    pub prompt_prefix: String,
    pub included: Vec<ContextItem>,
    pub used_words: usize,
    pub dropped_count: usize,
    pub drop_reasons: BTreeMap<String, usize>,
}

pub trait BudgetEstimator {
    fn estimate_words(&self, text: &str) -> usize;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WordBudgetEstimator;

impl BudgetEstimator for WordBudgetEstimator {
    fn estimate_words(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

pub fn build_active_context(
    candidates: &[ContextItem],
    budget: &ContextBudget,
    cfg: &ContextBuildConfig,
) -> BuiltContext {
    build_active_context_with_estimator(candidates, budget, cfg, &WordBudgetEstimator)
}

pub fn build_active_context_with_estimator(
    candidates: &[ContextItem],
    budget: &ContextBudget,
    cfg: &ContextBuildConfig,
    estimator: &dyn BudgetEstimator,
) -> BuiltContext {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| {
        scope_rank(&left.scope)
            .cmp(&scope_rank(&right.scope))
            .then_with(|| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.memory_id.as_u128().cmp(&right.memory_id.as_u128()))
    });

    let available = budget.max_words.saturating_sub(budget.reserve_words);
    let mut used_words = 0usize;
    let mut included = Vec::new();
    let mut lines = Vec::new();
    let mut last_header: Option<String> = None;
    let mut drop_reasons = BTreeMap::new();

    for item in ordered {
        let mut line = format!("- {}", item.text);
        if cfg.include_why {
            line.push_str(&format!(
                " (score={:.3}, sim={:.3}, imp={:.3}, conf={:.3}, scope={}, src={})",
                item.score,
                item.similarity,
                item.importance,
                item.confidence,
                scope_label(&item.scope),
                item.source
            ));
        }
        let line_words = estimator.estimate_words(&line);
        if used_words.saturating_add(line_words) > available {
            *drop_reasons.entry("budget".to_string()).or_insert(0) += 1;
            continue;
        }

        let header = section_header(&item.scope, &item.memory_type);
        if cfg.include_headers && last_header.as_deref() != Some(header) {
            lines.push(format!("{}:", header));
            last_header = Some(header.to_string());
        }

        lines.push(line);
        used_words += line_words;
        included.push(item);
    }

    BuiltContext {
        prompt_prefix: lines.join("\n"),
        dropped_count: candidates.len().saturating_sub(included.len()),
        included,
        used_words,
        drop_reasons,
    }
}

fn scope_rank(scope: &MemoryScope) -> usize {
    match scope {
        MemoryScope::Workspace => 0,
        MemoryScope::Session => 1,
        MemoryScope::User => 2,
        MemoryScope::Global => 3,
    }
}

fn section_header(scope: &MemoryScope, memory_type: &MemoryType) -> &'static str {
    match memory_type {
        MemoryType::ArtifactSummary => "Artifact Notes",
        MemoryType::Episodic => "Recent Episodic",
        _ => match scope {
            MemoryScope::Workspace => "Workspace Policies",
            MemoryScope::Session => "Session Context",
            MemoryScope::User => "User Preferences",
            MemoryScope::Global => "Global Defaults",
        },
    }
}

fn scope_label(scope: &MemoryScope) -> &'static str {
    match scope {
        MemoryScope::User => "user",
        MemoryScope::Session => "session",
        MemoryScope::Workspace => "workspace",
        MemoryScope::Global => "global",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(
        scope: MemoryScope,
        memory_type: MemoryType,
        score: f32,
        text: &str,
        memory_id: Uuid,
    ) -> ContextItem {
        ContextItem {
            memory_id,
            text: text.to_string(),
            memory_type,
            scope,
            similarity: 0.8,
            confidence: 0.7,
            importance: 0.6,
            score,
            source: "vector_search".to_string(),
        }
    }

    #[test]
    fn deterministic_ordering_with_uuid_tie_breaker() {
        let low = Uuid::from_u128(1);
        let high = Uuid::from_u128(2);
        let out = build_active_context(
            &[
                item(
                    MemoryScope::Session,
                    MemoryType::Policy,
                    1.0,
                    "second",
                    high,
                ),
                item(MemoryScope::Session, MemoryType::Policy, 1.0, "first", low),
            ],
            &ContextBudget {
                max_words: 50,
                reserve_words: 0,
            },
            &ContextBuildConfig {
                include_headers: true,
                include_why: false,
            },
        );
        assert_eq!(out.included.len(), 2);
        assert_eq!(out.included[0].memory_id, low);
        assert_eq!(out.included[1].memory_id, high);
    }

    #[test]
    fn respects_budget_hard_cap() {
        let out = build_active_context(
            &[
                item(
                    MemoryScope::Workspace,
                    MemoryType::Policy,
                    1.0,
                    "one two three four",
                    Uuid::new_v4(),
                ),
                item(
                    MemoryScope::Session,
                    MemoryType::Episodic,
                    0.9,
                    "five six seven eight",
                    Uuid::new_v4(),
                ),
            ],
            &ContextBudget {
                max_words: 5,
                reserve_words: 0,
            },
            &ContextBuildConfig {
                include_headers: false,
                include_why: false,
            },
        );
        assert_eq!(out.included.len(), 1);
        assert!(out.used_words <= 5);
        assert_eq!(out.dropped_count, 1);
    }

    #[test]
    fn empty_candidates_yield_empty_context() {
        let out = build_active_context(
            &[],
            &ContextBudget {
                max_words: 20,
                reserve_words: 0,
            },
            &ContextBuildConfig {
                include_headers: true,
                include_why: true,
            },
        );
        assert!(out.prompt_prefix.is_empty());
        assert!(out.included.is_empty());
        assert_eq!(out.used_words, 0);
    }
}
