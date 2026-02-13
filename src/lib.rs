pub mod adapter;
pub mod model;

pub use adapter::FoxstashAdapter;
pub use model::{MemoryOp, MemoryRecord, MemoryScope, MemoryStatus, MemoryType};

pub fn merge_candidate(existing: Option<&MemoryRecord>, candidate: &MemoryRecord) -> MemoryOp {
    match existing {
        None => MemoryOp::Add {
            record: candidate.clone(),
            reason: "no_similar_memory".to_string(),
        },
        Some(current) => {
            if current.text == candidate.text {
                MemoryOp::Noop {
                    memory_id: current.memory_id,
                    reason: "duplicate_text".to_string(),
                }
            } else {
                MemoryOp::Supersede {
                    old_memory_id: current.memory_id,
                    new_record: candidate.clone(),
                    reason: "higher_confidence_or_newer_fact".to_string(),
                }
            }
        }
    }
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
}
