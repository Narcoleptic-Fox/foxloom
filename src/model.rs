use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    User,
    Session,
    Workspace,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Profile,
    Episodic,
    Policy,
    ArtifactSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Superseded,
    Quarantined,
    Deleted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRecord {
    pub memory_id: Uuid,
    pub workspace_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub scope: MemoryScope,
    pub memory_type: MemoryType,
    pub text: String,
    pub json_fields: Value,
    pub embedding_ref: Option<String>,
    pub confidence: f32,
    pub importance: f32,
    pub decay_half_life_hours: Option<u32>,
    pub status: MemoryStatus,
    pub source_run_id: Option<Uuid>,
    pub updated_at: DateTime<Utc>,
}

impl MemoryRecord {
    pub fn new(memory_id: Uuid, scope: MemoryScope, memory_type: MemoryType, text: String) -> Self {
        Self {
            memory_id,
            workspace_id: None,
            user_id: None,
            session_id: None,
            scope,
            memory_type,
            text,
            json_fields: Value::Null,
            embedding_ref: None,
            confidence: 0.7,
            importance: 0.5,
            decay_half_life_hours: None,
            status: MemoryStatus::Active,
            source_run_id: None,
            updated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MemoryOp {
    Add {
        record: MemoryRecord,
        reason: String,
    },
    Update {
        memory_id: Uuid,
        patch: Value,
        reason: String,
    },
    Supersede {
        old_memory_id: Uuid,
        new_record: MemoryRecord,
        reason: String,
    },
    Delete {
        memory_id: Uuid,
        reason: String,
    },
    Noop {
        memory_id: Uuid,
        reason: String,
    },
}
