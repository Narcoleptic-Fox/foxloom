# Architecture

## Module Layout

- `src/model.rs`: canonical types and operations.
- `src/scoring.rs`: recency decay scoring.
- `src/context_builder.rs`: deterministic budgeted context formatting.
- `src/adapter.rs`: vector adapter trait contract.
- `src/foxstash_adapter.rs`: `foxstash-core` adapter implementation.
- `src/lib.rs`: public exports and merge entrypoint.

## High-Level Components

```mermaid
graph TD
    A[Application / Orchestrator] --> B[foxloom::model]
    A --> C[foxloom::merge_candidate]
    A --> D[foxloom::scoring]
    A --> E[foxloom::context_builder]
    A --> F[foxloom::adapter trait]
    F --> G[foxloom::FoxstashCoreAdapter]
    G --> H[foxstash-core HNSWIndex]
```

## Typical Runtime Data Path

```mermaid
sequenceDiagram
    participant App as Engine/App
    participant Adapter as FoxstashCoreAdapter
    participant Index as HNSWIndex
    participant Scoring as scoring
    participant Builder as context_builder

    App->>Adapter: similarity_search(query, top_k)
    Adapter->>Index: search(embedding, fetch_k)
    Index-->>Adapter: SearchResult[]
    Adapter-->>App: MemoryRecord[]
    App->>Scoring: decayed_importance(...)
    Scoring-->>App: effective_importance
    App->>Builder: build_active_context(...)
    Builder-->>App: BuiltContext
```

## Determinism Guarantees

- Context ordering is stable by:
  - scope precedence (`workspace -> session -> user -> global`)
  - descending score
  - UUID tie-break (`memory_id` as u128)
- Adapter retrieval has deterministic tie handling with id-based ordering.
- Retrieval widening is bounded (`8x top_k` cap) to avoid runaway fetch loops.
