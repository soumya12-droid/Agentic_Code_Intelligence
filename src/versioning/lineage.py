"""Phase 5 (snippet lineage for evolutionary retrieval) — see docs/PROJECT_PLAN.md section 3.3/7.

Lineage is implemented in index_store.py, next to the metadata it depends on: every snippet row has a
lineage_id (a new snippet starts its own; an edit, a revert or a move inherits its predecessor's),
VersionedIndex.search(as_of="all") groups results by lineage, history() returns a function's timeline,
and backfill_lineage() assigns ids to a store built before they existed. The pairwise old-to-new links
written in Phase 4 remain in the `lineage` table as an audit trail.
"""
