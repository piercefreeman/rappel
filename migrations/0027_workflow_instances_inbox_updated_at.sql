-- Track latest inbox update time for cache invalidation.

ALTER TABLE workflow_instances
ADD COLUMN IF NOT EXISTS inbox_updated_at TIMESTAMPTZ DEFAULT NOW();

UPDATE workflow_instances
SET inbox_updated_at = COALESCE(inbox_updated_at, created_at);

ALTER TABLE workflow_instances
ALTER COLUMN inbox_updated_at SET NOT NULL;
