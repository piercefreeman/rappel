-- Add indexes to speed up various slow queries

-- 1. Index for fail_instances_with_exhausted_actions query
-- Finds actions with terminal failure states to propagate to workflow instance
CREATE INDEX IF NOT EXISTS idx_action_queue_exhausted
ON action_queue (instance_id)
WHERE status IN ('exhausted', 'timed_out');

-- 2. Index for find_unstarted_instances query
-- Finds running workflows with next_action_seq=0 that need their first action created
CREATE INDEX IF NOT EXISTS idx_workflow_instances_unstarted
ON workflow_instances (created_at)
WHERE status = 'running' AND next_action_seq = 0;

-- 3. Index for garbage_collect_instances query
-- Finds completed/failed workflows older than retention period for cleanup
CREATE INDEX IF NOT EXISTS idx_workflow_instances_gc
ON workflow_instances (completed_at)
WHERE status IN ('completed', 'failed') AND completed_at IS NOT NULL;
