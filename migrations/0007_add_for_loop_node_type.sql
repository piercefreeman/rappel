-- Add 'for_loop' to valid node_type values
--
-- This enables for-loop control nodes to be processed by the runner
-- similar to barriers, but with their own iteration logic.

-- Drop the existing constraint
ALTER TABLE action_queue DROP CONSTRAINT IF EXISTS action_queue_node_type_check;

-- Add updated constraint with for_loop
ALTER TABLE action_queue ADD CONSTRAINT action_queue_node_type_check
    CHECK (node_type IN ('action', 'barrier', 'for_loop'));
