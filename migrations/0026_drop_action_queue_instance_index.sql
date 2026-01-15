-- Drop redundant action_queue instance index; covered by UNIQUE (instance_id, action_seq)
DROP INDEX IF EXISTS idx_action_queue_instance;
