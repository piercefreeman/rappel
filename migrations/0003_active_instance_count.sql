-- Track number of workflow instances currently owned by this pool runner.
ALTER TABLE worker_status ADD COLUMN active_instance_count INT NOT NULL DEFAULT 0;
