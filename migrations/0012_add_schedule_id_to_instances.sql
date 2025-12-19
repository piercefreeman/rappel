-- Migration: Add schedule_id to workflow_instances for schedule tracking
ALTER TABLE workflow_instances
ADD COLUMN schedule_id UUID REFERENCES workflow_schedules(id);

CREATE INDEX idx_workflow_instances_schedule_id
    ON workflow_instances(schedule_id);
