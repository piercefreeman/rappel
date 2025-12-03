-- Add module_name to workflow_versions
-- This stores the Python module path where the workflow is defined
-- (e.g., "myapp.workflows.order_processing")

ALTER TABLE workflow_versions
ADD COLUMN module_name TEXT NOT NULL DEFAULT 'default';
