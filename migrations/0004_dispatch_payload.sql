ALTER TABLE daemon_action_ledger
    ADD COLUMN dispatch_payload BYTEA;

UPDATE daemon_action_ledger
SET dispatch_payload = COALESCE(dispatch_payload, ''::BYTEA);

ALTER TABLE daemon_action_ledger
    ALTER COLUMN dispatch_payload SET NOT NULL;

ALTER TABLE daemon_action_ledger
    DROP COLUMN kwargs_payload;
