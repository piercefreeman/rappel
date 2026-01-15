# Optimization Scratch

Date: 2026-01-15

## Change 13: Inbox updated_at + start claim lease + inbox compaction hooks
- Added workflow_instances.inbox_updated_at for cache invalidation.
- Reclaim start claims after start_claim_timeout_ms.
- Added optional inbox compaction loop (disabled by default).

### Benchmark results (post inbox_updated_at + reclaim)
- queue-noop, 10k instances, 10s, 4 workers: 25003 actions / 10.04s (~2491.2 actions/sec)
- queue-noop, 50k instances, 10s, 4 workers: 23404 actions / 10.06s (~2327.5 actions/sec)
