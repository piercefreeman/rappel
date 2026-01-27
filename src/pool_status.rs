//! In-memory ring buffer for pool-level time-series metrics.
//!
//! Stores up to 24 hours of 1-minute resolution data (~25 KB per runner).
//! Binary format: `[u32 count][N x 18-byte entries]`

use std::collections::VecDeque;

use serde::Serialize;

/// Maximum entries: 24h * 60 min = 1440
const MAX_ENTRIES: usize = 1440;

/// Size of one serialized entry in bytes.
const ENTRY_SIZE: usize = 18;

/// A single time-series data point (18 bytes serialized).
#[derive(Debug, Clone, Copy)]
pub struct TimeSeriesEntry {
    /// Unix timestamp in seconds
    pub timestamp_secs: i64,
    /// Actions per second at this point
    pub actions_per_sec: f32,
    /// Number of active workers
    pub active_workers: u16,
    /// Average instance duration in seconds
    pub avg_instance_duration_secs: f32,
}

/// Ring buffer of time-series entries, max 1440 (24 hours at 1-minute resolution).
#[derive(Debug, Clone)]
pub struct PoolTimeSeries {
    entries: VecDeque<TimeSeriesEntry>,
}

impl PoolTimeSeries {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    /// Append an entry, dropping the oldest if over capacity.
    pub fn push(&mut self, entry: TimeSeriesEntry) {
        if self.entries.len() >= MAX_ENTRIES {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Encode to binary: `[u32 count][N x 18-byte entries]`
    ///
    /// Each entry is:
    /// - i64 timestamp_secs (8 bytes, little-endian)
    /// - f32 actions_per_sec (4 bytes, little-endian)
    /// - u16 active_workers (2 bytes, little-endian)
    /// - f32 avg_instance_duration_secs (4 bytes, little-endian)
    pub fn encode(&self) -> Vec<u8> {
        let count = self.entries.len() as u32;
        let mut buf = Vec::with_capacity(4 + self.entries.len() * ENTRY_SIZE);
        buf.extend_from_slice(&count.to_le_bytes());
        for entry in &self.entries {
            buf.extend_from_slice(&entry.timestamp_secs.to_le_bytes());
            buf.extend_from_slice(&entry.actions_per_sec.to_le_bytes());
            buf.extend_from_slice(&entry.active_workers.to_le_bytes());
            buf.extend_from_slice(&entry.avg_instance_duration_secs.to_le_bytes());
        }
        buf
    }

    /// Decode from binary. Returns `None` if the bytes are malformed.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let count = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
        if count > MAX_ENTRIES {
            return None;
        }
        let expected_len = 4 + count * ENTRY_SIZE;
        if bytes.len() < expected_len {
            return None;
        }
        let mut entries = VecDeque::with_capacity(count);
        for i in 0..count {
            let offset = 4 + i * ENTRY_SIZE;
            let timestamp_secs =
                i64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?);
            let actions_per_sec =
                f32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().ok()?);
            let active_workers =
                u16::from_le_bytes(bytes[offset + 12..offset + 14].try_into().ok()?);
            let avg_instance_duration_secs =
                f32::from_le_bytes(bytes[offset + 14..offset + 18].try_into().ok()?);
            entries.push_back(TimeSeriesEntry {
                timestamp_secs,
                actions_per_sec,
                active_workers,
                avg_instance_duration_secs,
            });
        }
        Some(Self { entries })
    }

    /// Convert to JSON-friendly entries for the frontend template.
    pub fn to_json_entries(&self) -> Vec<TimeSeriesJsonEntry> {
        self.entries
            .iter()
            .map(|e| TimeSeriesJsonEntry {
                t: e.timestamp_secs,
                aps: e.actions_per_sec,
                w: e.active_workers,
                d: e.avg_instance_duration_secs,
            })
            .collect()
    }
}

impl Default for PoolTimeSeries {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON-serializable time-series entry for the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct TimeSeriesJsonEntry {
    /// Unix timestamp (seconds)
    pub t: i64,
    /// Actions per second
    pub aps: f32,
    /// Active worker count
    pub w: u16,
    /// Average instance duration (seconds)
    pub d: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut ts = PoolTimeSeries::new();
        assert_eq!(ts.len(), 0);

        ts.push(TimeSeriesEntry {
            timestamp_secs: 1000,
            actions_per_sec: 1.5,
            active_workers: 4,
            avg_instance_duration_secs: 10.0,
        });
        assert_eq!(ts.len(), 1);
    }

    #[test]
    fn test_capacity_limit() {
        let mut ts = PoolTimeSeries::new();
        for i in 0..1500 {
            ts.push(TimeSeriesEntry {
                timestamp_secs: i as i64,
                actions_per_sec: 1.0,
                active_workers: 1,
                avg_instance_duration_secs: 0.0,
            });
        }
        assert_eq!(ts.len(), MAX_ENTRIES);
        // Oldest entries should have been dropped; first entry should be 60
        let json = ts.to_json_entries();
        assert_eq!(json[0].t, 60);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut ts = PoolTimeSeries::new();
        ts.push(TimeSeriesEntry {
            timestamp_secs: 1700000000,
            actions_per_sec: 3.14,
            active_workers: 8,
            avg_instance_duration_secs: 42.5,
        });
        ts.push(TimeSeriesEntry {
            timestamp_secs: 1700000060,
            actions_per_sec: 2.71,
            active_workers: 6,
            avg_instance_duration_secs: 38.0,
        });

        let bytes = ts.encode();
        let decoded = PoolTimeSeries::decode(&bytes).expect("decode should succeed");
        assert_eq!(decoded.len(), 2);

        let json = decoded.to_json_entries();
        assert_eq!(json[0].t, 1700000000);
        assert!((json[0].aps - 3.14).abs() < 0.001);
        assert_eq!(json[0].w, 8);
        assert!((json[0].d - 42.5).abs() < 0.01);

        assert_eq!(json[1].t, 1700000060);
        assert_eq!(json[1].w, 6);
    }

    #[test]
    fn test_decode_empty() {
        let ts = PoolTimeSeries::new();
        let bytes = ts.encode();
        let decoded = PoolTimeSeries::decode(&bytes).expect("decode should succeed");
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_decode_invalid_short() {
        assert!(PoolTimeSeries::decode(&[0, 1]).is_none());
    }

    #[test]
    fn test_decode_invalid_truncated() {
        // count says 1 entry but no entry data
        let bytes = 1u32.to_le_bytes();
        assert!(PoolTimeSeries::decode(&bytes).is_none());
    }

    #[test]
    fn test_to_json_entries() {
        let mut ts = PoolTimeSeries::new();
        ts.push(TimeSeriesEntry {
            timestamp_secs: 100,
            actions_per_sec: 5.0,
            active_workers: 2,
            avg_instance_duration_secs: 20.0,
        });
        let json = ts.to_json_entries();
        assert_eq!(json.len(), 1);
        assert_eq!(json[0].t, 100);
        assert!((json[0].aps - 5.0).abs() < f32::EPSILON);
        assert_eq!(json[0].w, 2);
    }
}
