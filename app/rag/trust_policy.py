"""Shared trust policy: source tiers and poisoning-scan thresholds.

Used at ingestion (tier assignment, quarantine) and by Layer 2 trust scoring at query time.
"""

# Source reputation S(d), assigned at upload. Every tier except DEFAULT_TIER requires the admin token.
SOURCE_TIERS = {
    "official": 1.00,
    "verified_internal": 0.90,
    "approved_external": 0.80,
    "unknown": 0.40,
    "untrusted": 0.10,
}
DEFAULT_TIER = "unknown"

# Thresholds on a chunk's anomaly score A(d) in [0, 1]
QUARANTINE_THRESHOLD = 0.8  # At or above: recorded for review, never indexed
FLAG_THRESHOLD = 0.3  # At or above: indexed, but flagged for Layer 2
