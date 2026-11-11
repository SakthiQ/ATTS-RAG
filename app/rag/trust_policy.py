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

# Layer 2 strict mode (default; ATTS_L2_STRICT=0 restores the earlier behaviour to reproduce the reports made before it):
# - only a user's OWN flagged uploads count against them when a question retrieves them;
# - a hash mismatch is not counted as the user's authorization failure;
# - an unverified passage that contradicts a trusted one on the same point is withheld, and "untrusted" sources are
#   used only when nothing better was found;
# - the query-time injection scan reads normalised text with the high-severity patterns of the threat feed;
# - missing trust metadata counts as untrusted, and passages the registry does not know fail the integrity check.
import os as _os
L2_STRICT = _os.getenv("ATTS_L2_STRICT", "1") == "1"
QUERY_SCAN_MIN_SEVERITY = 0.8   # threat-feed patterns with "almost no legitimate use" (see config/injection_patterns.yaml)
