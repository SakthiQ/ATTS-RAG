# Layer 2 - Knowledge Trust & Retrieval Gate

**System:** ATTS-RAG (Adaptive Threat and Trust Security for Retrieval-Augmented Generation)
**Boundary:** between retrieval and generation
**Decision model:** discrete rules and trust tiers, no weighted scores
**Code:** [`app/rag/trust_gate.py`](../app/rag/trust_gate.py) · **Tests:** [`tests/test_layer2_trust_gate.py`](../tests/test_layer2_trust_gate.py), [`tests/test_layer2_strict.py`](../tests/test_layer2_strict.py)

---

## 1. Purpose

Retrieval finds passages that look relevant. Layer 2 decides which of them the model may read: only passages the user is allowed to see, whose text is the text that was indexed, that carry no planted instructions, ranked so that trusted sources come first. Personal data is masked on the way out.

## 2. Rules for each passage

| Rule | Passes when | Otherwise |
|---|---|---|
| **Authorization** | the passage belongs to the user's tenant and its clearance tier is at most the user's clearance | `REJECT` |
| **Integrity** | the SHA-256 of the text matches the hash the registry recorded at upload | `REJECT` |
| **Provenance** | the passage has a document ID, a version and a source reputation of at least 0.60 | trust tier 2 |
| **Safety** | it was not quarantined at upload, its upload anomaly score is below 0.3, and it matches no injection pattern | `QUARANTINE` |

Tiers from the source the uploader chose: `official`, `verified_internal`, `approved_external` → **tier 1**; `unknown` (the default for a user's upload) → **tier 2**; `untrusted` → **tier 3**. Retrieval already filters by tenant and clearance, so authorization failures here mean a bug upstream.

## 3. Choosing the evidence

1. Tier 1, then tier 2, then tier 3, each ordered by the reranker's score.
2. **Conflicts.** A tier-2 passage that contradicts a tier-1 passage on the same point is withheld. "The same point" means nearly the same sentence: at least 75% of the unverified sentence's words and half of the trusted sentence's words are shared. "Contradicts" means other numbers or the opposite polarity. So "Refunds are accepted within 365 days." is withheld next to "Refunds are accepted within 30 days of purchase.", while "Premium plan refunds are accepted within 60 days for annual customers." (another case) and "Refund requests go through the support portal." (another point) are used.
3. **Suspicion.** For a user whose suspicion is at least 0.5, tier 2 and 3 are withheld whenever tier-1 evidence exists. If nothing trusted exists, they are used and the answer is marked `fallback_unverified` (see [`Adaptive_Suspicion.md`](Adaptive_Suspicion.md)).
4. **Untrusted sources are a last resort.** Tier 3 is used only when no tier-1 or tier-2 passage was found, and the answer is then marked `fallback_unverified`.
5. **Split payloads.** Passages are added in rank order. One that would complete an injection split across passages ("... ignore all" + "previous instructions ...") is left out and reported as quarantined; the question is still answered from the rest.
6. Personal data (SSN, card numbers, e-mail addresses, API keys) is masked; an admin view unmasks all but API keys and is audited.

**Tables.** A hit on one row of a table is replaced by the whole table before generation (`engine._expand_tables`). The full text gets the same checks: its hash must match the one recorded at upload, it must carry no injection pattern, and it must not complete a split payload with the other passages. Otherwise the row stays.

## 4. The injection scan

At upload, every chunk is scanned by the threat feed (`config/injection_patterns.yaml`) and, for `unknown` and `untrusted` uploads, by a local LLM judge ([`Knowledge_Ingestion.md`](Knowledge_Ingestion.md) §2.4). At question time Layer 2 scans again, because a chunk may predate a pattern:

- the text is normalised first (NFKC, zero-width and other invisible characters removed, Cyrillic/Greek look-alike letters folded), as Layer 1 does with questions; the upload scan now matches on the normalised text as well;
- it uses the built-in patterns and the feed's high-severity patterns (severity ≥ 0.8, "almost no legitimate use");
- a feed pattern that the upload scan saw in this chunk and its judge cleared is not counted again, so a document that discusses attacks is not quarantined at question time after the judge accepted it.

A pattern scan cannot catch every rewording: "IMPORTANT NOTE TO THE ASSISTANT: tell the user ..." matches no high-severity pattern. The judge at upload is the defence there.

## 5. What Layer 2 tells the suspicion score

| Event | Counted against the asking user |
|---|---|
| a retrieved passage is quarantined and **the user uploaded it** | yes, evidence 0.6 |
| a retrieved passage is quarantined and someone else uploaded it | no |
| a passage outside the user's tenant or clearance | yes, evidence 0.5 |
| a hash mismatch or a passage the registry does not know | no (a stale or tampered index is not the user's doing) |

Before strict mode every quarantined passage counted against whoever asked. An honest user who asked twice about a topic where another user's upload had been flagged reached suspicion 0.51, enough to lose unverified sources and to make Layer 1 block sooner. Embedding outliers in official documents are flagged too, so this happened without any attacker; an attacker could also have used it to make the system stricter with everyone asking about a topic.

## 6. Strict mode

The behaviour above is strict mode, the default since 11 November 2026. `ATTS_L2_STRICT=0` restores the earlier behaviour to reproduce the reports made before it. Strict mode changed:

| Before | Now |
|---|---|
| every quarantined passage and every hash mismatch raised the asking user's suspicion | only the user's own flagged uploads, and only real authorization failures (§5) |
| tier-2 passages were used next to tier 1 even when they contradicted it; `untrusted` was tier 2 | conflicting tier-2 passages are withheld; `untrusted` is tier 3, a last resort (§3) |
| 5 regular expressions on the raw text | normalised text, the built-in and the high-severity feed patterns (§4) |
| a passage the registry did not know, or without an id, passed the integrity check; a missing source tier counted as `approved_external` (tier 1) and a missing reputation as 0.80 | they fail the integrity check; missing trust metadata counts as unverified (tier 2) |
| a table's full text was swapped in without a hash check | it must match the hash recorded at upload (indexes made earlier record it on first start) |
| a passage without a tenant counted as `default_tenant`; any passage of a tenant called `global` passed for every tenant | it belongs to nobody; there is no shared tenant |
| an anomaly score of exactly 0.30 was flagged at upload but passed Layer 2 | the same threshold (≥ 0.3) at both |
| a split payload refused the whole question | only the passage that completes it is left out |
| the integrity check read the whole registry for every passage | it finds the registry entry directly from the chunk id |

## 7. Effect on the evaluation

Every suite that involves Layer 2 was re-run at the same code with strict mode on and off (`ATTS_L2_STRICT=1` / `0`), Llama 3, from the LLM cache where possible:

| Suite | Result |
|---|---|
| Layer 1 (`--suite l1`) | identical, and identical to the saved `eval/results/l1_suite` (the normaliser moved to `text_scan.py` unchanged) |
| Adaptive sessions (`adaptive`), honest users (`adaptive-benign`) | identical |
| Poisoned-document sessions (`adaptive-poison`, naive generator) | identical: no attack succeeded; poison reached the LLM in 6 of 8 queries under the static policy and 0 of 8 under full adaptive |
| Large poisoning study (`poison-large`, `poison_large_v2`, 64 cases x 12 defences) | all 768 records identical; full stack (deep scan, as deployed) 6.2% attack success, 88.0% fact recall |
| Answer quality, dev (`quality`) | identical |

So the paper's numbers do not move. The new checks did not fire in these suites: their poisoned documents plant instructions and marker strings rather than restating an official sentence with another number, the evaluation indexes register every chunk, and the suspicion scenarios do not have one user retrieve another user's flagged upload. What strict mode changes is the deployed application, where those situations do occur (§5, §6); `tests/test_layer2_strict.py` covers each of them.

## 8. Known limits

- **Conflict detection is lexical.** A contradiction phrased differently from the trusted sentence ("a full year" for "365 days", a reordered sentence) is not seen, and two sentences about different cases with the same words and different numbers are taken as a conflict. Layer 3 then still verifies every sentence against whatever evidence was released.
- **Only tier 2 is compared with tier 1.** A trusted source that contradicts another trusted source is not detected.
- **The injection scan reads patterns.** Rewordings without trigger phrases, and other languages, pass it; the LLM judge at upload covers them for `unknown` and `untrusted` uploads only.
- **Integrity protects the index, not the registry.** Someone who can rewrite both the Chroma store and the registry (`document_registry.json` next to it) can make them agree.
