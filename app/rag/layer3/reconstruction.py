from typing import List, Tuple

from app.rag.layer3.contracts import ClaimObject, ClaimVerificationResult
from app.rag.layer3.generation import Draft


class VerifiedAnswerReconstructor:
    """Rebuilds the released answer from the draft, keeping only sentences that passed verification.

    The draft's structure (headings, bullets, paragraphs) is preserved so the answer stays readable.
    Each kept sentence is cited with the evidence passage that was actually found to support it.
    """

    def reconstruct(
        self,
        draft: Draft,
        verification_results: List[ClaimVerificationResult],
    ) -> Tuple[str, List[ClaimObject], List[ClaimObject]]:
        by_id = {r.claim_id: r for r in verification_results}
        verified: List[ClaimObject] = []
        failed: List[ClaimObject] = []
        claim_iter = iter(draft.claims)

        rendered: List[Tuple[str, bool]] = []  # (text, is_heading)
        for line in draft.lines:
            kept: List[str] = []
            for sentence in line.sentences:
                if not sentence.checkable:
                    kept.append(sentence.text)
                    continue
                claim = next(claim_iter)
                result = by_id.get(claim.claim_id)
                if result is not None and result.status == "ENTAILMENT":
                    support = result.supporting_evidence_id or (claim.evidence_ids[0] if claim.evidence_ids else None)
                    ids = [support] if support else []
                    verified.append(claim.model_copy(update={"evidence_ids": ids}))
                    kept.append(sentence.text + (f" [{support}]" if support else ""))
                else:
                    failed.append(claim)
            if kept:
                is_heading = line.prefix.startswith("#")
                rendered.append((line.prefix + " ".join(kept), is_heading))

        # A heading with nothing verified beneath it is noise
        out: List[str] = []
        for i, (text, is_heading) in enumerate(rendered):
            if is_heading and (i + 1 == len(rendered) or rendered[i + 1][1]):
                continue
            out.append(text)

        if not verified:
            return "", [], failed
        answer = ""
        for i, text in enumerate(out):
            if i:
                both_bullets = text.startswith("- ") and out[i - 1].startswith("- ")
                answer += "\n" if both_bullets else "\n\n"
            answer += text
        return answer, verified, failed
