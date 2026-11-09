import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from app.rag.layer3.contracts import ChartSpec, ClaimObject

EVIDENCE_CHAR_BUDGET = 7000  # ~1800 tokens: what a CPU-only local LLM can read in reasonable time
_CITE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")
_LEADING_CITES = re.compile(r"^((?:\s*\[E\d+(?:\s*,\s*E\d+)*\])+)\s*(.*)$", re.S)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_TAG = r"\[E\d+(?:\s*,\s*E\d+)*\]"
_VERBS = r"(?:highlights?|states?|notes?|mentions?|describes?|explains?|shows?|says?|indicates?|reports?|emphasi[sz]es?)"
# Phrases that only make sense with the tag attached ("According to [E1], ...", "[E3] highlights that ...")
_ATTRIBUTION = [
    re.compile(r"(?:^|,\s*|\s+)(?:according to|per)\s+" + _TAG + r"\s*,?\s*", re.I),
    re.compile(r",?\s*\bas\s+(?:mentioned|described|stated|noted|shown|explained|highlighted|outlined)\s+in\s+" + _TAG, re.I),
    re.compile(_TAG + r"\s+(?:also\s+)?" + _VERBS + r"\s+(?:that\s+)?", re.I),
]
# Statements about the document rather than about the facts ("The evidence states that ...", "..., as stated in the
# handbook"). They add words no evidence sentence contains, so the verifier would reject an otherwise supported claim.
_META_LEAD = [
    re.compile(r"^(?:this is\s+)?according to\s+(?:the\s+)?[^,]{1,80},\s*which\s+(?:states?|says?|shows?|notes?|specifies)\s+that\s+", re.I),
    re.compile(r"^(?:the\s+)?(?:evidence|documents?|passages?|policy|handbook|text|sources?|report|schedule)\s+(?:also\s+)?"
               r"(?:states?|says?|notes?|indicates?|mentions?|specifies|shows?|confirms?)\s+that\s+", re.I),
    re.compile(r"^(?:as\s+)?(?:stated|mentioned|described|specified|outlined|noted|shown)\s+in\s+(?:the\s+)?[^,]{1,60},\s*", re.I),
    # "According to the Acme Corp Employee Handbook, employees ..." (a source named in words, without a tag)
    re.compile(r"^according to\s+(?:the\s+|our\s+)?[^,]{1,80},\s*", re.I),
]
# A document's file name in the sentence ("(acme_support_plans.md), customer data is kept ...", left over once
# "According to [E1]" is removed, or "... (source: handbook.pdf)"): words no evidence sentence contains
_FILENAME = re.compile(r"\s*(?:\b(?:according to|per|in|from)\s+(?:the\s+)?)?\(?\b(?:source:\s*)?[\w\-]+\.(?:md|pdf|txt|docx?|csv|xlsx?|html?)\b\)?\s*(?:,\s*)?", re.I)
_META_TAIL = re.compile(r",?\s*\b(?:as|which is)\s+(?:is\s+)?(?:stated|mentioned|described|specified|outlined|noted|shown|detailed)"
                        r"\s+in\s+(?:the\s+)?[A-Za-z][\w'’\- ]{0,50}?(?=[.!?;]*$)", re.I)
# Verifier v3 (reviewer point 4; default, ATTS_L3_V3=0 restores v2): longer document names in the trailing
# attribution ("as specified in the Paid Time Off section of the Acme Corp Employee Handbook"), and two more lead-ins
_V3 = os.getenv("ATTS_L3_V3", "1") == "1"
if _V3:
    _META_TAIL = re.compile(r",?\s*\b(?:as|which is)\s+(?:is\s+)?(?:stated|mentioned|described|specified|outlined|noted|shown|detailed)"
                            r"\s+in\s+(?:the\s+)?[A-Za-z][^.!?;,]{0,90}?(?=[.!?;]*$)", re.I)
    _META_LEAD = _META_LEAD + [
        re.compile(r"^(?:this is\s+)?(?:stated|mentioned|described|specified|outlined|noted|shown)\s+in\s+(?:the\s+)?[^,]{1,80},\s*"
                   r"which\s+(?:states?|says?|mentions?|notes?|specifies)\s+that\s+", re.I),
        re.compile(r"^this means that\s+", re.I),
    ]
_CHART_ANNOUNCE = re.compile(r"^\W*(here is|here are|below is)\b.{0,60}\b(chart|table|graph)", re.I)
_ABSTAIN = re.compile(
    r"\b(does not|do not|doesn't|don't|did not|is not|are not|no)\b.{0,40}\b"
    r"(mention|specif|provide|state|cover|describe|contain|include|discuss|say|available)", re.I)

_DEFAULT_PROMPTS = {
    "system_instruction": (
        "Answer using ONLY the numbered evidence. Write 'INTENT:' (one sentence) then 'ANSWER:'. "
        "End every factual sentence with its evidence number like [E1]."
    ),
    "chart_instruction": (
        'Reply with JSON {"chart": null} or {"chart": {"type","title","x_label","y_label","unit",'
        '"evidence_id","data":[{"label","value"}]}}.'
    ),
}


@dataclass
class Sentence:
    text: str
    evidence_ids: List[str]
    checkable: bool = True


@dataclass
class Line:
    prefix: str  # markdown marker kept from the draft: "- ", "## " or ""
    sentences: List[Sentence] = field(default_factory=list)


@dataclass
class Draft:
    intent: Optional[str]
    lines: List[Line]
    claims: List[ClaimObject]
    evidence_lookup: Dict[str, str]   # "E1" -> passage text
    evidence_to_uid: Dict[str, str]   # "E1" -> chunk_uid (for audit and citations)


def _load_prompts(path: str) -> Dict[str, str]:
    prompts = dict(_DEFAULT_PROMPTS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            for key in prompts:
                if cfg.get(key):
                    prompts[key] = cfg[key]
        except Exception as exc:  # a bad prompt file must not take the gate down
            logger.warning(f"Could not load prompt file {path}: {exc}")
    return prompts


class AnswerGenerator:
    """Asks the LLM for a descriptive, cited answer and turns it into checkable claims.

    The LLM only ever sees short handles (E1, E2, ...). They are mapped back to chunk ids here,
    because small local models cannot copy 64-character hashes reliably.
    """

    strip_meta = True   # switch for ablation studies: drop "as stated in the handbook" style clauses from claims

    @staticmethod
    def _strip_meta(text: str) -> str:
        """Removes attribution clauses that talk about the document instead of stating a fact."""
        out = _FILENAME.sub(" ", text).strip()
        out = re.sub(r"\s+([.,;:!?])", r"\1", out)
        for pattern in _META_LEAD:
            out = pattern.sub("", out, count=1)
        out = _META_TAIL.sub("", out)
        out = out.strip()
        if out and out != text and out[0].islower():
            out = out[0].upper() + out[1:]
        return out if len(out.split()) >= 3 else text   # never reduce a sentence to a fragment

    def __init__(self, llm_invoker: Optional[Callable[[str], str]] = None,
                 prompt_path: str = "prompts/enterprise_rag_v1.yaml",
                 json_invoker: Optional[Callable[[str], str]] = None):
        self.llm_invoker = llm_invoker
        self.json_invoker = json_invoker or llm_invoker
        self.prompts = _load_prompts(prompt_path)

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def label_evidence(evidence_package: List[Dict[str, Any]],
                       char_budget: int = EVIDENCE_CHAR_BUDGET) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        """Returns (prompt block, {E#: text}, {E#: chunk_uid}) within the character budget."""
        blocks, lookup, to_uid = [], {}, {}
        used = 0
        for item in evidence_package:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            if lookup and used + len(content) > char_budget:
                break
            label = f"E{len(lookup) + 1}"
            meta = item.get("metadata", {})
            source = meta.get("source") or meta.get("filename") or "document"
            page = meta.get("page")
            where = f"{source}, p.{page}" if page not in (None, "", "N/A") else source
            blocks.append(f"[{label}] (source: {where})\n{content}")
            lookup[label] = content
            to_uid[label] = meta.get("chunk_uid") or label
            used += len(content)
        return "\n\n".join(blocks), lookup, to_uid

    # ------------------------------------------------------------------ answer
    def generate(self, question: str, evidence_package: List[Dict[str, Any]]) -> Draft:
        block, lookup, to_uid = self.label_evidence(evidence_package)
        if not lookup:
            return Draft(None, [], [], lookup, to_uid)

        prompt = (
            f"{self.prompts['system_instruction']}\n\n"
            f"Evidence:\n{block}\n\n"
            f"Question: {question}"
        )
        raw = self._normalize_labels(self._call(prompt))
        if not re.search(r"\bANSWER\s*:\s*\S", raw, re.I):
            # The model stopped after the intent line; ask it to continue with the answer itself
            head = f"{raw.strip()}\n\nANSWER:\n"
            raw = head + self._normalize_labels(self._call(prompt + "\n\n" + head))
        intent, lines = self.parse(raw, set(lookup), self.strip_meta)
        claims, order = [], 0
        for line in lines:
            for sentence in line.sentences:
                if sentence.checkable:
                    order += 1
                    claims.append(ClaimObject(claim_id=f"C{order}", order=order,
                                              text=sentence.text, evidence_ids=sentence.evidence_ids))
        return Draft(intent, lines, claims, lookup, to_uid)

    @staticmethod
    def _normalize_labels(text: str) -> str:
        """Turns '**INTENT**:' / '## Answer:' style labels into plain 'INTENT:' / 'ANSWER:'."""
        return re.sub(r"[*_#]*\b(INTENT|ANSWER)\b[*_]*\s*:[*_]*", lambda m: m.group(1).upper() + ":", text, flags=re.I)

    def _call(self, prompt: str) -> str:
        if self.llm_invoker is None:
            return ""
        raw = self.llm_invoker(prompt)
        return str(getattr(raw, "content", raw) or "")

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def parse(raw: str, valid_ids: set, strip_meta: bool = True) -> Tuple[Optional[str], List[Line]]:
        """Splits 'INTENT: ... ANSWER: ...' into the intent and the answer lines.

        The prompt ends with 'INTENT:', so the model's reply usually starts mid-label.
        """
        text = AnswerGenerator._normalize_labels(raw).strip()
        intent = None
        # The model sometimes repeats its labels, so the answer is whatever follows the LAST "ANSWER:"
        matches = list(re.finditer(r"\bANSWER\s*:\s*", text, re.I))
        if matches:
            m = matches[-1]
            head, body = text[:m.start()], text[m.end():]
            head = re.sub(r"^\s*INTENT\s*:\s*", "", head.split("ANSWER:")[0].strip(), flags=re.I)
            intent = " ".join(head.split()) or None
        else:
            body = re.sub(r"^\s*(INTENT|ANSWER)\s*:\s*", "", text, flags=re.I)
        lines = [
            AnswerGenerator._parse_line(ln, valid_ids, strip_meta) for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("|") and not _CHART_ANNOUNCE.search(ln)
        ]
        return intent, [ln for ln in lines if ln.sentences]

    @staticmethod
    def _parse_line(raw_line: str, valid_ids: set, strip_meta: bool = True) -> Line:
        heading = re.match(r"^\s*(#{1,6}\s+)(.*)$", raw_line)
        if heading:
            return Line(prefix=heading.group(1), sentences=[Sentence(heading.group(2).strip(), [], checkable=False)])
        bullet = _BULLET.match(raw_line)
        prefix = "- " if bullet else ""
        body = raw_line[bullet.end():] if bullet else raw_line.strip()

        sentences: List[Sentence] = []
        for part in re.split(r"(?<=[.!?])\s+(?=[\[A-Z0-9\"'(])", body):
            part = part.strip()
            if not part:
                continue
            lead = _LEADING_CITES.match(part)
            if lead and sentences:  # "Fact. [E1] Next fact." -> the tag belongs to the previous sentence
                for eid in AnswerGenerator._ids(lead.group(1), valid_ids):
                    if eid not in sentences[-1].evidence_ids:
                        sentences[-1].evidence_ids.append(eid)
                part = lead.group(2).strip()
                if not part:
                    continue
            ids = AnswerGenerator._ids(part, valid_ids)
            bare = part
            for pattern in _ATTRIBUTION:
                bare = pattern.sub(lambda m: " " if m.start() else "", bare)
            bare = bare.strip()
            if bare and bare[0].islower() and bare != part:
                bare = bare[0].upper() + bare[1:]
            clean = " ".join(re.sub(r"\*\*|__|`", "", _CITE.sub("", bare)).split())  # emphasis marks confuse NLI
            clean = re.sub(r"\s+([.,;:!?])", r"\1", clean)
            if strip_meta and clean:
                clean = AnswerGenerator._strip_meta(clean)
            if not clean:
                continue
            sentences.append(Sentence(clean, ids))
        # Charts are produced separately, so uncited talk about "the chart" is noise
        sentences = [s for s in sentences if s.evidence_ids or not re.search(r"\b(chart|graph|table)\b", s.text, re.I)]
        for sentence in sentences:
            words = len(sentence.text.split())
            uncited = not sentence.evidence_ids
            structural = uncited and (words < 4 or (sentence.text.endswith(":") and words <= 10))
            abstain = uncited and bool(_ABSTAIN.search(sentence.text))
            sentence.checkable = not (structural or abstain)
        return Line(prefix=prefix, sentences=sentences)

    @staticmethod
    def _ids(fragment: str, valid_ids: set) -> List[str]:
        found: List[str] = []
        for group in _CITE.findall(fragment):
            for eid in re.split(r"\s*,\s*", group):
                if eid in valid_ids and eid not in found:
                    found.append(eid)
        return found

    # ------------------------------------------------------------------ chart
    def propose_chart(self, question: str, lookup: Dict[str, str], verify: bool = True) -> Optional[ChartSpec]:
        """Asks for a chart spec, then keeps it only if every label and number is in the evidence."""
        if self.llm_invoker is None or not lookup:
            return None
        evidence = "\n\n".join(f"[{k}]\n{v}" for k, v in lookup.items())
        prompt = f"{self.prompts['chart_instruction']}\n\nEvidence:\n{evidence}\n\nQuestion: {question}\n\nJSON:"
        raw = self.json_invoker(prompt) if self.json_invoker else ""
        match = re.search(r"\{[\s\S]*\}", str(getattr(raw, "content", raw) or ""))
        if not match:
            return None
        try:
            payload = json.loads(match.group(0)).get("chart")
            if not payload:
                return None
            spec = ChartSpec.model_validate(payload)
        except Exception as exc:
            logger.info(f"Chart proposal discarded (unparseable): {exc}")
            return None
        return verify_chart(spec, lookup) if verify else spec  # verify=False only for the no-verification ablation


def _numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def _number_forms(value: float) -> set:
    forms = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    return {f.rstrip("0").rstrip(".") if "." in f else f for f in forms} | forms


# "1. Treating pilots as the end goal", "Pitfall #2: ...", "(3) ...": a model numbers the labels of a list, but in a table the
# number sits in its own column, so the numbering is dropped before the label is looked up.
_ENUMERATOR = re.compile(r"^\s*(?:(?:pitfall|item|step|phase|point|no|number|rank)\s*)?[#(]?\s*\d{1,2}\s*[.):\-–]*\s+", re.I)


def _plain(text: str) -> str:
    """Lower-case letters, digits, % and $ only, single spaces: so punctuation and layout do not hide a match."""
    return " ".join(re.sub(r"[^a-z0-9%$ ]+", " ", text.lower()).split())


def _label_forms(label: str) -> set:
    stripped = label.strip()
    while True:
        shorter = _ENUMERATOR.sub("", stripped, count=1)
        if shorter == stripped or not shorter.strip():
            break
        stripped = shorter
    return {f for f in (_plain(stripped), _plain(label)) if f}


def _windows(passage: str) -> List[str]:
    """Places where a label and its value may sit together: one table row, or a line and the next (PDF text is hard-wrapped)."""
    lines = [ln for ln in passage.split("\n") if ln.strip()]
    out = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") or i + 1 >= len(lines):
            out.append(ln)
        else:
            out.append(ln + " " + lines[i + 1])
    return out


def verify_chart(spec: ChartSpec, lookup: Dict[str, str]) -> Optional[ChartSpec]:
    """Deterministic check: a chart is released only if all of its data is traceable to the evidence.

    Each point's label and value must appear together in one place of one passage: the same table row, or the same
    (or the next) line of text. So a number cannot be borrowed from another passage, or from another row of the same table."""
    if not 2 <= len(spec.data) <= 20:
        return None
    windows = [(w, _plain(w), _numbers(w)) for t in lookup.values() for w in _windows(t)]
    for point in spec.data:
        forms = _label_forms(point.label)
        wanted = _number_forms(point.value)
        traceable = any(any(f in plain for f in forms) and wanted & nums for _, plain, nums in windows)
        if not traceable:
            logger.info(f"Chart rejected: '{point.label}'={point.value} not found together in one table row or line of the evidence")
            return None
    return spec


def wants_chart(question: str, lookup: Dict[str, str]) -> bool:
    """Cheap pre-check so the second LLM call is only made when a chart could plausibly help."""
    if sum(len(_numbers(t)) for t in lookup.values()) < 3:
        return False
    explicit = re.search(r"\b(chart|graph|plot|visuali[sz]e|diagram)\b", question, re.I)
    quantitative = re.search(
        r"\b(compare|comparison|versus|vs\.?|trend|growth|change|over time|percent(age)?|share|rate|"
        r"distribution|breakdown|how (many|much)|rank|statistics|numbers?|figures?|data)\b", question, re.I)
    return bool(explicit or quantitative)
