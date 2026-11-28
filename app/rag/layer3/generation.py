import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from app.rag import tracing
from app.rag.layer3.contracts import ChartPoint, ChartSpec, ClaimObject

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
# Verifier v3 (feedback point 4; default, ATTS_L3_V3=0 restores v2): longer document names in the trailing
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

# Strict release (default; ATTS_L3_STRICT_RELEASE=0 restores the earlier rule to reproduce the reports made before it).
# Text that is released without verification must not be able to state a fact:
# - a note that the documents do not cover something is released unchecked only if the documents are its subject
#   ("The handbook does not specify ...") and it goes on to say nothing else ("..., so it is not possible to ...");
# - headings, short lines and lead-ins ("Key points:") are released only if they carry no number or negation and add no
#   word beyond the question, the verified sentences of the answer and a few generic lead-in words;
# - the intent line is cut to one sentence and dropped if it brings in a number or a negation the question does not have.
STRICT_RELEASE = os.getenv("ATTS_L3_STRICT_RELEASE", "1") == "1"
_DOC_SUBJECT = (r"(?:the\s+|this\s+|these\s+)?(?:provided\s+|available\s+|given\s+|retrieved\s+)?"
                r"(?:evidence|documents?|passages?|sources?|context|text|information|policy|handbook|records?|materials?|excerpts?)")
_ABSTAIN_STRICT = re.compile(
    r"^(?:however,\s*|unfortunately,\s*)?(?:"
    + _DOC_SUBJECT + r"\s+(?:does not|do not|doesn't|don't|did not)\s+(?:explicitly\s+|specifically\s+|directly\s+|clearly\s+)?"
    r"(?:mention|specify|provide|state|cover|describe|contain|include|discuss|say|address|give|list|detail)"
    r"|there\s+(?:is|are)\s+no\s+(?:information|mention|details?|data|evidence)"
    r"|no\s+(?:information|details?|data)\s+(?:is|are)\s+(?:available|provided|given)"
    r")\b", re.I)
# Anything after the note itself is a claim: "..., but it implies ...", "..., so it is not possible ...", "...; managers ..."
_ABSTAIN_MORE = re.compile(r"\b(?:but|so|therefore|thus|hence|however|although|though|whereas|while|instead|and|"
                           r"meaning|means|implies|implying|suggests?|indicates?)\b|[;,]", re.I)
_UNCHECKED_NEGATION = re.compile(r"\b(?:not|no|never|cannot|none|without|neither|nor|only)\b|n't", re.I)
_DIGIT = re.compile(r"\d")
_WORDS = re.compile(r"[a-z0-9]+")
_STRUCT_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were", "for", "on", "that", "with",
                "as", "by", "it", "this", "be", "from", "at", "their", "its", "can", "has", "have", "these", "those", "your",
                "you", "our", "we", "i", "s"}
LEAD_IN_WORDS = {"here", "key", "main", "point", "points", "following", "details", "detail", "steps", "step", "summary",
                 "reasons", "reason", "options", "option", "answer", "short", "overview", "requirements", "requirement",
                 "rules", "rule", "factors", "ways", "things", "important", "note", "notes", "list", "below", "follows",
                 "include", "includes", "including", "consider", "considerations", "some", "what", "know", "need",
                 "information", "brief", "summarized", "breakdown", "highlights", "conclusion", "background"}


def _words(text: str) -> set:
    return {w for w in _WORDS.findall(text.lower()) if w not in _STRUCT_STOP and len(w) > 1}


def _numbers_in(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def is_plain_abstention(text: str, question: str = "") -> bool:
    """A note that the documents do not cover something, and nothing more: safe to release without verification."""
    return (bool(_ABSTAIN_STRICT.search(text)) and not _ABSTAIN_MORE.search(_ABSTAIN_STRICT.sub("", text, count=1))
            and _numbers_in(text) <= _numbers_in(question) and len(text.split()) <= 30)


def structural_text_allowed(text: str, allowed_words: set) -> bool:
    """A heading, short line or lead-in may be released unverified only if it cannot state a fact of its own."""
    return (not _DIGIT.search(text) and not _UNCHECKED_NEGATION.search(text)
            and _words(text) <= (allowed_words | LEAD_IN_WORDS))


def release_intent(intent: Optional[str], question: str) -> Optional[str]:
    """The intent line as shown to the user: one sentence that restates the question, never a fact of its own."""
    if not intent or not STRICT_RELEASE:
        return intent
    first = re.split(r"(?<=[.!?])\s+", intent.strip(), maxsplit=1)[0].strip()
    if (not first or len(first.split()) > 40 or not _numbers_in(first) <= _numbers_in(question)
            or len(_UNCHECKED_NEGATION.findall(first)) > len(_UNCHECKED_NEGATION.findall(question))):
        return None
    return first

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
    question: str = ""                # what unverified structural text may repeat (strict release)


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
            with tracing.llm_purpose("answer_continue"):
                raw = head + self._normalize_labels(self._call(prompt + "\n\n" + head))
        intent, lines = self.parse(raw, set(lookup), self.strip_meta, question=question)
        claims, order = [], 0
        for line in lines:
            for sentence in line.sentences:
                if sentence.checkable:
                    order += 1
                    claims.append(ClaimObject(claim_id=f"C{order}", order=order,
                                              text=sentence.text, evidence_ids=sentence.evidence_ids))
        return Draft(intent, lines, claims, lookup, to_uid, question=question)

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
    def parse(raw: str, valid_ids: set, strip_meta: bool = True, question: str = "") -> Tuple[Optional[str], List[Line]]:
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
            AnswerGenerator._parse_line(ln, valid_ids, strip_meta, question) for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("|") and not _CHART_ANNOUNCE.search(ln)
        ]
        return intent, [ln for ln in lines if ln.sentences]

    @staticmethod
    def _parse_line(raw_line: str, valid_ids: set, strip_meta: bool = True, question: str = "") -> Line:
        heading = re.match(r"^\s*(#{1,6}\s+)(.*)$", raw_line)
        if heading:
            text = heading.group(2).strip()
            if STRICT_RELEASE and text.rstrip("*_ ").endswith("?"):
                return Line(prefix="", sentences=[])  # a heading that repeats a question (see below) is dropped
            return Line(prefix=heading.group(1), sentences=[Sentence(text, [], checkable=False)])
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
        if STRICT_RELEASE:
            # A question states no fact, yet its words occur in the evidence, so it could pass the extractive check and
            # be released with a citation ("Who can work remotely? [E1]"): the answer never needs to repeat a question
            sentences = [s for s in sentences if not s.text.rstrip("*_ ").endswith("?")]
        for sentence in sentences:
            words = len(sentence.text.split())
            uncited = not sentence.evidence_ids
            structural = uncited and (words < 4 or (sentence.text.endswith(":") and words <= 10))
            if STRICT_RELEASE:
                # a short line or lead-in with a number or a negation states a fact ("Limit: 500 dollars."): verify it
                structural = structural and not _DIGIT.search(sentence.text) and not _UNCHECKED_NEGATION.search(sentence.text)
                abstain = uncited and is_plain_abstention(sentence.text, question)
            else:
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
        return verify_chart(spec, lookup, question) if verify else spec  # verify=False only for the no-verification ablation


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


CHART_WORDS = {"chart", "comparison", "compared", "per", "vs", "versus", "value", "values", "count", "counts", "share",
               "shares", "breakdown", "distribution", "trend", "over", "time", "total", "totals", "number", "numbers",
               "amount", "amounts", "percentage", "percent", "rate", "rates", "top", "figures", "category", "categories",
               "label", "labels", "level", "levels", "type", "types", "item", "items"}
NEUTRAL_CHART_TITLE = "Figures from the evidence"
_UNIT_ALIASES = {"%": ("%", "percent"), "percent": ("%", "percent"), "percentage": ("%", "percent"),
                 "$": ("$", "usd", "dollar"), "usd": ("$", "usd", "dollar"), "dollars": ("$", "usd", "dollar")}


def _label_in(form: str, plain: str) -> bool:
    """Whole words only: the label "Q1" must not match "Q10"."""
    return f" {form} " in f" {plain} "


def _unit_in(unit: str, window: str) -> bool:
    u, w = unit.strip().lower(), window.lower()
    for alias in _UNIT_ALIASES.get(u, (u,)):
        if (alias in w) if not alias[0].isalnum() else re.search(r"\b" + re.escape(alias), w):
            return True
    return False


# ---------- Chart check v2: the points must be ONE series of the evidence ----------
# Each point appearing somewhere in the evidence is not enough: a chart could mix columns, quarters, tables or years,
# show a Total next to its own parts, plot list numbering, swap two values of one sentence, or draw a pie of values
# that are not parts of one whole. v2 accepts a chart only if all its points are one series of ONE passage:
#   R1 one passage; R2 in a table, one column, each value in its own row's cell (a cell with a range or several
#   numbers holds no single value); R3 one unit; R4 no Total next to its parts; R5 list numbering is never a value;
#   R6 a pie only for shares adding up to 100 or for all the parts of a Total row, otherwise it is shown as a bar
#   chart; R7 in a sentence with several figures, each label must get the figure that goes with it in one
#   consistent reading, otherwise the chart is refused; R8 a line chart only over time-like labels, in the document's
#   order (put back in that order if needed), otherwise it is shown as a bar chart.
# A chart may also be one table ROW over its columns (R9: the labels are column names, "Q1", "Q2", "Q3"; exactly one
# row holds all the values, in one unit), and a label may be a numbered name such as "Phase 1" or "Pitfall #3" for
# the row whose numbering column holds that number (R10; a bare "1" is not a label, so a numbering cell is never
# matched as label text). A row chart names its row only in its title, so the title must name that row and no other;
# otherwise the row's own name becomes the title (R11).
# Default on; ATTS_L3_CHART_V2=0 restores the earlier per-point check to reproduce the reports made before it.
CHART_V2 = os.getenv("ATTS_L3_CHART_V2", "1") == "1"
_TOTAL_LABEL = re.compile(r"^\s*(?:grand\s+)?(?:total|sum|overall|all)\b", re.I)
_NUMBERING_HEADER = re.compile(r"^\s*(?:phase|step|pitfall|item|wave|stage|rank|no\.?|number|nr\.?|#|s\.?\s?no\.?)\s*$", re.I)
_TIME_LABEL = re.compile(
    r"^(?:(?:fy\s?)?(?:19|20)\d\d|q[1-4](?:\s+(?:fy\s?)?(?:19|20)\d\d)?|(?:19|20)\d\d\s+q[1-4]|h[12](?:\s+(?:19|20)\d\d)?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+(?:19|20)\d\d)?|(?:week|day|month|year)\s+\d{1,3})$", re.I)
_SEPARATOR_ROW = re.compile(r"^\|?[\s:|-]+\|?$")
_CELL_NUMBER = re.compile(r"^\s*[$€£]?\s*(\d[\d,]*\.?\d*)\s*([%a-z$€£]*)\s*$", re.I)
_TOKEN = re.compile(r"\d[\d,]*\.?\d*%?|[a-z]+|%")


def _pipe_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    """(header, rows) for each block of '|' lines; the --- separator row is optional."""
    blocks, current = [], []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            current.append(s)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    tables = []
    for block in blocks:
        cells = [[c.strip() for c in l.strip("|").split("|")] for l in block
                 if not (_SEPARATOR_ROW.fullmatch(l) and "-" in l)]
        if len(cells) >= 2:
            tables.append((cells[0], cells[1:]))
    return tables


def _cell_number(cell: str) -> Optional[Tuple[str, str]]:
    """The single number a cell states and its unit, or None (text, ranges such as '2-3 weeks', phone numbers)."""
    m = _CELL_NUMBER.match(cell)
    return (m.group(1).replace(",", "").rstrip("."), (m.group(2) or "").lower()) if m else None


def _is_numbering(header: List[str], rows: List[List[str]], j: int) -> bool:
    """R5: a column of list numbers 1, 2, 3 ... (the first column, or one named Phase, Step, No. ...)."""
    values = [_cell_number(r[j]) if j < len(r) else None for r in rows]
    if not all(v and v[1] == "" and v[0].isdigit() for v in values):
        return False
    named = j < len(header) and bool(_NUMBERING_HEADER.match(header[j]))
    consecutive = [int(v[0]) for v in values] == list(range(1, len(rows) + 1))
    return named or (consecutive and j == 0)


def _is_total_row(row: List[str], j: int) -> bool:
    return any(_TOTAL_LABEL.match(c) for k, c in enumerate(row) if k != j)


def _numbering_columns(header: List[str], rows: List[List[str]]) -> set:
    width = max([len(header)] + [len(r) for r in rows])
    return {j for j in range(width) if _is_numbering(header, rows, j)}


def _numbered_names(header: List[str], row: List[str], numbering: set) -> set:
    """R10: the names a numbered row goes by ("Phase 1" for the row whose Phase column holds 1)."""
    return {_plain(f"{header[k]} {row[k]}") for k in numbering if k < len(header) and k < len(row)}


def _label_names_row(label: str, header: List[str], row: List[str], j: int, numbering: set) -> bool:
    """The label names this row: it is in the row's other cells (never a numbering cell), or it is a numbered name."""
    forms = _label_forms(label)
    rest = _plain(" ".join(c for k, c in enumerate(row) if k != j and k not in numbering))
    names = _numbered_names(header, row, numbering)
    return any(_label_in(f, rest) or f in names for f in forms)


def _row_name(row: List[str], numbering: set) -> str:
    """The row's own name: its first cell that is neither a number nor list numbering."""
    for k, c in enumerate(row):
        if k not in numbering and c.strip() and not _cell_number(c):
            return c.strip()
    return ""


def _table_point_windows(point: ChartPoint, lookup: Dict[str, str]) -> List[Tuple[str, str]]:
    """Per-point check for table labels that are not written in the value's row (R9, R10): the value's column name, or
    a numbered name of its row. Returns (the row as a line of text, passage) for each place that holds the point."""
    forms, wanted = _label_forms(point.label), _number_forms(point.value)
    out = []
    for passage in lookup.values():
        for header, rows in _pipe_tables(passage):
            numbering = _numbering_columns(header, rows)
            for row in rows:
                names = _numbered_names(header, row, numbering)
                for j, cell in enumerate(row):
                    number = _cell_number(cell)
                    if j in numbering or not number or number[0] not in wanted:
                        continue
                    column = _plain(header[j]) if j < len(header) else ""
                    if any(_label_in(f, column) or f in names for f in forms):
                        out.append(("| " + " | ".join(row) + " |", passage))
    return out


def _table_series(spec: ChartSpec, text: str):
    """R2, R3, R5: the chart as one column of one table in `text`: (header, rows, column, row of each point, unit)."""
    for header, rows in _pipe_tables(text):
        width = max([len(header)] + [len(r) for r in rows])
        numbering = _numbering_columns(header, rows)
        for j in range(width):
            if j in numbering:
                continue
            chosen: List[int] = []
            units = set()
            for point in spec.data:
                wanted = _number_forms(point.value)
                hit = None
                for i, row in enumerate(rows):
                    if i in chosen or j >= len(row):
                        continue
                    number = _cell_number(row[j])
                    if not number or number[0] not in wanted:
                        continue
                    if _label_names_row(point.label, header, row, j, numbering):
                        hit, unit = i, number[1]
                        break
                if hit is None:
                    break
                chosen.append(hit)
                units.add(unit)
            else:
                if len(units) == 1:
                    return header, rows, j, chosen, units.pop()
    return None


def _row_series(spec: ChartSpec, text: str):
    """R9: the chart as one row of one table over its columns, the labels being column names. Returns (header, rows,
    row index, column of each point, unit), "ambiguous" when the values fit more than one row, or None."""
    for header, rows in _pipe_tables(text):
        numbering = _numbering_columns(header, rows)
        candidates = []
        for i, row in enumerate(rows):
            columns: List[int] = []
            units = set()
            for point in spec.data:
                forms, wanted = _label_forms(point.label), _number_forms(point.value)
                hit = None
                for j, cell in enumerate(row):
                    if j in numbering or j in columns or j >= len(header):
                        continue
                    number = _cell_number(cell)
                    if number and number[0] in wanted and any(_label_in(f, _plain(header[j])) for f in forms):
                        hit, unit = j, number[1]
                        break
                if hit is None:
                    break
                columns.append(hit)
                units.add(unit)
            else:
                if len(units) == 1:
                    candidates.append((i, columns, units.pop()))
        if len(candidates) > 1:
            return "ambiguous"
        if candidates:
            i, columns, unit = candidates[0]
            return header, rows, i, columns, unit
    return None


def _titled_for_row(spec: ChartSpec, rows: List[List[str]], i: int, numbering: set) -> ChartSpec:
    """R11: a row chart names its row only in its title, so the title must name that row and no other row."""
    name = _row_name(rows[i], numbering)
    title = _plain(spec.title)
    own = _plain(name)
    names_own = bool(own) and _label_in(own, title)
    others = {_plain(_row_name(r, numbering)) for k, r in enumerate(rows) if k != i} - {"", own}
    names_other = any(_label_in(o, title) for o in others if o not in own)
    if names_own and not names_other:
        return spec
    logger.info("Chart title replaced by the name of the table row its values come from")
    return spec.model_copy(update={"title": name or NEUTRAL_CHART_TITLE})


def _tokens(text: str) -> List[str]:
    """Words and numbers; a full stop after a number ends the sentence ("in 2025.") and is not part of it."""
    return [tok.rstrip(".") if tok[0].isdigit() else tok for tok in _TOKEN.findall(text.lower())]


def _number_token(token: str) -> str:
    return token.rstrip("%").replace(",", "").rstrip(".")


def _prose_series(spec: ChartSpec, text: str):
    """R7: the chart as figures stated in the sentences of `text`, each label paired with its own figure.

    In a line (joined with the next, as PDF text is hard-wrapped) each label of the chart owns the figures between it
    and the next label ("APAC grew 35% while EMEA grew 21%"), or those between the previous label and it ("from 120
    staff in 2024 to 150 staff in 2025"). A point counts only if one of the two readings gives every label of the
    chart in that line its own value (a label with no figure next to it in that line is decided in another line); a
    label written twice in one line makes the pairing unclear, so that line is not used. Returns (position of each
    point, whether each value is a percentage) or None."""
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("|")]
    windows = [ln + " " + lines[i + 1] if i + 1 < len(lines) else ln for i, ln in enumerate(lines)]
    positions: Dict[int, Tuple[int, int]] = {}
    percent: Dict[int, bool] = {}
    for wi, window in enumerate(windows):
        toks = _tokens(window)
        found: Dict[int, List[int]] = {}
        lengths: Dict[int, int] = {}
        for k, point in enumerate(spec.data):
            for form in sorted(_label_forms(point.label), key=len):
                ft = _tokens(form)
                starts = [s for s in range(len(toks) - len(ft) + 1) if ft and toks[s:s + len(ft)] == ft]
                if starts:
                    found[k], lengths[k] = starts, len(ft)
                    break
        if not found or any(len(s) > 1 for s in found.values()):
            continue
        spans = sorted((s[0], s[0] + lengths[k], k) for k, s in found.items())
        inside = {t for a, b, _ in spans for t in range(a, b)}
        numbers = [(t, _number_token(tok)) for t, tok in enumerate(toks) if t not in inside and tok[0].isdigit()]
        for direction in ("after", "before"):
            matched = {}
            for idx, (a, b, k) in enumerate(spans):
                if direction == "after":
                    lo, hi = b, spans[idx + 1][0] if idx + 1 < len(spans) else len(toks)
                else:
                    lo, hi = spans[idx - 1][1] if idx else 0, a
                wanted = _number_forms(spec.data[k].value)
                own = [(t, n) for t, n in numbers if lo <= t < hi]
                if not own:
                    continue   # no figure next to this label in this line (it may be on the next one): not decided here
                hits = [t for t, n in own if n in wanted]
                if not hits:
                    break      # the label's own figure is a different number: this reading does not fit the chart
                t = hits[0]
                matched[k] = (a, toks[t].endswith("%") or (t + 1 < len(toks) and toks[t + 1] in ("%", "percent")))
            else:
                if not matched:
                    continue
                for k, (a, pct) in matched.items():
                    if k not in positions:
                        positions[k] = (wi, a)
                        percent[k] = pct
                break
    if len(positions) != len(spec.data):
        return None
    return positions, percent


def _reshaped(spec: ChartSpec, chart_type: str, order: List[int]) -> ChartSpec:
    """The same chart, as `chart_type` with its points in `order`; the chart itself when nothing changes."""
    if chart_type == spec.type and order == list(range(len(spec.data))):
        return spec
    return spec.model_copy(update={"type": chart_type, "data": [spec.data[i] for i in order]})


def _shaped_for(spec: ChartSpec, is_whole: bool, source_order: List[Tuple[int, int]]) -> ChartSpec:
    """R6 and R8 on a chart whose points are one series: the right chart type, the document's order for a line."""
    n = len(spec.data)
    if spec.type == "pie" and not is_whole:
        logger.info("Chart shown as a bar chart: a pie needs shares adding up to 100 or all the parts of a total")
        return _reshaped(spec, "bar", list(range(n)))
    if spec.type == "line":
        if not all(_TIME_LABEL.match(_plain(p.label)) for p in spec.data):
            logger.info("Chart shown as a bar chart: a line needs time-like labels (years, quarters, months, weeks)")
            return _reshaped(spec, "bar", list(range(n)))
        order = sorted(range(n), key=lambda k: source_order[k])
        if order != list(range(n)):
            logger.info("Chart points put in the document's order (line chart over time)")
        return _reshaped(spec, "line", order)
    return spec


def _one_series(spec: ChartSpec, lookup: Dict[str, str]) -> Optional[ChartSpec]:
    """Chart check v2 (see CHART_V2): the chart if its points are one series of one passage, possibly as a bar chart
    or put in the document's order, else None."""
    if any(_TOTAL_LABEL.match(p.label) for p in spec.data):
        logger.info("Chart rejected: a total is shown next to its own parts")
        return None
    for text in lookup.values():
        table = _table_series(spec, text)
        if table is not None:
            header, rows, j, chosen, unit = table
            if any(_is_total_row(rows[i], j) for i in chosen):
                logger.info("Chart rejected: a total is shown next to its own parts")
                return None
            values = [p.value for p in spec.data]
            percent = unit == "%" or bool(re.search(r"%|percent|share", header[j] if j < len(header) else "", re.I))
            parts = [i for i, r in enumerate(rows) if j < len(r) and _cell_number(r[j]) and not _is_total_row(r, j)]
            totals = [_cell_number(r[j]) for r in rows if j < len(r) and _cell_number(r[j]) and _is_total_row(r, j)]
            whole = (percent and abs(sum(values) - 100) <= 1.0) or (
                sorted(chosen) == parts and len(totals) == 1
                and abs(sum(values) - float(totals[0][0])) <= 0.005 * abs(float(totals[0][0])) + 0.01 * len(values))
            return _shaped_for(spec, whole, [(0, i) for i in chosen])
        row_series = _row_series(spec, text)
        if row_series == "ambiguous":
            logger.info("Chart rejected: its values fit more than one row of the table, so its row cannot be named")
            return None
        if row_series is not None:
            header, rows, i, columns, unit = row_series
            values = [p.value for p in spec.data]
            numbering = _numbering_columns(header, rows)
            row = rows[i]
            percent = unit == "%"
            totals = [j for j in range(min(len(header), len(row))) if _TOTAL_LABEL.match(header[j]) and _cell_number(row[j])]
            parts = [j for j in range(min(len(header), len(row))) if j not in numbering and j not in totals
                     and _cell_number(row[j]) and _cell_number(row[j])[1] == unit]
            whole = (percent and abs(sum(values) - 100) <= 1.0) or (
                sorted(columns) == parts and len(totals) == 1
                and abs(sum(values) - float(_cell_number(row[totals[0]])[0])) <= 0.005 * abs(float(_cell_number(row[totals[0]])[0])) + 0.01 * len(values))
            shaped = _shaped_for(spec, whole, [(0, j) for j in columns])
            return _titled_for_row(shaped, rows, i, numbering)
        prose = _prose_series(spec, text)
        if prose is not None:
            positions, percent = prose
            if len(set(percent.values())) > 1:
                logger.info("Chart rejected: its values are in different units")
                return None
            whole = all(percent.values()) and abs(sum(p.value for p in spec.data) - 100) <= 1.0
            return _shaped_for(spec, whole, [positions[k] for k in range(len(spec.data))])
    logger.info("Chart rejected: its points are not one series of one passage (one table column, or figures stated together)")
    return None


def verify_chart(spec: ChartSpec, lookup: Dict[str, str], question: str = "") -> Optional[ChartSpec]:
    """Deterministic check: a chart is released only if all of its data is traceable to the evidence.

    Each point's label and value must appear together in one place of one passage: the same table row, or the same
    (or the next) line of text. So a number cannot be borrowed from another passage, or from another row of the same table.
    With strict release, the text around the data is checked too: the unit must be written next to every value, and the
    title and axis labels may not carry a number that is not in the question or the rows the data came from, a negation,
    or a word that is not in the question, the labels or the passages the data came from (a table's header names the
    columns). A failing title is replaced by a neutral one and failing axis labels or units are left out,
    so the text cannot claim what the data does not show ("Refunds collapsed by 90% this year").
    Chart check v2 (CHART_V2, default) also requires the points to be one series of one passage (_one_series)."""
    if not 2 <= len(spec.data) <= 20:
        return None
    windows = [(w, _plain(w), _numbers(w), t) for t in lookup.values() for w in _windows(t)]
    sources: List[List[Tuple[str, str]]] = []   # for each point, the (window, passage) pairs that hold its label and value
    for point in spec.data:
        forms = _label_forms(point.label)
        wanted = _number_forms(point.value)
        found = [(raw, passage) for raw, plain, nums, passage in windows if wanted & nums and any(
            (_label_in(f, plain) if STRICT_RELEASE else f in plain) for f in forms)]
        if not found and CHART_V2:
            found = _table_point_windows(point, lookup)
        if not found:
            logger.info(f"Chart rejected: '{point.label}'={point.value} not found together in one table row or line of the evidence")
            return None
        sources.append(found)
    if CHART_V2:
        spec = _one_series(spec, lookup)
        if spec is None:
            return None
    if not STRICT_RELEASE:
        return spec

    update: Dict[str, Any] = {}
    if spec.unit and not all(any(_unit_in(spec.unit, w) for w, _ in found) for found in sources):
        update["unit"] = ""
    allowed = _words(question) | CHART_WORDS | _words(spec.unit)
    numbers = _numbers_in(question)
    for found in sources:
        for w, passage in found:
            allowed |= _words(passage)
            numbers |= _numbers_in(w)
    for point in spec.data:
        allowed |= _words(point.label)

    def text_ok(text: str) -> bool:
        return (_numbers_in(text) <= numbers and not _UNCHECKED_NEGATION.search(text)
                and {w for w in _words(text) if len(w) > 2} <= allowed)

    if spec.title and not text_ok(spec.title):
        update["title"] = NEUTRAL_CHART_TITLE
    for axis in ("x_label", "y_label"):
        if getattr(spec, axis) and not text_ok(getattr(spec, axis)):
            update[axis] = ""
    if spec.evidence_id and spec.evidence_id not in lookup:
        update["evidence_id"] = None
    if update:
        logger.info(f"Chart text not found in the evidence, replaced: {sorted(update)}")
        return spec.model_copy(update=update)
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
