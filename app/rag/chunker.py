import os
import uuid
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = 256  # all-MiniLM-L6-v2 silently truncates anything longer, including [CLS] and [SEP]
SEPARATORS = ["\n\n", "\n", ".", " ", ""]
TABLE_WHOLE_MAX_TOKENS = 1800  # Matches engine.CONTEXT_TOKEN_BUDGET; a table above this can't fit as the sole element


class DocumentChunker:
    """Splits documents into small, searchable pieces, each prefixed with its context.

    Paragraphs are split to fit the embedding model, measured with its own tokenizer, so no chunk
    is cut off when it is embedded. Tables are kept whole for the LLM (metadata['whole_content']):
    what gets embedded and searched is a small summary view plus one small view per row, each
    pointing back to the same parent_id, so a hit on any part of a table can retrieve the whole
    thing at query time.
    """

    def __init__(self, target_tokens: int = 220, chunk_overlap: int = 30,
                 model_name: str = EMBEDDING_MODEL, max_tokens: int = MODEL_MAX_TOKENS):
        if target_tokens > max_tokens:
            raise ValueError(f"target_tokens ({target_tokens}) exceeds the model limit ({max_tokens})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.target_tokens = target_tokens
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        """Tokens as the embedding model sees them, including [CLS] and [SEP]."""
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    @staticmethod
    def context_prefix(metadata: Dict[str, Any]) -> str:
        title = os.path.splitext(metadata.get("source", ""))[0]
        section = metadata.get("section", "")
        return f"{title} > {section}" if section else title

    def _chunk_paragraph(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        prefix = self.context_prefix(doc["metadata"])
        # The prefix, its newline and the two special tokens come out of the same budget
        body_budget = max(32, self.target_tokens - self.count_tokens(prefix + "\n"))
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=body_budget,
            chunk_overlap=min(self.chunk_overlap, body_budget // 4),
            separators=SEPARATORS,
        )
        chunks = []
        for i, split in enumerate(splitter.split_text(doc["content"])):
            metadata = doc["metadata"].copy()
            metadata.update({"chunk_id": i, "element_type": "paragraph", "parent_id": "", "view_type": "whole"})
            chunks.append({"content": f"{prefix}\n{split}" if prefix else split, "metadata": metadata})
        return chunks

    def _chunk_table(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        prefix = self.context_prefix(doc["metadata"])
        table_data = doc.get("table_data", {})
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])
        caption = table_data.get("caption", "")

        whole = f"{prefix}\n{doc['content']}".strip()
        too_large = self.count_tokens(whole) > TABLE_WHOLE_MAX_TOKENS

        parent_id = f"tbl_{uuid.uuid4().hex[:12]}"
        base_metadata = doc["metadata"].copy()
        base_metadata.update({
            "element_type": "table",
            "parent_id": parent_id,
            "whole_content": "" if too_large else whole,
            "table_rows": len(rows),
            "table_too_large": too_large,
        })

        def view(content: str, view_type: str, chunk_id: int, row_index: int) -> Dict[str, Any]:
            metadata = base_metadata.copy()
            metadata.update({"view_type": view_type, "chunk_id": chunk_id, "row_index": row_index})
            return {"content": content[: self.max_tokens * 6], "metadata": metadata}  # defensive char cap

        # row_index is the row's position within table_data['rows'] (0-based), separate from
        # chunk_id (this view's position among the table's OWN views). -1 marks the summary view,
        # which describes the whole table rather than one row. Lets a citation or audit trail point
        # at the exact source row without re-parsing the row view's text.
        views = [view(
            f"{prefix}\nTable summary: {caption} Columns: {', '.join(columns)}. {len(rows)} rows.".strip(),
            "summary", 0, -1,
        )]
        for i, row in enumerate(rows):
            row_text = f"{prefix}\nRow -> " + ", ".join(f"{c}={v}" for c, v in zip(columns, row))
            views.append(view(row_text, "row", i + 1, i))
        return views

    def chunk_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in docs:
            if doc["metadata"].get("type") == "table":
                chunks.extend(self._chunk_table(doc))
            else:
                chunks.extend(self._chunk_paragraph(doc))
        return chunks
