### Better chunking than the baseline. Splits text on detected known headers for arxiv_papaers : Abstract, Method, Experiments....
### If a section is still too large, it gets sub-chunked with chunk_fixed_size as a fallback.


from __future__ import annotations
import re
from src.arxiv_rag.chunking.fixed_size import chunk_fixed_size
from src.arxiv_rag.chunking.models import Chunk


_SECTION_NAMES = (
    r"Abstract|Introduction|Related Work|Background|"
    r"Methods?|Methodology|Approach|"
    r"Experiments?(?: Setup)?|Results?|Evaluation|"
    r"Discussion|Conclusions?|Acknowledgh?ments?|References|Appendix"
)
_HEADER_PATTERN = re.compile(
    rf"^\s*(?:\d+\.?\d*\.?\s+)?({_SECTION_NAMES})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MAX_SECTION_WORDS = 350

def chunk_by_structure(text, paper_id):
    matches = list(_HEADER_PATTERN.finditer(text))

    if not matches:
        sections = [("Full Text", text)]
    else:
        sections = []
        if matches[0].start() > 0:
            sections.append(("Preamble", text[: matches[0].start()]))
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append((title, text[start:end]))

    chunks = []
    chunk_index = 0
    for title, section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue
        sub_chunks = chunk_fixed_size(section_text, paper_id, chunk_size=_MAX_SECTION_WORDS, overlap=30)
        for sub in sub_chunks:
            chunks.append(Chunk(
                chunk_id=f"{paper_id}_struct_{chunk_index}",
                paper_id=paper_id,
                chunk_index=chunk_index,
                text=sub.text,
                method="structured",
                section_title=title,
            ))
            chunk_index += 1
    return chunks