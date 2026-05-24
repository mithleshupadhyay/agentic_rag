import html
import logging
import re

from fastapi import HTTPException

from agentic_rag.shared.schemas.common import Citation
from agentic_rag.shared.schemas.retrieval import (
    ContextBuildRequest,
    ContextBuildResponse,
    ContextChunk,
)


logger = logging.getLogger(__name__)


def build_context(request: ContextBuildRequest) -> ContextBuildResponse:
    logger.info(
        f"[ContextBuilder] Building context candidates={len(request.chunks)} "
        f"max_chunks={request.max_context_chunks} max_tokens={request.max_tokens}"
    )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Context build query is required.")

    context_chunks: list[ContextChunk] = []
    seen_chunk_ids: set[str] = set()
    seen_content: set[str] = set()
    total_tokens = 0

    for candidate in request.chunks:
        if len(context_chunks) >= request.max_context_chunks:
            break

        chunk_key = str(candidate.chunk_id)
        if chunk_key in seen_chunk_ids:
            logger.info(f"[ContextBuilder] Skipping duplicate chunk {candidate.chunk_id}")
            continue

        content = candidate.content or (candidate.citation.quote if candidate.citation else None)
        if not content:
            logger.info(f"[ContextBuilder] Skipping empty chunk {candidate.chunk_id}")
            continue

        cleaned_content = re.sub(r"<[^>]+>", "", content)
        cleaned_content = html.unescape(cleaned_content)
        cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
        if not cleaned_content:
            logger.info(f"[ContextBuilder] Skipping blank chunk {candidate.chunk_id}")
            continue

        normalized_content = cleaned_content.lower()
        if normalized_content in seen_content:
            logger.info(f"[ContextBuilder] Skipping repeated content chunk {candidate.chunk_id}")
            continue

        token_count_value = candidate.metadata.get("token_count")
        token_count = token_count_value if isinstance(token_count_value, int) else 0
        if token_count <= 0:
            token_count = max(1, len(cleaned_content.split()))

        remaining_tokens = request.max_tokens - total_tokens
        if remaining_tokens <= 0:
            break

        if token_count > remaining_tokens:
            words = cleaned_content.split()
            cleaned_content = " ".join(words[:remaining_tokens]).strip()
            if not cleaned_content:
                break
            token_count = remaining_tokens

        citation_source = candidate.citation
        citation = Citation(
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            title=citation_source.title if citation_source else None,
            source_uri=citation_source.source_uri if citation_source else None,
            page_number=citation_source.page_number if citation_source else None,
            section_path=citation_source.section_path if citation_source else None,
            quote=cleaned_content,
            score=citation_source.score if citation_source and citation_source.score is not None else candidate.score,
        )

        context_chunks.append(
            ContextChunk(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=cleaned_content,
                token_count=token_count,
                citation=citation,
                metadata={
                    **candidate.metadata,
                    "score": candidate.score,
                    "source": str(candidate.source),
                },
            )
        )
        seen_chunk_ids.add(chunk_key)
        seen_content.add(normalized_content)
        total_tokens += token_count

    logger.info(
        f"[ContextBuilder] Built context chunks={len(context_chunks)} "
        f"token_count={total_tokens}"
    )
    return ContextBuildResponse(
        context=context_chunks,
        token_count=total_tokens,
    )
