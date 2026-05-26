import logging
import re
from dataclasses import dataclass, field

from agentic_rag.shared.schemas.retrieval import ContextChunk


logger = logging.getLogger(__name__)


@dataclass
class AnswerVerificationResult:
    passed: bool
    reason: str
    cited_source_numbers: list[int] = field(default_factory=list)


def verify_answer_support(
    answer: str,
    context: list[ContextChunk],
) -> AnswerVerificationResult:
    normalized_answer = answer.strip()
    if not context:
        return AnswerVerificationResult(
            passed=True,
            reason="No context was available for answer verification.",
        )

    if not normalized_answer:
        logger.warning("[AnswerVerifier] Answer verification failed because answer is empty")
        return AnswerVerificationResult(
            passed=False,
            reason="Answer is empty.",
        )

    citation_matches = re.findall(r"\[(\d+)\]", normalized_answer)
    if not citation_matches:
        logger.warning(
            f"[AnswerVerifier] Answer verification failed because citations are missing "
            f"context_chunks={len(context)}"
        )
        return AnswerVerificationResult(
            passed=False,
            reason="Answer did not cite retrieved context.",
        )

    cited_source_numbers = sorted({int(match) for match in citation_matches})
    valid_source_numbers = set(range(1, len(context) + 1))
    invalid_source_numbers = [
        source_number
        for source_number in cited_source_numbers
        if source_number not in valid_source_numbers
    ]
    if invalid_source_numbers:
        logger.warning(
            f"[AnswerVerifier] Answer verification failed because citations are invalid "
            f"cited_source_numbers={cited_source_numbers} "
            f"context_chunks={len(context)}"
        )
        return AnswerVerificationResult(
            passed=False,
            reason=(
                "Answer cited source numbers that were not present in retrieved context."
            ),
            cited_source_numbers=cited_source_numbers,
        )

    logger.info(
        f"[AnswerVerifier] Answer verification passed "
        f"context_chunks={len(context)} cited_source_numbers={cited_source_numbers}"
    )
    return AnswerVerificationResult(
        passed=True,
        reason="Answer citations match retrieved context.",
        cited_source_numbers=cited_source_numbers,
    )
