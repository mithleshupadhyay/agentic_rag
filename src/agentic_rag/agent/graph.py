import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from agentic_rag.agent.runtime import (
    SAFE_FALLBACK_ANSWER,
    evaluate_agent_guardrails,
    record_agent_step,
    start_agent_state,
)
from agentic_rag.core.models.user_context import UserContext
from agentic_rag.llm.gateway import generate_chat_completion, stream_chat_completion
from agentic_rag.query.answer_verifier import verify_answer_support
from agentic_rag.retrieval.bm25_search import search_bm25_chunks
from agentic_rag.retrieval.context_builder import build_context
from agentic_rag.shared.db.crud.agent_runs import (
    create_agent_run,
    is_agent_run_cancelled,
    record_agent_run_step,
    save_agent_checkpoint,
)
from agentic_rag.shared.schemas.agent import (
    AgentCheckpoint,
    AgentGraphState,
    AgentLimits,
    AgentNodeName,
    AgentRunStatus,
    AgentStreamEvent,
    AgentStreamEventType,
    AgentStateModel,
)
from agentic_rag.shared.schemas.auth import AuthContext
from agentic_rag.shared.schemas.llm import (
    ChatCompletionRequest,
    LLMMessage,
    LLMStreamEventType,
)
from agentic_rag.shared.schemas.retrieval import (
    ContextBuildRequest,
    RetrievalFilters,
    RetrievalStrategy,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentGraphRunResult:
    state: AgentStateModel
    checkpoints: list[AgentCheckpoint]
    status: AgentRunStatus
    stop_reason: str | None = None


def classify_intent_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before intent classification "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.CLASSIFY_INTENT.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.CLASSIFY_INTENT.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
            }

    logger.info(
        f"[AgentGraph] Classifying intent "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    agent_state.intent = "retrieve_and_answer"

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.CLASSIFY_INTENT,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def rewrite_query_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before query rewrite "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.REWRITE_QUERY.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.REWRITE_QUERY.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Rewriting query "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    agent_state.rewritten_query = agent_state.query.strip()

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.REWRITE_QUERY,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def plan_filters_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before filter planning "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.PLAN_FILTERS.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.PLAN_FILTERS.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Planning filters "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    if "query" not in agent_state.filters:
        agent_state.filters["query"] = agent_state.rewritten_query or agent_state.query

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.PLAN_FILTERS,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def select_retrieval_strategy_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before retrieval strategy selection "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(
                AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value
            )
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Selecting retrieval strategy "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Prepare state for this graph node.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    agent_state.retrieval_strategy = RetrievalStrategy.BM25

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.SELECT_RETRIEVAL_STRATEGY,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def bm25_search_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before BM25 retrieval "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.BM25_SEARCH.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.BM25_SEARCH.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Running BM25 retrieval "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Search only through the authorized retrieval service.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    query_text = agent_state.rewritten_query or agent_state.query
    user_context = UserContext(
        id=agent_state.auth.user_id,
        customer_id=agent_state.auth.tenant_id,
        tenant_id=agent_state.auth.tenant_id,
        workspace_id=agent_state.auth.workspace_id,
        roles=agent_state.auth.roles,
        group_ids=agent_state.auth.group_ids,
        scopes=agent_state.auth.scopes,
        acl_version=agent_state.auth.acl_version,
    )
    retrieval_response = search_bm25_chunks(
        user_context=user_context,
        query=query_text,
        filters=graph_state.retrieval_filters,
        limit=graph_state.retrieval_limit,
        search_client=graph_state.search_client,
    )
    agent_state.retrieval_strategy = retrieval_response.strategy
    agent_state.retrieved_candidates = retrieval_response.candidates
    agent_state.authorized_chunks = retrieval_response.candidates
    top_score = None
    if retrieval_response.candidates:
        top_score = retrieval_response.candidates[0].score
    active_filter_fields = []
    if graph_state.retrieval_filters.workspace_id:
        active_filter_fields.append("workspace_id")
    if graph_state.retrieval_filters.document_ids:
        active_filter_fields.append("document_ids")
    if graph_state.retrieval_filters.source_types:
        active_filter_fields.append("source_types")
    if graph_state.retrieval_filters.tags:
        active_filter_fields.append("tags")
    if graph_state.retrieval_filters.metadata:
        active_filter_fields.append("metadata")
    if graph_state.retrieval_filters.date_range:
        active_filter_fields.append("date_range")
    agent_state.replay_metadata[AgentNodeName.BM25_SEARCH.value] = {
        "node_name": AgentNodeName.BM25_SEARCH.value,
        "stage": "retrieval",
        "tool_name": "bm25_search",
        "input": {
            "retrieval_strategy": retrieval_response.strategy.value,
            "retrieval_limit": graph_state.retrieval_limit,
            "workspace_id": graph_state.retrieval_filters.workspace_id,
            "active_filter_fields": active_filter_fields,
            "metadata_filter_keys": sorted(graph_state.retrieval_filters.metadata),
            "query_length": len(query_text),
        },
        "output": {
            "candidate_count": len(retrieval_response.candidates),
            "authorized_chunk_count": len(agent_state.authorized_chunks),
            "latency_ms": retrieval_response.latency_ms,
            "top_score": top_score,
            "document_ids": [
                str(candidate.document_id)
                for candidate in retrieval_response.candidates[:10]
            ],
            "chunk_ids": [
                str(candidate.chunk_id)
                for candidate in retrieval_response.candidates[:10]
            ],
        },
        "summary": (
            f"BM25 retrieval returned {len(retrieval_response.candidates)} "
            f"authorized candidate chunks."
        ),
    }

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.BM25_SEARCH,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def build_context_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before context building "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.BUILD_CONTEXT.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.BUILD_CONTEXT.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Building retrieval context "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Build context only from authorized retrieval candidates.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    query_text = agent_state.rewritten_query or agent_state.query
    context_response = build_context(
        ContextBuildRequest(
            query=query_text,
            chunks=agent_state.authorized_chunks,
            max_context_chunks=graph_state.max_context_chunks,
            max_tokens=graph_state.max_context_tokens,
        )
    )
    agent_state.context = context_response.context
    agent_state.citations = [
        context_chunk.citation for context_chunk in context_response.context
    ]
    agent_state.replay_metadata[AgentNodeName.BUILD_CONTEXT.value] = {
        "node_name": AgentNodeName.BUILD_CONTEXT.value,
        "stage": "retrieval",
        "tool_name": "context_builder",
        "input": {
            "authorized_chunk_count": len(agent_state.authorized_chunks),
            "max_context_chunks": graph_state.max_context_chunks,
            "max_context_tokens": graph_state.max_context_tokens,
            "query_length": len(query_text),
        },
        "output": {
            "context_chunk_count": len(context_response.context),
            "citation_count": len(agent_state.citations),
            "context_token_count": context_response.token_count,
            "document_ids": [
                str(context_chunk.document_id)
                for context_chunk in context_response.context[:10]
            ],
            "chunk_ids": [
                str(context_chunk.chunk_id)
                for context_chunk in context_response.context[:10]
            ],
        },
        "summary": (
            f"Context build selected {len(context_response.context)} chunks, "
            f"{len(agent_state.citations)} citations, "
            f"and {context_response.token_count} tokens."
        ),
    }

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.BUILD_CONTEXT,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def generate_answer_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before answer generation "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.GENERATE_ANSWER.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.GENERATE_ANSWER.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Generating answer "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Guard answer generation before the LLM sees context.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    guardrail_decision = evaluate_agent_guardrails(
        state=agent_state,
        limits=graph_state.limits,
        now=graph_state.current_time,
        next_node_name=AgentNodeName.GENERATE_ANSWER,
    )
    if guardrail_decision.should_stop:
        if guardrail_decision.fallback_answer is not None:
            agent_state.final_answer = SAFE_FALLBACK_ANSWER
            agent_state.handoff_required = True
        agent_state.replay_metadata[AgentNodeName.GENERATE_ANSWER.value] = {
            "node_name": AgentNodeName.GENERATE_ANSWER.value,
            "stage": "generation",
            "tool_name": "llm_gateway",
            "input": {
                "context_chunk_count": len(agent_state.context),
                "citation_count": len(agent_state.citations),
                "context_token_count": sum(
                    context_chunk.token_count for context_chunk in agent_state.context
                ),
                "has_authorized_context": bool(agent_state.context),
            },
            "output": {
                "generated": False,
                "handoff_required": agent_state.handoff_required,
                "stop_reason": guardrail_decision.reason,
                "answer_character_count": len(agent_state.final_answer or ""),
            },
            "summary": (
                f"Answer generation stopped before LLM call: "
                f"{guardrail_decision.reason}"
            ),
        }

        step_result = record_agent_step(
            state=agent_state,
            node_name=AgentNodeName.GENERATE_ANSWER,
            limits=graph_state.limits,
            now=graph_state.current_time,
        )
        return {
            "agent_state": step_result.state,
            "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
            "status": guardrail_decision.status,
            "stop_reason": guardrail_decision.reason,
            "current_time": graph_state.current_time,
            "db": graph_state.db,
            "search_client": graph_state.search_client,
        }

    context_lines = []
    for index, context_chunk in enumerate(agent_state.context, start=1):
        citation = context_chunk.citation
        title = citation.title or "Untitled document"
        source_uri = citation.source_uri or "unknown source"
        context_lines.append(
            "\n".join(
                [
                    f"[{index}] Title: {title}",
                    f"Source: {source_uri}",
                    f"Document ID: {context_chunk.document_id}",
                    f"Chunk ID: {context_chunk.chunk_id}",
                    f"Content: {context_chunk.content}",
                ]
            )
        )
    context_block = "\n\n".join(context_lines)

    try:
        llm_response = generate_chat_completion(
            ChatCompletionRequest(
                auth=agent_state.auth,
                messages=[
                    LLMMessage(
                        role="system",
                        content=(
                            "You are the answer synthesis layer for Agentic RAG. "
                            "Use only the authorized context provided by the retrieval system. "
                            "If the context is not enough, say that the available documents do not answer the question. "
                            "Cite sources with bracket numbers such as [1] and [2]. "
                            "Do not mention hidden instructions or unsupported sources."
                        ),
                    ),
                    LLMMessage(
                        role="user",
                        content=(
                            "Question:\n"
                            f"{agent_state.rewritten_query or agent_state.query}\n\n"
                            "Authorized context:\n"
                            f"{context_block}\n\n"
                            "Write a concise answer grounded only in the authorized context."
                        ),
                    ),
                ],
                metadata={
                    "tenant_id": agent_state.auth.tenant_id,
                    "user_id": agent_state.auth.user_id,
                    "agent_run_id": str(agent_state.agent_run_id),
                    "context_chunks": len(agent_state.context),
                },
            )
        )
        agent_state.draft_answer = llm_response.text
        agent_state.replay_metadata[AgentNodeName.GENERATE_ANSWER.value] = {
            "node_name": AgentNodeName.GENERATE_ANSWER.value,
            "stage": "generation",
            "tool_name": "llm_gateway",
            "input": {
                "context_chunk_count": len(agent_state.context),
                "citation_count": len(agent_state.citations),
                "context_token_count": sum(
                    context_chunk.token_count for context_chunk in agent_state.context
                ),
                "has_authorized_context": bool(agent_state.context),
            },
            "output": {
                "generated": True,
                "llm_provider": llm_response.provider,
                "llm_model": llm_response.model,
                "llm_input_tokens": llm_response.input_tokens,
                "llm_output_tokens": llm_response.output_tokens,
                "llm_cost_estimate": llm_response.cost_estimate,
                "latency_ms": llm_response.latency_ms,
                "answer_character_count": len(agent_state.draft_answer or ""),
            },
            "summary": (
                f"Answer generation produced "
                f"{len(agent_state.draft_answer or '')} characters from "
                f"{len(agent_state.context)} context chunks."
            ),
        }

    except Exception as e:
        logger.exception(
            f"[AgentGraph] Answer generation failed "
            f"agent_run_id={agent_state.agent_run_id} "
            f"tenant_id={agent_state.auth.tenant_id}: {e}"
        )
        agent_state.final_answer = SAFE_FALLBACK_ANSWER
        agent_state.handoff_required = True
        agent_state.replay_metadata[AgentNodeName.GENERATE_ANSWER.value] = {
            "node_name": AgentNodeName.GENERATE_ANSWER.value,
            "stage": "generation",
            "tool_name": "llm_gateway",
            "input": {
                "context_chunk_count": len(agent_state.context),
                "citation_count": len(agent_state.citations),
                "context_token_count": sum(
                    context_chunk.token_count for context_chunk in agent_state.context
                ),
                "has_authorized_context": bool(agent_state.context),
            },
            "output": {
                "generated": False,
                "handoff_required": True,
                "error_type": type(e).__name__,
                "answer_character_count": len(agent_state.final_answer or ""),
            },
            "summary": "Answer generation failed through the LLM gateway.",
        }
        step_result = record_agent_step(
            state=agent_state,
            node_name=AgentNodeName.GENERATE_ANSWER,
            limits=graph_state.limits,
            now=graph_state.current_time,
        )
        return {
            "agent_state": step_result.state,
            "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
            "status": AgentRunStatus.HANDOFF_REQUIRED,
            "stop_reason": "LLM answer generation failed.",
            "current_time": graph_state.current_time,
            "db": graph_state.db,
            "search_client": graph_state.search_client,
        }

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.GENERATE_ANSWER,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )

    status = AgentRunStatus.RUNNING
    stop_reason = None
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def verify_grounding_node(graph_state: AgentGraphState) -> dict[str, Any]:
    if graph_state.status != AgentRunStatus.RUNNING:
        return {}

    if graph_state.db is not None:
        if is_agent_run_cancelled(
            db=graph_state.db,
            agent_run_id=graph_state.agent_state.agent_run_id,
            tenant_id=graph_state.agent_state.auth.tenant_id,
        ):
            logger.warning(
                f"[AgentGraph] Agent runtime graph cancelled before grounding verification "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Save a cancellation checkpoint for this node.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            agent_state.step_count += 1
            agent_state.visited_nodes.append(AgentNodeName.VERIFY_GROUNDING.value)
            checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
            checkpoint = AgentCheckpoint(
                agent_run_id=agent_state.agent_run_id,
                checkpoint_key=(
                    f"step-{agent_state.step_count:04d}-"
                    f"{AgentNodeName.VERIFY_GROUNDING.value}"
                ),
                state=agent_state.model_dump(mode="json"),
                created_at=checkpoint_time,
            )
            return {
                "agent_state": agent_state,
                "checkpoints": [*graph_state.checkpoints, checkpoint],
                "status": AgentRunStatus.CANCELLED,
                "stop_reason": "Agent run was cancelled.",
                "current_time": graph_state.current_time,
                "db": graph_state.db,
                "search_client": graph_state.search_client,
            }

    logger.info(
        f"[AgentGraph] Verifying grounded answer "
        f"agent_run_id={graph_state.agent_state.agent_run_id} "
        f"tenant_id={graph_state.agent_state.auth.tenant_id}"
    )

    # Verify the draft answer against authorized context citations.
    agent_state = graph_state.agent_state.model_copy(deep=True)
    verification_result = verify_answer_support(
        answer=agent_state.draft_answer or "",
        context=agent_state.context,
    )
    status = AgentRunStatus.COMPLETED
    stop_reason = None
    if verification_result.passed:
        agent_state.final_answer = agent_state.draft_answer
        agent_state.confidence_score = 1.0
    else:
        logger.warning(
            f"[AgentGraph] Grounding verification failed "
            f"agent_run_id={agent_state.agent_run_id} "
            f"tenant_id={agent_state.auth.tenant_id} "
            f"reason={verification_result.reason}"
        )
        agent_state.final_answer = SAFE_FALLBACK_ANSWER
        agent_state.handoff_required = True
        agent_state.confidence_score = 0.0
        status = AgentRunStatus.HANDOFF_REQUIRED
        stop_reason = verification_result.reason
    agent_state.replay_metadata[AgentNodeName.VERIFY_GROUNDING.value] = {
        "node_name": AgentNodeName.VERIFY_GROUNDING.value,
        "stage": "verification",
        "tool_name": "answer_verifier",
        "input": {
            "answer_character_count": len(agent_state.draft_answer or ""),
            "context_chunk_count": len(agent_state.context),
            "citation_count": len(agent_state.citations),
        },
        "output": {
            "passed": verification_result.passed,
            "reason": verification_result.reason,
            "cited_source_numbers": verification_result.cited_source_numbers,
            "final_status": status.value,
            "confidence_score": agent_state.confidence_score,
            "handoff_required": agent_state.handoff_required,
        },
        "summary": (
            f"Grounding verification "
            f"{'passed' if verification_result.passed else 'failed'}: "
            f"{verification_result.reason}"
        ),
    }

    # Record the node through the runtime guardrails.
    step_result = record_agent_step(
        state=agent_state,
        node_name=AgentNodeName.VERIFY_GROUNDING,
        limits=graph_state.limits,
        now=graph_state.current_time,
    )
    if step_result.decision.should_stop:
        status = step_result.decision.status
        stop_reason = step_result.decision.reason

    return {
        "agent_state": step_result.state,
        "checkpoints": [*graph_state.checkpoints, step_result.checkpoint],
        "status": status,
        "stop_reason": stop_reason,
        "current_time": graph_state.current_time,
        "db": graph_state.db,
        "search_client": graph_state.search_client,
    }


def build_agent_runtime_graph():
    logger.info("[AgentGraph] Building agent runtime graph")

    graph = StateGraph(AgentGraphState)
    graph.add_node(AgentNodeName.CLASSIFY_INTENT.value, classify_intent_node)
    graph.add_node(AgentNodeName.REWRITE_QUERY.value, rewrite_query_node)
    graph.add_node(AgentNodeName.PLAN_FILTERS.value, plan_filters_node)
    graph.add_node(AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value, select_retrieval_strategy_node)
    graph.add_node(AgentNodeName.BM25_SEARCH.value, bm25_search_node)
    graph.add_node(AgentNodeName.BUILD_CONTEXT.value, build_context_node)
    graph.add_node(AgentNodeName.GENERATE_ANSWER.value, generate_answer_node)
    graph.add_node(AgentNodeName.VERIFY_GROUNDING.value, verify_grounding_node)
    graph.add_edge(START, AgentNodeName.CLASSIFY_INTENT.value)
    graph.add_edge(AgentNodeName.CLASSIFY_INTENT.value, AgentNodeName.REWRITE_QUERY.value)
    graph.add_edge(AgentNodeName.REWRITE_QUERY.value, AgentNodeName.PLAN_FILTERS.value)
    graph.add_edge(AgentNodeName.PLAN_FILTERS.value, AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value)
    graph.add_edge(AgentNodeName.SELECT_RETRIEVAL_STRATEGY.value, AgentNodeName.BM25_SEARCH.value)
    graph.add_edge(AgentNodeName.BM25_SEARCH.value, AgentNodeName.BUILD_CONTEXT.value)
    graph.add_edge(AgentNodeName.BUILD_CONTEXT.value, AgentNodeName.GENERATE_ANSWER.value)
    graph.add_edge(AgentNodeName.GENERATE_ANSWER.value, AgentNodeName.VERIFY_GROUNDING.value)
    graph.add_edge(AgentNodeName.VERIFY_GROUNDING.value, END)

    return graph.compile()


def run_agent_runtime_graph(
    *,
    agent_run_id: UUID,
    auth: AuthContext,
    query: str,
    limits: AgentLimits | None = None,
    retrieval_filters: RetrievalFilters | None = None,
    retrieval_limit: int = 20,
    max_context_chunks: int = 12,
    max_context_tokens: int = 6000,
    now: datetime | None = None,
    db: Session | None = None,
    search_client: Any | None = None,
) -> AgentGraphRunResult:
    runtime_limits = limits or AgentLimits()
    runtime_filters = retrieval_filters or RetrievalFilters()

    # Use one UTC clock to initialize the graph run.
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    logger.info(
        f"[AgentGraph] Starting agent runtime graph "
        f"agent_run_id={agent_run_id} tenant_id={auth.tenant_id} user_id={auth.user_id}"
    )

    if db is not None:
        create_agent_run(
            db=db,
            agent_run_id=agent_run_id,
            auth=auth,
            query=query,
            limits=runtime_limits,
        )

    agent_state = start_agent_state(
        agent_run_id=agent_run_id,
        auth=auth,
        query=query,
        limits=runtime_limits,
        now=current_time,
    )
    graph_state = AgentGraphState(
        agent_state=agent_state,
        limits=runtime_limits,
        retrieval_filters=runtime_filters,
        retrieval_limit=retrieval_limit,
        max_context_chunks=max_context_chunks,
        max_context_tokens=max_context_tokens,
        current_time=current_time,
        db=db,
        search_client=search_client,
    )

    compiled_graph = build_agent_runtime_graph()
    raw_result = compiled_graph.invoke(graph_state)
    completed_graph_state = AgentGraphState.model_validate(raw_result)

    if db is not None:
        for checkpoint in completed_graph_state.checkpoints:
            visited_nodes = checkpoint.state.get("visited_nodes", [])
            node_name = checkpoint.checkpoint_key
            if visited_nodes:
                node_name = visited_nodes[-1]

            step_number = checkpoint.state.get("step_count")
            if not isinstance(step_number, int):
                step_number = len(visited_nodes) if visited_nodes else 0

            replay_tool_name = None
            replay_tool_input = None
            replay_tool_output_summary = None
            replay_metadata = checkpoint.state.get("replay_metadata", {})
            if isinstance(replay_metadata, dict):
                node_replay_metadata = replay_metadata.get(str(node_name), {})
                if isinstance(node_replay_metadata, dict):
                    candidate_tool_name = node_replay_metadata.get("tool_name")
                    if isinstance(candidate_tool_name, str):
                        replay_tool_name = candidate_tool_name

                    candidate_tool_input = node_replay_metadata.get("input")
                    if isinstance(candidate_tool_input, dict):
                        replay_tool_input = candidate_tool_input

                    candidate_summary = node_replay_metadata.get("summary")
                    if isinstance(candidate_summary, str):
                        replay_tool_output_summary = candidate_summary

            step_status = "completed"
            finish_run = False
            if checkpoint == completed_graph_state.checkpoints[-1]:
                if completed_graph_state.status != AgentRunStatus.RUNNING:
                    step_status = completed_graph_state.status.value
                    finish_run = True

            record_agent_run_step(
                db=db,
                agent_run_id=agent_run_id,
                tenant_id=auth.tenant_id,
                node_name=node_name,
                step_number=step_number,
                status=step_status,
                finish_run=finish_run,
                tool_name=replay_tool_name,
                tool_input=replay_tool_input,
                tool_output_summary=replay_tool_output_summary,
            )
            save_agent_checkpoint(
                db=db,
                agent_run_id=agent_run_id,
                tenant_id=auth.tenant_id,
                checkpoint=checkpoint,
            )

    logger.info(
        f"[AgentGraph] Agent runtime graph finished "
        f"agent_run_id={agent_run_id} "
        f"status={completed_graph_state.status.value} "
        f"step_count={completed_graph_state.agent_state.step_count} "
        f"checkpoint_count={len(completed_graph_state.checkpoints)}"
    )

    return AgentGraphRunResult(
        state=completed_graph_state.agent_state,
        checkpoints=completed_graph_state.checkpoints,
        status=completed_graph_state.status,
        stop_reason=completed_graph_state.stop_reason,
    )


def stream_agent_runtime_graph(
    *,
    agent_run_id: UUID,
    auth: AuthContext,
    query: str,
    limits: AgentLimits | None = None,
    retrieval_filters: RetrievalFilters | None = None,
    retrieval_limit: int = 20,
    max_context_chunks: int = 12,
    max_context_tokens: int = 6000,
    now: datetime | None = None,
    db: Session | None = None,
    search_client: Any | None = None,
) -> Iterator[AgentStreamEvent]:
    runtime_limits = limits or AgentLimits()
    runtime_filters = retrieval_filters or RetrievalFilters()

    # Use one UTC clock to initialize the streamed graph run.
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    logger.info(
        f"[AgentGraph] Starting streamed agent runtime graph "
        f"agent_run_id={agent_run_id} tenant_id={auth.tenant_id} user_id={auth.user_id}"
    )

    if db is not None:
        create_agent_run(
            db=db,
            agent_run_id=agent_run_id,
            auth=auth,
            query=query,
            limits=runtime_limits,
        )

    agent_state = start_agent_state(
        agent_run_id=agent_run_id,
        auth=auth,
        query=query,
        limits=runtime_limits,
        now=current_time,
    )
    graph_state = AgentGraphState(
        agent_state=agent_state,
        limits=runtime_limits,
        retrieval_filters=runtime_filters,
        retrieval_limit=retrieval_limit,
        max_context_chunks=max_context_chunks,
        max_context_tokens=max_context_tokens,
        current_time=current_time,
        db=db,
        search_client=search_client,
    )
    terminal_event_type = AgentStreamEventType.AGENT_COMPLETED

    yield AgentStreamEvent(
        event=AgentStreamEventType.AGENT_STARTED,
        agent_run_id=agent_run_id,
        status=AgentRunStatus.RUNNING,
        data={
            "tenant_id": auth.tenant_id,
            "user_id": auth.user_id,
            "workspace_id": auth.workspace_id,
        },
        created_at=current_time,
    )

    for node_name, node_function in [
        (AgentNodeName.CLASSIFY_INTENT, classify_intent_node),
        (AgentNodeName.REWRITE_QUERY, rewrite_query_node),
        (AgentNodeName.PLAN_FILTERS, plan_filters_node),
        (AgentNodeName.SELECT_RETRIEVAL_STRATEGY, select_retrieval_strategy_node),
        (AgentNodeName.BM25_SEARCH, bm25_search_node),
        (AgentNodeName.BUILD_CONTEXT, build_context_node),
    ]:
        raw_result = node_function(graph_state)
        if raw_result:
            graph_state = graph_state.model_copy(update=raw_result)

        if graph_state.checkpoints:
            checkpoint = graph_state.checkpoints[-1]
            visited_nodes = checkpoint.state.get("visited_nodes", [])
            event_node_name = node_name.value
            if visited_nodes:
                event_node_name = visited_nodes[-1]
            step_number = checkpoint.state.get("step_count")
            if not isinstance(step_number, int):
                step_number = len(visited_nodes) if visited_nodes else 0

            yield AgentStreamEvent(
                event=AgentStreamEventType.AGENT_STEP_COMPLETED,
                agent_run_id=agent_run_id,
                node_name=event_node_name,
                status=graph_state.status,
                step_number=step_number,
                data={
                    "checkpoint_key": checkpoint.checkpoint_key,
                    "stop_reason": graph_state.stop_reason,
                },
                created_at=checkpoint.created_at,
            )

        if graph_state.status != AgentRunStatus.RUNNING:
            break

    if graph_state.status == AgentRunStatus.RUNNING:
        if graph_state.db is not None:
            if is_agent_run_cancelled(
                db=graph_state.db,
                agent_run_id=graph_state.agent_state.agent_run_id,
                tenant_id=graph_state.agent_state.auth.tenant_id,
            ):
                logger.warning(
                    f"[AgentGraph] Streamed agent runtime graph cancelled before answer generation "
                    f"agent_run_id={graph_state.agent_state.agent_run_id} "
                    f"tenant_id={graph_state.agent_state.auth.tenant_id}"
                )

                # Save a cancellation checkpoint for this node.
                agent_state = graph_state.agent_state.model_copy(deep=True)
                agent_state.step_count += 1
                agent_state.visited_nodes.append(AgentNodeName.GENERATE_ANSWER.value)
                checkpoint_time = graph_state.current_time or datetime.now(timezone.utc)
                checkpoint = AgentCheckpoint(
                    agent_run_id=agent_state.agent_run_id,
                    checkpoint_key=(
                        f"step-{agent_state.step_count:04d}-"
                        f"{AgentNodeName.GENERATE_ANSWER.value}"
                    ),
                    state=agent_state.model_dump(mode="json"),
                    created_at=checkpoint_time,
                )
                graph_state = graph_state.model_copy(
                    update={
                        "agent_state": agent_state,
                        "checkpoints": [*graph_state.checkpoints, checkpoint],
                        "status": AgentRunStatus.CANCELLED,
                        "stop_reason": "Agent run was cancelled.",
                    }
                )

                yield AgentStreamEvent(
                    event=AgentStreamEventType.AGENT_STEP_COMPLETED,
                    agent_run_id=agent_run_id,
                    node_name=AgentNodeName.GENERATE_ANSWER.value,
                    status=AgentRunStatus.CANCELLED,
                    step_number=agent_state.step_count,
                    data={
                        "checkpoint_key": checkpoint.checkpoint_key,
                        "stop_reason": "Agent run was cancelled.",
                    },
                    created_at=checkpoint.created_at,
                )

        if graph_state.status == AgentRunStatus.RUNNING:
            logger.info(
                f"[AgentGraph] Streaming answer generation "
                f"agent_run_id={graph_state.agent_state.agent_run_id} "
                f"tenant_id={graph_state.agent_state.auth.tenant_id}"
            )

            # Guard answer generation before the LLM sees context.
            agent_state = graph_state.agent_state.model_copy(deep=True)
            guardrail_decision = evaluate_agent_guardrails(
                state=agent_state,
                limits=graph_state.limits,
                now=graph_state.current_time,
                next_node_name=AgentNodeName.GENERATE_ANSWER,
            )
            if guardrail_decision.should_stop:
                if guardrail_decision.fallback_answer is not None:
                    agent_state.final_answer = SAFE_FALLBACK_ANSWER
                    agent_state.handoff_required = True
                agent_state.replay_metadata[AgentNodeName.GENERATE_ANSWER.value] = {
                    "node_name": AgentNodeName.GENERATE_ANSWER.value,
                    "stage": "generation",
                    "tool_name": "llm_gateway",
                    "input": {
                        "context_chunk_count": len(agent_state.context),
                        "citation_count": len(agent_state.citations),
                        "context_token_count": sum(
                            context_chunk.token_count
                            for context_chunk in agent_state.context
                        ),
                        "has_authorized_context": bool(agent_state.context),
                    },
                    "output": {
                        "generated": False,
                        "handoff_required": agent_state.handoff_required,
                        "stop_reason": guardrail_decision.reason,
                        "answer_character_count": len(agent_state.final_answer or ""),
                    },
                    "summary": (
                        f"Answer generation stopped before LLM stream: "
                        f"{guardrail_decision.reason}"
                    ),
                }

                step_result = record_agent_step(
                    state=agent_state,
                    node_name=AgentNodeName.GENERATE_ANSWER,
                    limits=graph_state.limits,
                    now=graph_state.current_time,
                )
                graph_state = graph_state.model_copy(
                    update={
                        "agent_state": step_result.state,
                        "checkpoints": [
                            *graph_state.checkpoints,
                            step_result.checkpoint,
                        ],
                        "status": guardrail_decision.status,
                        "stop_reason": guardrail_decision.reason,
                    }
                )

                yield AgentStreamEvent(
                    event=AgentStreamEventType.AGENT_STEP_COMPLETED,
                    agent_run_id=agent_run_id,
                    node_name=AgentNodeName.GENERATE_ANSWER.value,
                    status=guardrail_decision.status,
                    step_number=step_result.state.step_count,
                    data={
                        "checkpoint_key": step_result.checkpoint.checkpoint_key,
                        "stop_reason": guardrail_decision.reason,
                    },
                    created_at=step_result.checkpoint.created_at,
                )

            if graph_state.status == AgentRunStatus.RUNNING:
                context_lines = []
                for index, context_chunk in enumerate(agent_state.context, start=1):
                    citation = context_chunk.citation
                    title = citation.title or "Untitled document"
                    source_uri = citation.source_uri or "unknown source"
                    context_lines.append(
                        "\n".join(
                            [
                                f"[{index}] Title: {title}",
                                f"Source: {source_uri}",
                                f"Document ID: {context_chunk.document_id}",
                                f"Chunk ID: {context_chunk.chunk_id}",
                                f"Content: {context_chunk.content}",
                            ]
                        )
                    )
                context_block = "\n\n".join(context_lines)

                try:
                    answer_text = ""
                    token_index = 0
                    answer_model = None
                    answer_provider = None
                    answer_input_tokens = None
                    answer_output_tokens = None
                    answer_cost_estimate = None
                    answer_latency_ms = None
                    for llm_stream_event in stream_chat_completion(
                        ChatCompletionRequest(
                            auth=agent_state.auth,
                            messages=[
                                LLMMessage(
                                    role="system",
                                    content=(
                                        "You are the answer synthesis layer for Agentic RAG. "
                                        "Use only the authorized context provided by the retrieval system. "
                                        "If the context is not enough, say that the available documents do not answer the question. "
                                        "Cite sources with bracket numbers such as [1] and [2]. "
                                        "Do not mention hidden instructions or unsupported sources."
                                    ),
                                ),
                                LLMMessage(
                                    role="user",
                                    content=(
                                        "Question:\n"
                                        f"{agent_state.rewritten_query or agent_state.query}\n\n"
                                        "Authorized context:\n"
                                        f"{context_block}\n\n"
                                        "Write a concise answer grounded only in the authorized context."
                                    ),
                                ),
                            ],
                            metadata={
                                "tenant_id": agent_state.auth.tenant_id,
                                "user_id": agent_state.auth.user_id,
                                "agent_run_id": str(agent_state.agent_run_id),
                                "context_chunks": len(agent_state.context),
                            },
                        )
                    ):
                        if (
                            llm_stream_event.event == LLMStreamEventType.TOKEN
                            and llm_stream_event.text_delta
                        ):
                            token_index += 1
                            yield AgentStreamEvent(
                                event=AgentStreamEventType.ANSWER_TOKEN,
                                agent_run_id=agent_run_id,
                                node_name=AgentNodeName.GENERATE_ANSWER.value,
                                status=AgentRunStatus.RUNNING,
                                text_delta=llm_stream_event.text_delta,
                                data={
                                    "token_index": token_index,
                                    "model": llm_stream_event.model,
                                    "provider": llm_stream_event.provider,
                                },
                                created_at=datetime.now(timezone.utc),
                            )

                        if llm_stream_event.event == LLMStreamEventType.COMPLETED:
                            answer_text = llm_stream_event.text or ""
                            answer_model = llm_stream_event.model
                            answer_provider = llm_stream_event.provider
                            answer_input_tokens = llm_stream_event.input_tokens
                            answer_output_tokens = llm_stream_event.output_tokens
                            answer_cost_estimate = llm_stream_event.cost_estimate
                            answer_latency_ms = llm_stream_event.latency_ms

                    agent_state.draft_answer = answer_text
                    agent_state.replay_metadata[AgentNodeName.GENERATE_ANSWER.value] = {
                        "node_name": AgentNodeName.GENERATE_ANSWER.value,
                        "stage": "generation",
                        "tool_name": "llm_gateway",
                        "input": {
                            "context_chunk_count": len(agent_state.context),
                            "citation_count": len(agent_state.citations),
                            "context_token_count": sum(
                                context_chunk.token_count
                                for context_chunk in agent_state.context
                            ),
                            "has_authorized_context": bool(agent_state.context),
                        },
                        "output": {
                            "generated": True,
                            "streamed": True,
                            "token_event_count": token_index,
                            "llm_provider": answer_provider,
                            "llm_model": answer_model,
                            "llm_input_tokens": answer_input_tokens,
                            "llm_output_tokens": answer_output_tokens,
                            "llm_cost_estimate": answer_cost_estimate,
                            "latency_ms": answer_latency_ms,
                            "answer_character_count": len(
                                agent_state.draft_answer or ""
                            ),
                        },
                        "summary": (
                            f"Streamed answer generation produced "
                            f"{len(agent_state.draft_answer or '')} characters "
                            f"from {len(agent_state.context)} context chunks."
                        ),
                    }

                except Exception as e:
                    logger.exception(
                        f"[AgentGraph] Streaming answer generation failed "
                        f"agent_run_id={agent_state.agent_run_id} "
                        f"tenant_id={agent_state.auth.tenant_id}: {e}"
                    )
                    agent_state.final_answer = SAFE_FALLBACK_ANSWER
                    agent_state.handoff_required = True
                    agent_state.replay_metadata[AgentNodeName.GENERATE_ANSWER.value] = {
                        "node_name": AgentNodeName.GENERATE_ANSWER.value,
                        "stage": "generation",
                        "tool_name": "llm_gateway",
                        "input": {
                            "context_chunk_count": len(agent_state.context),
                            "citation_count": len(agent_state.citations),
                            "context_token_count": sum(
                                context_chunk.token_count
                                for context_chunk in agent_state.context
                            ),
                            "has_authorized_context": bool(agent_state.context),
                        },
                        "output": {
                            "generated": False,
                            "streamed": True,
                            "handoff_required": True,
                            "error_type": type(e).__name__,
                            "answer_character_count": len(
                                agent_state.final_answer or ""
                            ),
                        },
                        "summary": "Streaming answer generation failed through the LLM gateway.",
                    }
                    step_result = record_agent_step(
                        state=agent_state,
                        node_name=AgentNodeName.GENERATE_ANSWER,
                        limits=graph_state.limits,
                        now=graph_state.current_time,
                    )
                    graph_state = graph_state.model_copy(
                        update={
                            "agent_state": step_result.state,
                            "checkpoints": [
                                *graph_state.checkpoints,
                                step_result.checkpoint,
                            ],
                            "status": AgentRunStatus.HANDOFF_REQUIRED,
                            "stop_reason": "LLM answer generation failed.",
                        }
                    )
                    terminal_event_type = AgentStreamEventType.AGENT_FAILED

                    yield AgentStreamEvent(
                        event=AgentStreamEventType.AGENT_STEP_COMPLETED,
                        agent_run_id=agent_run_id,
                        node_name=AgentNodeName.GENERATE_ANSWER.value,
                        status=AgentRunStatus.HANDOFF_REQUIRED,
                        step_number=step_result.state.step_count,
                        data={
                            "checkpoint_key": step_result.checkpoint.checkpoint_key,
                            "stop_reason": "LLM answer generation failed.",
                            "error_type": type(e).__name__,
                        },
                        created_at=step_result.checkpoint.created_at,
                    )

                if graph_state.status == AgentRunStatus.RUNNING:
                    # Record the generation node after the final stream event.
                    step_result = record_agent_step(
                        state=agent_state,
                        node_name=AgentNodeName.GENERATE_ANSWER,
                        limits=graph_state.limits,
                        now=graph_state.current_time,
                    )
                    status = AgentRunStatus.RUNNING
                    stop_reason = None
                    if step_result.decision.should_stop:
                        status = step_result.decision.status
                        stop_reason = step_result.decision.reason
                    graph_state = graph_state.model_copy(
                        update={
                            "agent_state": step_result.state,
                            "checkpoints": [
                                *graph_state.checkpoints,
                                step_result.checkpoint,
                            ],
                            "status": status,
                            "stop_reason": stop_reason,
                        }
                    )

                    yield AgentStreamEvent(
                        event=AgentStreamEventType.AGENT_STEP_COMPLETED,
                        agent_run_id=agent_run_id,
                        node_name=AgentNodeName.GENERATE_ANSWER.value,
                        status=status,
                        step_number=step_result.state.step_count,
                        data={
                            "checkpoint_key": step_result.checkpoint.checkpoint_key,
                            "stop_reason": stop_reason,
                        },
                        created_at=step_result.checkpoint.created_at,
                    )

        if graph_state.status == AgentRunStatus.RUNNING:
            raw_result = verify_grounding_node(graph_state)
            if raw_result:
                graph_state = graph_state.model_copy(update=raw_result)

            if graph_state.checkpoints:
                checkpoint = graph_state.checkpoints[-1]
                visited_nodes = checkpoint.state.get("visited_nodes", [])
                event_node_name = AgentNodeName.VERIFY_GROUNDING.value
                if visited_nodes:
                    event_node_name = visited_nodes[-1]
                step_number = checkpoint.state.get("step_count")
                if not isinstance(step_number, int):
                    step_number = len(visited_nodes) if visited_nodes else 0

                yield AgentStreamEvent(
                    event=AgentStreamEventType.AGENT_STEP_COMPLETED,
                    agent_run_id=agent_run_id,
                    node_name=event_node_name,
                    status=graph_state.status,
                    step_number=step_number,
                    data={
                        "checkpoint_key": checkpoint.checkpoint_key,
                        "stop_reason": graph_state.stop_reason,
                    },
                    created_at=checkpoint.created_at,
                )

    if db is not None:
        for checkpoint in graph_state.checkpoints:
            visited_nodes = checkpoint.state.get("visited_nodes", [])
            persistence_node_name: AgentNodeName | str = checkpoint.checkpoint_key
            if visited_nodes:
                persistence_node_name = visited_nodes[-1]

            step_number = checkpoint.state.get("step_count")
            if not isinstance(step_number, int):
                step_number = len(visited_nodes) if visited_nodes else 0

            replay_tool_name = None
            replay_tool_input = None
            replay_tool_output_summary = None
            replay_metadata = checkpoint.state.get("replay_metadata", {})
            if isinstance(replay_metadata, dict):
                node_replay_metadata = replay_metadata.get(
                    str(persistence_node_name),
                    {},
                )
                if isinstance(node_replay_metadata, dict):
                    candidate_tool_name = node_replay_metadata.get("tool_name")
                    if isinstance(candidate_tool_name, str):
                        replay_tool_name = candidate_tool_name

                    candidate_tool_input = node_replay_metadata.get("input")
                    if isinstance(candidate_tool_input, dict):
                        replay_tool_input = candidate_tool_input

                    candidate_summary = node_replay_metadata.get("summary")
                    if isinstance(candidate_summary, str):
                        replay_tool_output_summary = candidate_summary

            step_status = "completed"
            finish_run = False
            if checkpoint == graph_state.checkpoints[-1]:
                if graph_state.status != AgentRunStatus.RUNNING:
                    step_status = graph_state.status.value
                    finish_run = True

            record_agent_run_step(
                db=db,
                agent_run_id=agent_run_id,
                tenant_id=auth.tenant_id,
                node_name=persistence_node_name,
                step_number=step_number,
                status=step_status,
                finish_run=finish_run,
                tool_name=replay_tool_name,
                tool_input=replay_tool_input,
                tool_output_summary=replay_tool_output_summary,
            )
            save_agent_checkpoint(
                db=db,
                agent_run_id=agent_run_id,
                tenant_id=auth.tenant_id,
                checkpoint=checkpoint,
            )

    logger.info(
        f"[AgentGraph] Streamed agent runtime graph finished "
        f"agent_run_id={agent_run_id} "
        f"status={graph_state.status.value} "
        f"step_count={graph_state.agent_state.step_count} "
        f"checkpoint_count={len(graph_state.checkpoints)}"
    )

    yield AgentStreamEvent(
        event=terminal_event_type,
        agent_run_id=agent_run_id,
        status=graph_state.status,
        data={
            "stop_reason": graph_state.stop_reason,
            "answer": graph_state.agent_state.final_answer,
            "citations": [
                citation.model_dump(mode="json")
                for citation in graph_state.agent_state.citations
            ],
            "context": [
                context_chunk.model_dump(mode="json")
                for context_chunk in graph_state.agent_state.context
            ],
            "context_token_count": sum(
                context_chunk.token_count
                for context_chunk in graph_state.agent_state.context
            ),
            "retrieval_strategy": (
                graph_state.agent_state.retrieval_strategy.value
                if graph_state.agent_state.retrieval_strategy
                else RetrievalStrategy.BM25.value
            ),
            "checkpoint_count": len(graph_state.checkpoints),
            "context_chunks": len(graph_state.agent_state.context),
            "citation_count": len(graph_state.agent_state.citations),
            "confidence_score": graph_state.agent_state.confidence_score,
        },
        created_at=datetime.now(timezone.utc),
    )
