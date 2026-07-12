from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import agentic_rag.workers.email as email_worker_module
from agentic_rag.shared.db.base import Base
from agentic_rag.shared.db.crud.identity import encrypt_outbox_value
from agentic_rag.shared.db.models import EmailOutbox
from agentic_rag.workers.email import process_next_email


class FakeSMTP:
    messages: list[EmailMessage] = []

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def starttls(self) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        return None

    def send_message(self, message: EmailMessage) -> None:
        self.messages.append(message)


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def email_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSMTP.messages.clear()
    monkeypatch.setattr(email_worker_module.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_worker_module.settings, "email_delivery_provider", "smtp")
    monkeypatch.setattr(email_worker_module.settings, "smtp_host", "mailpit")
    monkeypatch.setattr(email_worker_module.settings, "smtp_port", 1025)
    monkeypatch.setattr(email_worker_module.settings, "smtp_use_tls", False)
    monkeypatch.setattr(email_worker_module.settings, "smtp_username", "")
    monkeypatch.setattr(email_worker_module.settings, "email_worker_max_attempts", 3)
    monkeypatch.setattr(email_worker_module.settings, "email_worker_lease_seconds", 60)
    monkeypatch.setattr(
        email_worker_module.settings,
        "invitation_context_secret",
        "test-invitation-encryption-secret-with-sufficient-length",
    )


def test_process_invitation_resend_sends_and_marks_outbox_sent(db: Session) -> None:
    acceptance_url = "http://localhost:5173/invite?token=single-use-token"
    outbox = EmailOutbox(
        email_type="invitation_resend",
        recipient="member@example.com",
        subject="Your invitation",
        template_data={
            "tenant_name": "Example Company",
            "inviter_name": "Tenant Owner",
            "expires_at": "2026-07-13T10:00:00+00:00",
            "encrypted_acceptance_url": encrypt_outbox_value(acceptance_url),
            "assignments": [
                {"department": "Development", "role": "Editor"},
            ],
        },
        idempotency_key="invitation:resend:1",
    )
    db.add(outbox)
    db.commit()

    assert process_next_email(db) is True

    db.refresh(outbox)
    assert outbox.status == "sent"
    assert outbox.attempts == 1
    assert outbox.sent_at is not None
    assert len(FakeSMTP.messages) == 1
    assert acceptance_url in FakeSMTP.messages[0].get_body(preferencelist=("plain",)).get_content()


def test_process_next_email_reclaims_an_expired_processing_lease(db: Session) -> None:
    outbox = EmailOutbox(
        email_type="invitation",
        recipient="member@example.com",
        subject="Invitation",
        template_data={
            "tenant_name": "Example Company",
            "encrypted_acceptance_url": encrypt_outbox_value(
                "http://localhost:5173/invite?token=single-use-token"
            ),
        },
        status="processing",
        attempts=1,
        available_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        idempotency_key="invitation:lease:1",
    )
    db.add(outbox)
    db.commit()

    assert process_next_email(db) is True

    db.refresh(outbox)
    assert outbox.status == "sent"
    assert outbox.attempts == 2


def test_process_next_email_leaves_an_active_processing_lease_alone(db: Session) -> None:
    outbox = EmailOutbox(
        email_type="invitation",
        recipient="member@example.com",
        subject="Invitation",
        template_data={},
        status="processing",
        attempts=1,
        available_at=datetime.now(timezone.utc) + timedelta(seconds=30),
        idempotency_key="invitation:lease:2",
    )
    db.add(outbox)
    db.commit()

    assert process_next_email(db) is False

    db.refresh(outbox)
    assert outbox.status == "processing"
    assert outbox.attempts == 1


def test_process_next_email_records_only_the_error_type(db: Session) -> None:
    outbox = EmailOutbox(
        email_type="invitation",
        recipient="member@example.com",
        subject="Invitation",
        template_data={"encrypted_acceptance_url": "invalid-secret-value"},
        idempotency_key="invitation:error:1",
    )
    db.add(outbox)
    db.commit()

    assert process_next_email(db) is True

    db.refresh(outbox)
    assert outbox.status == "pending"
    assert outbox.last_error == "InvalidToken"
    assert "invalid-secret-value" not in outbox.last_error
