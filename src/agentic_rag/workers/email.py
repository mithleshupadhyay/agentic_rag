import html
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from agentic_rag.shared.config import settings
from agentic_rag.shared.db.crud.identity import decrypt_outbox_value
from agentic_rag.shared.db.models import EmailOutbox
from agentic_rag.shared.db.session import get_sync_session_factory


logger = logging.getLogger(__name__)


def process_next_email(db: Session) -> bool:
    now = datetime.now(timezone.utc)
    query = (
        db.query(EmailOutbox)
        .filter(
            or_(
                and_(
                    EmailOutbox.status == "pending",
                    EmailOutbox.available_at <= now,
                ),
                and_(
                    EmailOutbox.status == "processing",
                    EmailOutbox.available_at <= now,
                ),
            ),
            EmailOutbox.attempts < settings.email_worker_max_attempts,
        )
        .order_by(EmailOutbox.available_at, EmailOutbox.created_at)
    )
    if db.get_bind().dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)
    outbox = query.first()
    if outbox is None:
        return False

    outbox.status = "processing"
    outbox.attempts += 1
    outbox.available_at = now + timedelta(
        seconds=settings.email_worker_lease_seconds
    )
    db.commit()

    try:
        template_data = outbox.template_data
        if outbox.email_type not in {"invitation", "invitation_resend"}:
            raise RuntimeError(f"Unsupported email type: {outbox.email_type}")
        encrypted_url = template_data.get("encrypted_acceptance_url")
        if not isinstance(encrypted_url, str) or not encrypted_url:
            raise RuntimeError(
                "Invitation email is missing its encrypted acceptance URL."
            )
        acceptance_url = decrypt_outbox_value(encrypted_url)

        tenant_name_text = str(template_data.get("tenant_name") or "your company")
        inviter_name_text = str(
            template_data.get("inviter_name") or "A company administrator"
        )
        tenant_name = html.escape(tenant_name_text)
        inviter_name = html.escape(inviter_name_text)
        personal_message = template_data.get("personal_message")
        assignments = template_data.get("assignments") or []
        assignment_lines = []
        if isinstance(assignments, list):
            for assignment in assignments:
                if not isinstance(assignment, dict):
                    continue
                department = html.escape(
                    str(assignment.get("department") or "Department")
                )
                role = html.escape(str(assignment.get("role") or "Member"))
                assignment_lines.append(f"<li>{department}: {role}</li>")

        message = EmailMessage()
        message["From"] = f"{settings.email_from_name} <{settings.email_from_address}>"
        message["To"] = outbox.recipient
        message["Subject"] = outbox.subject
        message.set_content(
            f"{inviter_name_text} invited you to {tenant_name_text}.\n\n"
            f"Accept invitation: {acceptance_url}\n\n"
            f"This link expires at {template_data.get('expires_at', 'the stated expiry time')}."
        )
        personal_message_html = ""
        if isinstance(personal_message, str) and personal_message.strip():
            personal_message_html = f"<p>{html.escape(personal_message.strip())}</p>"
        assignments_html = ""
        if assignment_lines:
            assignments_html = f"<ul>{''.join(assignment_lines)}</ul>"
        message.add_alternative(
            "<html><body>"
            f"<p>{inviter_name} invited you to <strong>{tenant_name}</strong>.</p>"
            f"{personal_message_html}{assignments_html}"
            f'<p><a href="{html.escape(acceptance_url, quote=True)}">Accept invitation</a></p>'
            f"<p>This link expires at {html.escape(str(template_data.get('expires_at') or 'the stated expiry time'))}.</p>"
            f"<p>Questions? Contact {html.escape(settings.support_email)}.</p>"
            "</body></html>",
            subtype="html",
        )

        if settings.email_delivery_provider == "smtp":
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=15,
            ) as smtp:
                if settings.smtp_use_tls:
                    smtp.starttls()
                if settings.smtp_username:
                    smtp.login(settings.smtp_username, settings.smtp_password)
                smtp.send_message(message)
        elif settings.email_delivery_provider == "console":
            logger.info(
                f"[EmailWorker] Console delivery accepted outbox={outbox.id} "
                f"type={outbox.email_type} recipient={outbox.recipient}"
            )
        else:
            raise RuntimeError("Unsupported email delivery provider.")

        outbox.status = "sent"
        outbox.sent_at = datetime.now(timezone.utc)
        outbox.last_error = None
        db.commit()
        logger.info(
            f"[EmailWorker] Email sent outbox={outbox.id} type={outbox.email_type} "
            f"attempt={outbox.attempts}"
        )
        return True
    except Exception as error:
        db.rollback()
        outbox = db.query(EmailOutbox).filter(EmailOutbox.id == outbox.id).one()
        outbox.last_error = type(error).__name__
        if outbox.attempts >= settings.email_worker_max_attempts:
            outbox.status = "failed"
        else:
            outbox.status = "pending"
            retry_seconds = min(900, 15 * (2 ** (outbox.attempts - 1)))
            outbox.available_at = datetime.now(timezone.utc) + timedelta(
                seconds=retry_seconds
            )
        db.commit()
        logger.exception(
            f"[EmailWorker] Email delivery failed outbox={outbox.id} "
            f"type={outbox.email_type} attempt={outbox.attempts}: {type(error).__name__}"
        )
        return True


def run_email_worker_loop() -> None:
    logger.info("[EmailWorker] Worker loop started")
    session_factory = get_sync_session_factory()
    while True:
        try:
            with session_factory() as db:
                processed = process_next_email(db)
            if not processed:
                time.sleep(settings.email_worker_poll_seconds)
        except KeyboardInterrupt:
            logger.info("[EmailWorker] Worker loop stopped")
            return
        except Exception as error:
            logger.exception(f"[EmailWorker] Worker loop error: {type(error).__name__}")
            time.sleep(settings.email_worker_poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=os.getenv("LOGGING_LEVEL", "INFO").upper(),
    )
    run_email_worker_loop()
