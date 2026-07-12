"""Harden membership row-level security policies.

Revision ID: d4f0a2c6e8b1
Revises: c3e8f1a4d9b2
"""

from alembic import op


revision = "d4f0a2c6e8b1"
down_revision = "c3e8f1a4d9b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "postgresql":
        return

    op.execute(
        "DROP POLICY IF EXISTS tenant_memberships_tenant_isolation "
        "ON tenant_memberships"
    )
    op.execute(
        "CREATE POLICY tenant_memberships_read ON tenant_memberships FOR SELECT "
        "USING (current_user = 'agentic_rag_worker' OR "
        "tenant_key = current_setting('app.tenant_id', true) OR "
        "user_id::text = current_setting('app.user_id', true))"
    )
    op.execute(
        "CREATE POLICY tenant_memberships_write ON tenant_memberships "
        "FOR ALL USING (current_user = 'agentic_rag_worker' OR "
        "tenant_key = current_setting('app.tenant_id', true)) "
        "WITH CHECK (current_user = 'agentic_rag_worker' OR "
        "tenant_key = current_setting('app.tenant_id', true))"
    )

    op.execute(
        "DROP POLICY IF EXISTS department_memberships_tenant_isolation "
        "ON department_memberships"
    )
    op.execute(
        "CREATE POLICY department_memberships_read ON department_memberships FOR SELECT "
        "USING (current_user = 'agentic_rag_worker' OR "
        "tenant_id::text = current_setting('app.tenant_uuid', true) OR "
        "user_id::text = current_setting('app.user_id', true))"
    )
    op.execute(
        "CREATE POLICY department_memberships_write ON department_memberships "
        "FOR ALL USING (current_user = 'agentic_rag_worker' OR "
        "tenant_id::text = current_setting('app.tenant_uuid', true)) "
        "WITH CHECK (current_user = 'agentic_rag_worker' OR "
        "tenant_id::text = current_setting('app.tenant_uuid', true))"
    )


def downgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "postgresql":
        return

    for policy_name, table_name in (
        ("department_memberships_write", "department_memberships"),
        ("department_memberships_read", "department_memberships"),
        ("tenant_memberships_write", "tenant_memberships"),
        ("tenant_memberships_read", "tenant_memberships"),
    ):
        op.execute(f"DROP POLICY IF EXISTS {policy_name} ON {table_name}")

    op.execute(
        "CREATE POLICY tenant_memberships_tenant_isolation ON tenant_memberships "
        "USING (current_user = 'agentic_rag_worker' OR "
        "tenant_key = current_setting('app.tenant_id', true)) "
        "WITH CHECK (current_user = 'agentic_rag_worker' OR "
        "tenant_key = current_setting('app.tenant_id', true))"
    )
    op.execute(
        "CREATE POLICY department_memberships_tenant_isolation "
        "ON department_memberships USING (current_user = 'agentic_rag_worker' OR "
        "tenant_id::text = current_setting('app.tenant_uuid', true)) "
        "WITH CHECK (current_user = 'agentic_rag_worker' OR "
        "tenant_id::text = current_setting('app.tenant_uuid', true))"
    )
