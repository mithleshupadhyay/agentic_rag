from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from agentic_rag.shared.config import settings
from agentic_rag.shared.db import models  # noqa: F401
from agentic_rag.shared.db.base import Base


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.sync_database_url)

target_metadata = Base.metadata


def compare_column_type(
    migration_context,
    inspected_column,
    metadata_column,
    inspected_type,
    metadata_type,
):
    if (
        migration_context.dialect.name == "sqlite"
        and metadata_column.table.name == "chunk_embeddings"
        and metadata_column.name == "embedding"
    ):
        return False
    return None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=compare_column_type,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=compare_column_type,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
