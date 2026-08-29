from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _column_ddl_type(column) -> str:
    """Map a SQLAlchemy column's type to a DDL fragment for ALTER TABLE ADD COLUMN."""
    return column.type.compile(dialect=engine.dialect)


def _migrate_missing_columns():
    """
    Lightweight, dependency-free 'migration': for any model column that
    doesn't yet exist on its table (e.g. an existing cardiac.db predating a
    new field), add it with ALTER TABLE ADD COLUMN. Existing rows/data are
    left untouched; new columns come back NULL until populated. This avoids
    needing Alembic for a SQLite-backed MVP while never dropping data.
    """
    inspector = inspect(engine)
    if not inspector.has_table("cardiac_reports"):
        return  # fresh DB — create_all() already built the full schema

    existing_cols = {c["name"] for c in inspector.get_columns("cardiac_reports")}
    from app import models  # noqa: F401 (ensures model metadata is registered)

    with engine.begin() as conn:
        for column in models.CardiacReport.__table__.columns:
            if column.name in existing_cols:
                continue
            ddl_type = _column_ddl_type(column)
            conn.execute(text(f'ALTER TABLE cardiac_reports ADD COLUMN "{column.name}" {ddl_type}'))


def init_db():
    # Import models so they're registered on Base before create_all
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate_missing_columns()
