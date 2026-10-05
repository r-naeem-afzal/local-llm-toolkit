"""Database backends — one class per engine, so the repository contains no dialect.

The store used to speak SQLite directly: `?` placeholders, `on conflict … do update`,
`pragma journal_mode`, and a file size read off disk. All four are SQLite-specific, so
supporting MySQL by adding `if engine == "mysql"` branches would have scattered dialect
decisions through every query in the repository.

Instead each engine is a class implementing `DatabaseBackend`, which owns three things:

1. **How to connect** — and how to keep a connection usable, which differs sharply: a
   SQLite connection is a file handle that never expires, while a MySQL server closes
   idle connections on its own.
2. **The schema** — including column types, which are not portable. `text` is a fine
   primary key in SQLite and illegal as one in MySQL.
3. **The dialect-sensitive statements** — upserts, retention deletes, size and reclaim.

Adding PostgreSQL later means adding one class here and one entry in `BACKENDS`. No
existing query changes, which is the open/closed principle doing real work rather than
being cited decoratively.

## Why the repository cannot simply write portable SQL

Three of the statements it needs have no portable form:

* **Upsert.** SQLite: `on conflict(id) do update set status = excluded.status`.
  MySQL: `on duplicate key update status = values(status)`. Different keyword, different
  way to name the incoming row.
* **Retention by count.** The natural query is
  `delete from payloads where call_id not in (select id from calls order by ts desc
  limit ?)`. MySQL rejects that outright — "This version of MySQL doesn't yet support
  'LIMIT & IN/ALL/ANY/SOME subquery'" — so it needs the subquery wrapped in a derived
  table to launder the LIMIT.
* **Size and reclaim.** SQLite is one file, so size is `stat()` and reclaim is `VACUUM`.
  MySQL keeps size in `information_schema` and reclaims with `OPTIMIZE TABLE`.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from decimal import Decimal
from pathlib import Path
from typing import Any

from .config import DatabaseConfig


class DatabaseBackend(ABC):
    """Everything engine-specific about storing call records.

    `placeholder` exists because the DB-API leaves parameter style up to the driver:
    sqlite3 uses `?` and PyMySQL uses `%s`. Queries in the repository are written with
    `?` and passed through `sql()`, so there is exactly one place where that translation
    happens rather than two copies of every query.
    """

    name: str = "abstract"
    placeholder: str = "?"

    @abstractmethod
    def connect(self) -> Any:
        """Open a new connection, configured and ready to use."""

    @abstractmethod
    def is_alive(self, connection: Any) -> bool:
        """Whether a cached connection can still be used."""

    @abstractmethod
    def schema_statements(self) -> list[str]:
        """DDL to create tables and indexes, each statement executed separately."""

    @abstractmethod
    def existing_columns(self, connection: Any, table: str) -> set[str]:
        """Column names currently on `table`, for detecting missing migrations."""

    @abstractmethod
    def add_column(self, connection: Any, table: str, column: str) -> None:
        """Add one column that a newer version introduced."""

    @abstractmethod
    def size_bytes(self, connection: Any) -> int:
        """How much disk the stored data occupies."""

    def sql(self, template: str) -> str:
        """Translate a `?`-style query into this driver's parameter style.

            "select * from calls where id = ?"   (sqlite)  ->  unchanged
            "select * from calls where id = ?"   (mysql)   ->  "... where id = %s"

        Written this way so every query in the repository reads identically regardless of
        engine. The alternative — each query duplicated per dialect — is where drift and
        subtle divergence between backends comes from.
        """
        return template if self.placeholder == "?" else template.replace("?", self.placeholder)

    # ── running statements ──
    #
    # These exist because the two drivers disagree about cursors. sqlite3 lets you call
    # `connection.execute(...)` directly, while PyMySQL requires an explicit
    # `connection.cursor()` used as a context manager. Hiding that here means the
    # repository never mentions a cursor and reads the same for either engine.

    @abstractmethod
    def query(self, connection: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Run a SELECT and return rows as dicts, so `row["ts"]` works on both engines."""

    @abstractmethod
    def execute(self, connection: Any, sql: str, params: tuple = ()) -> int:
        """Run a statement and return the number of rows it affected."""

    def begin_transaction(self, connection: Any) -> None:
        """Start an explicit transaction for a multi-statement write.

        A no-op where the driver already begins one implicitly (sqlite3). Needed where
        the connection runs in autocommit mode, so that two related statements either
        both land or neither does — otherwise the dashboard could read a call row whose
        payload has not been written yet.
        """

    def commit(self, connection: Any) -> None:
        connection.commit()

    def rollback(self, connection: Any) -> None:
        connection.rollback()

    # ── statements that have no portable form ──

    @abstractmethod
    def upsert_call(self) -> str:
        """Insert-or-update a call row.

        Insert-or-update rather than a plain UPDATE because a call that fails before the
        model is reached never got a start row, and an UPDATE would then affect zero rows
        and lose the error record entirely.
        """

    @abstractmethod
    def upsert_payload(self) -> str:
        """Insert-or-update the response and reasoning for a call.

        Must preserve whichever of the two this write did not supply — writing only a
        response must not blank out reasoning already stored for the same call.
        """

    @abstractmethod
    def replace_call_start(self) -> str:
        """Write the initial 'running' row, replacing any row with the same id."""

    @abstractmethod
    def replace_payload_prompt(self) -> str:
        """Write the prompt, replacing any prompt already stored for the call."""

    @abstractmethod
    def delete_payloads_beyond_count(self) -> str:
        """Delete payloads outside the newest N calls."""

    def tolerates_duplicate_index_error(self) -> bool:
        """Whether re-running the schema DDL can raise a benign "index exists" error."""
        return False

    def checkpoint(self, connection: Any) -> None:
        """Flush pending writes so a size reading reflects reality. No-op by default."""

    def reclaim(self, connection: Any) -> None:
        """Return freed space to the filesystem. Optional per engine."""


class SqliteBackend(DatabaseBackend):
    """SQLite — the default. One file, no server, nothing to install.

    Chosen as the default because the toolkit is primarily a local tool and a local tool
    that requires a database server to be running before it can log anything has failed
    at its job.
    """

    name = "sqlite"
    placeholder = "?"

    def __init__(self, config: DatabaseConfig, data_dir: Path | None = None) -> None:
        self._config = config
        # data_dir is passed in rather than recomputed, so that a Settings with a custom
        # data_dir puts the database there too. Resolving it independently here would
        # silently write to the default location and the configured directory would sit
        # empty — a confusing failure that looks like history being lost.
        self._path = Path(config.resolved_sqlite_path(data_dir))

    @property
    def path(self) -> Path:
        return self._path

    def connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path, timeout=self._config.timeout_s)
        # row_factory makes rows behave like dicts (row["ts"]) rather than tuples
        # (row[1]), so adding a column to a select cannot silently shift what the
        # positional reads mean.
        connection.row_factory = sqlite3.Row

        # WAL ("write-ahead log") writes changes to a side file and lets readers carry on
        # against the main file. Without it a pipeline writing calls and a dashboard
        # reading them block each other, so the dashboard stalls exactly when there is
        # most to look at.
        connection.execute("pragma journal_mode = wal")
        # If a write does hit contention, wait rather than failing immediately.
        connection.execute(f"pragma busy_timeout = {int(self._config.timeout_s * 1000)}")
        # `normal` does not fsync on every commit. The trade: an OS crash could lose the
        # last few records. Acceptable because these are observability records, whereas
        # `full` would make every model call pay a disk sync.
        connection.execute("pragma synchronous = normal")
        return connection

    def is_alive(self, connection: Any) -> bool:
        # A SQLite connection is a local file handle. It cannot time out or be closed by
        # a peer, so it is alive unless the process explicitly closed it.
        return True

    def schema_statements(self) -> list[str]:
        return [
            """
            create table if not exists calls (
                id          text primary key,
                ts          text not null,
                tool        text,
                model       text,
                status      text,
                duration_ms integer,
                tokens_in   integer,
                tokens_out  integer,
                error       text,
                stage       text,
                meta_json   text
            )
            """,
            "create index if not exists calls_ts_idx    on calls (ts desc)",
            "create index if not exists calls_model_idx on calls (model)",
            "create index if not exists calls_tool_idx  on calls (tool)",
            """
            create table if not exists payloads (
                call_id   text primary key,
                prompt    text,
                response  text,
                reasoning text
            )
            """,
        ]

    def existing_columns(self, connection: Any, table: str) -> set[str]:
        """Column names via `pragma table_info`, which returns the name at index 1.

            [(0,'call_id','text',…), (1,'prompt','text',…)]  ->  {"call_id", "prompt"}
        """
        return {row[1] for row in connection.execute(f"pragma table_info({table})")}

    def add_column(self, connection: Any, table: str, column: str) -> None:
        # ALTER TABLE ADD COLUMN is cheap in SQLite: it edits the stored schema and does
        # not rewrite existing rows, which read back as NULL. That is the right value for
        # calls recorded before the column existed.
        connection.execute(f"alter table {table} add column {column} text")

    def size_bytes(self, connection: Any) -> int:
        try:
            return self._path.stat().st_size
        except OSError:
            # The file does not exist until the first write. 0 is correct, and lets the
            # dashboard render before anything has been recorded.
            return 0

    def query(self, connection: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cursor = connection.execute(self.sql(sql), params)
        # sqlite3.Row supports mapping access but is not a dict, and the API layer
        # serialises these to JSON — so convert once here rather than at every call site.
        return [dict(row) for row in cursor.fetchall()]

    def execute(self, connection: Any, sql: str, params: tuple = ()) -> int:
        return connection.execute(self.sql(sql), params).rowcount

    def upsert_call(self) -> str:
        return """
        insert into calls (id, ts, tool, model, status, duration_ms,
                           tokens_in, tokens_out, error, stage, meta_json)
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(id) do update set
            status      = excluded.status,
            duration_ms = excluded.duration_ms,
            tokens_in   = coalesce(excluded.tokens_in, calls.tokens_in),
            tokens_out  = coalesce(excluded.tokens_out, calls.tokens_out),
            error       = excluded.error,
            stage       = excluded.stage,
            -- `coalesce`, not a plain assignment, for the same reason as the token
            -- counts: the finishing write does not always carry metadata, and
            -- `meta_json = excluded.meta_json` would then erase what the starting write
            -- recorded — the URL and question a call was about.
            --
            -- Omitting this line entirely was the original bug. Metadata added when a
            -- call *finishes* was silently discarded, because the conflict clause only
            -- updates the columns it names. It cost nothing until the client began
            -- recording timing at the end of a call: the values were computed, passed,
            -- accepted and dropped, and the only symptom was an empty column.
            meta_json   = coalesce(excluded.meta_json, calls.meta_json)
        """

    def upsert_payload(self) -> str:
        # `excluded` is SQLite's name for the row that failed to insert, so
        # coalesce(excluded.x, payloads.x) means "the new value if given, else keep the
        # stored one".
        return """
        insert into payloads (call_id, response, reasoning) values (?, ?, ?)
        on conflict(call_id) do update set
            response  = coalesce(excluded.response, payloads.response),
            reasoning = coalesce(excluded.reasoning, payloads.reasoning)
        """

    def replace_call_start(self) -> str:
        return (
            "insert or replace into calls (id, ts, tool, model, status, meta_json)"
            " values (?, ?, ?, ?, 'running', ?)"
        )

    def replace_payload_prompt(self) -> str:
        return "insert or replace into payloads (call_id, prompt) values (?, ?)"

    def delete_payloads_beyond_count(self) -> str:
        return (
            "delete from payloads where call_id not in"
            " (select id from calls order by ts desc limit ?)"
        )

    def checkpoint(self, connection: Any) -> None:
        # Fold the write-ahead log back into the main file. Without this, deletions live
        # only in the WAL and the database *appears to grow* after a prune — which looks
        # exactly like the prune having failed.
        connection.execute("pragma wal_checkpoint(truncate)")

    def reclaim(self, connection: Any) -> None:
        """Rebuild the file so freed pages return to the filesystem.

        Deleting rows does not shrink a SQLite file: the freed pages are kept for reuse.
        VACUUM rewrites the whole database to reclaim them.
        """
        self.checkpoint(connection)
        # VACUUM cannot run inside a transaction, and Python's sqlite3 may have opened an
        # implicit one, so commit before issuing it or the statement fails.
        connection.commit()
        connection.execute("vacuum")


class MySqlBackend(DatabaseBackend):
    """MySQL / MariaDB, for when the history should outlive one machine.

    Worth the extra moving part when several machines write to one history, or when the
    dashboard runs somewhere other than the box holding the GPU. The cost is that the
    server must be reachable before anything can be logged.
    """

    name = "mysql"
    # PyMySQL uses printf-style parameters, not question marks. Getting this wrong does
    # not raise a clear error — the driver reports a syntax error near '?', which reads
    # as though the SQL itself is malformed.
    placeholder = "%s"

    # utf8mb4 stores four bytes per character, and an indexed VARCHAR key is limited to
    # 3072 bytes on modern InnoDB (767 on older versions). 191 characters is the
    # conventional safe width — 191 * 4 = 764 — and comfortably fits call ids like
    # "26620-3". A `text` column, which is what SQLite uses here, cannot be a primary key
    # in MySQL at all without an explicit prefix length.
    _KEY_WIDTH = 191

    def __init__(self, config: DatabaseConfig, data_dir: Path | None = None) -> None:
        self._config = config
        # data_dir is accepted and ignored so both backends share one constructor
        # signature — that is what lets `build_backend` construct either without
        # branching on which engine it picked (Liskov substitution, in practice).
        del data_dir

    def connect(self) -> Any:
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            # Names the fix, because the failure is a missing optional dependency rather
            # than anything wrong with the configuration.
            raise RuntimeError(
                "MySQL backend requires PyMySQL — install it with: "
                "pip install 'local-llm[mysql]'"
            ) from exc

        return pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.name,
            charset=self._config.charset,
            connect_timeout=int(self._config.timeout_s),
            # DictCursor makes rows behave like dicts, matching sqlite3.Row so the
            # repository's row["column"] access works identically on both engines. Without
            # it every read would need a second, tuple-based code path.
            cursorclass=DictCursor,
            # Autocommit ON, with multi-statement writes wrapped explicitly by
            # `begin_transaction` below.
            #
            # This was originally False, on the reasoning that the repository manages its
            # own transactions. That turned out to be a serious bug, found by testing:
            # with autocommit off, InnoDB starts a transaction on the *first SELECT* and
            # holds it open until something commits. A read-only client — which the
            # dashboard is — therefore parks an open transaction indefinitely.
            #
            # The observed consequence was a hang. `OPTIMIZE TABLE` needs a table
            # metadata lock, and it waited behind an idle connection that had held a
            # transaction for 498 seconds purely from having run a SELECT:
            #
            #     id 25  Query  178s  "Waiting for table metadata lock"  optimize table …
            #     id 17  Sleep  494s  (holding the transaction that blocks it)
            #
            # It would also have pinned the read view, so the dashboard would keep showing
            # a stale snapshot of history no matter how many new calls were written.
            # SQLite never shows either symptom, which is exactly why this needed a real
            # server to find.
            autocommit=True,
        )

    def is_alive(self, connection: Any) -> bool:
        """Whether a cached connection still works, reconnecting if it does not.

        This is the difference that matters most between the engines. A MySQL server
        closes connections that have been idle longer than `wait_timeout` — eight hours by
        default, but often minutes behind a proxy or load balancer. A dashboard left open
        overnight would otherwise fail with "MySQL server has gone away" on the first
        query of the morning, and would keep failing, because nothing would replace the
        dead connection.

        `ping(reconnect=True)` checks the connection and transparently re-establishes it,
        which turns a hard failure into a brief pause.
        """
        try:
            connection.ping(reconnect=True)
            return True
        except Exception:
            return False

    def schema_statements(self) -> list[str]:
        return [
            f"""
            create table if not exists calls (
                id          varchar({self._KEY_WIDTH}) primary key,
                ts          varchar(40) not null,
                tool        varchar(120),
                model       varchar(200),
                status      varchar(20),
                duration_ms int,
                tokens_in   int,
                tokens_out  int,
                -- longtext because an error can carry a truncated server response, which
                -- routinely exceeds the 65,535 bytes a plain `text` column allows.
                error       longtext,
                stage       varchar(60),
                meta_json   longtext
            ) engine=InnoDB default charset=utf8mb4
            """,
            # MySQL has no `create index if not exists`, so a re-run raises "Duplicate key
            # name". The repository swallows that specific case — see `SchemaMigrator`.
            "create index calls_ts_idx    on calls (ts)",
            "create index calls_model_idx on calls (model)",
            "create index calls_tool_idx  on calls (tool)",
            f"""
            create table if not exists payloads (
                call_id   varchar({self._KEY_WIDTH}) primary key,
                -- All three hold model text that can run to tens of thousands of
                -- characters: a fetched article prompt measured 18,738 characters, and
                -- reasoning is often longer than the answer.
                prompt    longtext,
                response  longtext,
                reasoning longtext
            ) engine=InnoDB default charset=utf8mb4
            """,
        ]

    def tolerates_duplicate_index_error(self) -> bool:
        """MySQL has no `create index if not exists`, so re-running the DDL raises.

        SQLite's `create index if not exists` makes schema setup idempotent. MySQL has no
        such form for indexes, so the second startup against an existing database raises
        "Duplicate key name: 'calls_ts_idx'". That is not an error worth surfacing — the
        index exists, which is the desired state — so the migrator is told it may ignore
        it. Without this the process would fail on every start after the first.
        """
        return True

    def existing_columns(self, connection: Any, table: str) -> set[str]:
        """Column names from information_schema, scoped to the configured database.

        `table_schema = database()` matters: without it, a server hosting several
        databases with a `payloads` table would return columns from all of them and the
        migration check would wrongly conclude nothing needs adding.
        """
        with connection.cursor() as cursor:
            cursor.execute(
                "select column_name from information_schema.columns"
                " where table_schema = database() and table_name = %s",
                (table,),
            )
            # DictCursor returns dicts, and MySQL versions disagree on the case of the
            # column label (`column_name` vs `COLUMN_NAME`), so take the single value from
            # each row rather than indexing by name.
            return {str(next(iter(row.values()))) for row in cursor.fetchall()}

    def add_column(self, connection: Any, table: str, column: str) -> None:
        with connection.cursor() as cursor:
            cursor.execute(f"alter table {table} add column {column} longtext")

    def size_bytes(self, connection: Any) -> int:
        """Stored size from information_schema, data plus indexes.

        MySQL has no single file to stat, and these figures are the server's own estimate
        rather than an exact byte count — accurate enough for a "how big is this getting"
        display, which is all it is used for.
        """
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "select coalesce(sum(data_length + index_length), 0) as bytes"
                    " from information_schema.tables"
                    " where table_schema = database() and table_name in ('calls','payloads')"
                )
                row = cursor.fetchone()
                return int(next(iter(row.values())) or 0) if row else 0
        except Exception:
            return 0

    def begin_transaction(self, connection: Any) -> None:
        # Explicit BEGIN, because the connection is in autocommit mode. Without this a
        # two-statement write (call row + payload row) would be two transactions, and a
        # crash between them would leave a call with no payload.
        connection.begin()

    def query(self, connection: Any, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with connection.cursor() as cursor:
            cursor.execute(self.sql(sql), params)
            # DictCursor already yields dicts, but the values still need normalising —
            # see _normalise_row.
            return [self._normalise_row(row) for row in cursor.fetchall()]

    @staticmethod
    def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
        """Convert MySQL's Decimal aggregates to plain ints.

        MySQL returns `SUM()` and `COUNT()` over integer columns as `decimal.Decimal`,
        whereas SQLite returns a plain `int`. That difference is invisible until the
        values are serialised: `json.dumps` raises "Object of type Decimal is not JSON
        serializable", so `/stats` answered 200 on SQLite and **500 on MySQL** with no
        other symptom. Normalising here rather than at each call site means every future
        aggregate query is covered by default.

            {"calls": 18, "errors": Decimal("3"), "ms": Decimal("165216")}
              ->  {"calls": 18, "errors": 3, "ms": 165216}

        Non-integral values keep their fractional part as a float rather than being
        truncated, so a genuine decimal column would not be silently rounded.
        """
        normalised: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, Decimal):
                # `value == value.to_integral_value()` asks "is there a fractional
                # part?" without converting first and losing the answer.
                normalised[key] = (
                    int(value) if value == value.to_integral_value() else float(value)
                )
            else:
                normalised[key] = value
        return normalised

    def execute(self, connection: Any, sql: str, params: tuple = ()) -> int:
        with connection.cursor() as cursor:
            cursor.execute(self.sql(sql), params)
            return cursor.rowcount

    def upsert_call(self) -> str:
        # `values(col)` is MySQL's way to name the value that would have been inserted —
        # the equivalent of SQLite's `excluded`. It is deprecated in MySQL 8.0.20+ in
        # favour of a row alias, but is still accepted there and is the only form that
        # also works on MariaDB and older MySQL, so it is the portable choice.
        return """
        insert into calls (id, ts, tool, model, status, duration_ms,
                           tokens_in, tokens_out, error, stage, meta_json)
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        on duplicate key update
            status      = values(status),
            duration_ms = values(duration_ms),
            tokens_in   = coalesce(values(tokens_in), tokens_in),
            tokens_out  = coalesce(values(tokens_out), tokens_out),
            error       = values(error),
            stage       = values(stage),
            -- See the SQLite backend for why this is a coalesce and why its absence was
            -- a bug: metadata recorded when a call finishes was accepted and dropped.
            meta_json   = coalesce(values(meta_json), meta_json)
        """

    def upsert_payload(self) -> str:
        return """
        insert into payloads (call_id, response, reasoning) values (%s, %s, %s)
        on duplicate key update
            response  = coalesce(values(response), response),
            reasoning = coalesce(values(reasoning), reasoning)
        """

    def replace_call_start(self) -> str:
        # REPLACE deletes any existing row and inserts afresh, which is the same effect
        # as SQLite's `insert or replace` and is correct here: a start row has no earlier
        # state worth preserving.
        return (
            "replace into calls (id, ts, tool, model, status, meta_json)"
            " values (%s, %s, %s, %s, 'running', %s)"
        )

    def replace_payload_prompt(self) -> str:
        return "replace into payloads (call_id, prompt) values (%s, %s)"

    def delete_payloads_beyond_count(self) -> str:
        """Delete payloads outside the newest N calls.

        The extra `select … from (…) as keep` wrapper is not redundant. MySQL refuses a
        LIMIT inside an IN subquery outright:

            This version of MySQL doesn't yet support
            'LIMIT & IN/ALL/ANY/SOME subquery'

        Wrapping it in a derived table — a subquery in the FROM clause, which MySQL
        materialises first — launders the LIMIT into an ordinary result set that IN will
        accept. Same meaning, and the only form that runs.
        """
        return (
            "delete from payloads where call_id not in ("
            "  select id from (select id from calls order by ts desc limit %s) as keep"
            ")"
        )

    def reclaim(self, connection: Any) -> None:
        """Rebuild the tables so deleted rows stop occupying disk.

        InnoDB keeps freed pages inside the tablespace after a delete, exactly as SQLite
        keeps freed pages in its file. OPTIMIZE TABLE rewrites the table to release them.

        It needs an exclusive table metadata lock, so it blocks behind any connection
        holding an open transaction on these tables — which is why the connection above
        runs in autocommit mode. With autocommit off, a single idle reader was enough to
        make this wait forever.
        """
        with connection.cursor() as cursor:
            cursor.execute("optimize table calls, payloads")


# The engine registry. Adding PostgreSQL means writing the class and adding one line
# here — nothing that already works has to be touched.
BACKENDS: dict[str, type[DatabaseBackend]] = {
    "sqlite": SqliteBackend,
    "mysql": MySqlBackend,
}


def build_backend(config: DatabaseConfig, data_dir: Path | None = None) -> DatabaseBackend:
    """Create the backend named by the configuration.

    Fails loudly and lists the valid options. A silent fallback to SQLite would leave the
    user watching an empty MySQL table with no indication of why.
    """
    engine = config.backend_key()
    backend_class = BACKENDS.get(engine)
    if backend_class is None:
        raise ValueError(
            f"unknown database engine {config.engine!r} — "
            f"expected one of: {', '.join(sorted(BACKENDS))}"
        )
    return backend_class(config, data_dir)
