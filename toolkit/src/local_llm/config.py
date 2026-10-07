"""Configuration — a single data class describing how the toolkit should behave.

`Settings` is deliberately *only* data. It has no methods that do work, because every
other class in the package takes a `Settings` instance through its constructor, and a
configuration object that also performed actions would drag those actions into every
class that merely needed to read a value.

There is no module-level `get_settings()` singleton. That was the earlier shape and it
inverted the dependencies the wrong way: each class reached out to a global to discover
its own configuration, which meant no class could be tested with different settings
without patching a module attribute. Now the entry point builds one `Settings` and hands
it down — see `container.Toolkit`, the single place that does this wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    """Where the call database and live-progress files live.

    Anchored to the installed package's project root rather than the current working
    directory. If it used the working directory, running the CLI from two different
    folders would silently create two different databases and history would appear to
    vanish depending on where you happened to be standing.

        <site-packages>/local_llm/config.py  ->  <project root>/.local-llm-data
    """
    return Path(__file__).resolve().parents[2] / ".local-llm-data"


class DatabaseConfig(BaseModel):
    """Where call history is stored — every database detail in one central block.

    Name the engine, give it the connection settings it needs, and the toolkit uses it.
    Nothing outside this block mentions the database, so switching engines is a change
    here and nowhere else.

        # .env — SQLite (the default; works with nothing set at all)
        LOCAL_LLM_DATABASE__ENGINE=sqlite
        LOCAL_LLM_DATABASE__NAME=C:/data/calls.db

        # .env — MySQL
        LOCAL_LLM_DATABASE__ENGINE=mysql
        LOCAL_LLM_DATABASE__NAME=local_llm
        LOCAL_LLM_DATABASE__HOST=127.0.0.1
        LOCAL_LLM_DATABASE__PORT=3306
        LOCAL_LLM_DATABASE__USER=llm
        LOCAL_LLM_DATABASE__PASSWORD=secret

    The double underscore is pydantic-settings' nested delimiter: `DATABASE__HOST` sets
    `settings.database.host`. Two underscores rather than one so a field name that itself
    contains an underscore stays unambiguous — `DATABASE__TIMEOUT_S` splits cleanly into
    `database` and `timeout_s`, which a single delimiter could not do.

    Or set it in code:

        Settings(database=DatabaseConfig(engine="mysql", name="local_llm",
                                         host="db.internal", user="llm", password="…"))

    `name` means different things per engine — a file path for SQLite, a schema name for
    MySQL — because keeping one field means switching engines does not also mean learning
    a different set of keys.
    """

    # A Literal rather than a plain str so a typo is caught by validation at startup,
    # naming the valid options, instead of failing later with "unknown engine" — or worse,
    # silently falling back to SQLite while the user waits for rows to appear in MySQL.
    engine: Literal["sqlite", "mysql", "mariadb"] = Field(
        default="sqlite",
        description="Which database to use. 'mariadb' is handled by the MySQL backend.",
    )
    name: str = Field(
        default="",
        description="SQLite: path to the .db file. MySQL: the database (schema) name. "
                    "Empty means the default path under the data directory.",
    )

    # ── server engines only; ignored by SQLite ──
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=3306)
    user: str = Field(default="")
    password: str = Field(default="")
    # utf8mb4 is the only MySQL charset that stores the whole of Unicode. The older
    # `utf8` alias is a three-byte subset that cannot hold emoji or many CJK characters,
    # and model output routinely contains both — inserting them would raise
    # "Incorrect string value" and lose the record.
    charset: str = Field(default="utf8mb4")

    timeout_s: float = Field(
        default=5.0,
        description="Connect timeout, and SQLite's busy timeout for lock contention.",
    )

    def backend_key(self) -> str:
        """The registry key for the configured engine.

        MariaDB speaks the MySQL wire protocol and accepts the same SQL, so it maps onto
        the same backend rather than duplicating a class that would be identical.

            "sqlite"   ->  "sqlite"
            "mysql"    ->  "mysql"
            "mariadb"  ->  "mysql"
        """
        return "mysql" if self.engine == "mariadb" else self.engine

    def resolved_sqlite_path(self, data_dir: Path | None = None) -> Path:
        """The SQLite file to use, defaulting to calls.db inside the data directory.

            name="" , data_dir=<project>/.local-llm-data  ->  <project>/.local-llm-data/calls.db
            name="C:/tmp/x.db"                            ->  C:/tmp/x.db
        """
        if self.name:
            return Path(self.name)
        base = data_dir if data_dir is not None else _default_data_dir()
        return base / "calls.db"


class Settings(BaseSettings):
    """Every knob the toolkit reads, resolved from the environment.

    Subclasses pydantic's `BaseSettings`, which means each field is populated from an
    environment variable named `LOCAL_LLM_<FIELD>` (or from a `.env` file) and validated
    to the declared type. That is why there is no manual `os.environ` reading anywhere:
    the type annotation *is* the parsing rule.

        LOCAL_LLM_MODEL=qwen/qwen3-14b   ->  Settings().model == "qwen/qwen3-14b"
        LOCAL_LLM_CONCURRENCY=abc        ->  ValidationError at startup, not later
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_LLM_",
        # Two locations because the package is used both from its own directory and from
        # the parent project that installed it editable; checking both means one .env
        # works in either case.
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        # Lets a nested block be configured from flat environment variables:
        # LOCAL_LLM_DATABASE__HOST sets settings.database.host. Without this,
        # `database` could only be set from code and the .env file could not
        # reach into it at all.
        env_nested_delimiter="__",
        # Ignore unrelated LOCAL_LLM_* variables rather than failing. Without this, a
        # leftover experimental variable in the environment would stop the tool starting.
        extra="ignore",
    )

    # ── model endpoint ──
    # 127.0.0.1 rather than "localhost", and this is not cosmetic — it was measured.
    #
    # On Windows, "localhost" resolves to the IPv6 address ::1 *before* IPv4 127.0.0.1.
    # LM Studio binds only to IPv4, so every connection first attempts ::1, waits for that
    # attempt to time out, and only then falls back. Measured against the same server:
    #
    #     http://localhost:1234/v1/models    2017 ms
    #     http://127.0.0.1:1234/v1/models       7 ms
    #
    # A 280x penalty on every request, and it is invisible: the call still succeeds, just
    # slowly, so it reads as "the model server is sluggish" rather than as a DNS problem.
    # The synchronous health probe paid the full 2 s; async streaming calls paid about
    # 270 ms each because httpx attempts both families concurrently there.
    #
    # Use a literal IPv4 address unless the server is genuinely remote.
    url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="OpenAI-compatible base URL. Bionic/LM Studio serves this on port 1234. "
                    "Prefer 127.0.0.1 over localhost: see the note above.",
    )
    model: str = Field(default="qwen/qwen3-14b", description="Model id to request.")
    api_key: str = Field(
        default="not-needed", description="Sent as a bearer token; local servers ignore it."
    )

    # ── generation ──
    temperature: float = 0.0
    # 2048 is generous on purpose. A reasoning model spends this budget on its own
    # thinking *before* answering — Qwen3 14B used 203 output tokens simply to say "OK" —
    # so a tight default would produce empty answers that look like model failures.
    max_tokens: int = 2048
    request_timeout_s: float = Field(
        default=300.0, description="Covers the whole stream, not just headers."
    )

    # ── page fetching ──
    fetch_timeout_s: float = 30.0
    max_page_chars: int = Field(
        default=24_000, description="Truncate page text before it reaches the model."
    )

    # Below this many characters of extracted text, re-fetch the page in a real browser.
    # 1,200 is deliberately low: it catches an empty or consent-gated page without paying
    # for a browser launch on every ordinary article. It does NOT catch a page whose body
    # extracted fine but whose *table* was built by JavaScript — Paddle's supported-
    # countries page yields a perfectly normal-looking 2,715 characters with the country
    # list missing entirely. That case is caught by the caller passing `expect`, not by
    # this threshold, and assuming otherwise is what produced a week of confident wrong
    # answers about which countries are supported.
    browser_escalate_below_chars: int = Field(default=1200, ge=0)
    # How long to wait for a rendered page before giving up on one wait strategy.
    # Generous because the pages worth escalating to a browser for are the slow ones —
    # a documentation site loading a table from an API is the whole use case.
    browser_page_timeout_s: float = Field(default=45.0, gt=0)
    # A fixed settle delay after the page reports ready. Frameworks routinely render a
    # table one tick later, and without this the fetch wins the race and captures the
    # empty shell — which is indistinguishable from a page that genuinely has no table.
    browser_settle_s: float = Field(default=2.5, ge=0)
    # When a membership check attributes a term to a heading this many lines back, re-read
    # the page in a browser before trusting the attribution. Measured: fetched over HTTP,
    # Lemon Squeezy's supported-countries page arrives with its whole country list joined
    # onto a single line, so line-based heading attribution put Pakistan under the coarse
    # "Supported countries" 86 lines back instead of the precise "Bank payouts supported in
    # the following countries:" immediately above it. The browser's DOM text keeps one item
    # per line. Without this, the check answers the question with the wrong list name — and
    # the list name is the entire answer.
    membership_reattribute_above_lines: int = Field(default=20, ge=0)

    # ── concurrency ──
    concurrency: int = Field(
        default=2,
        ge=1,
        description="Simultaneous model calls. One GPU: more than 2 thrashes VRAM rather than going faster.",
    )

    # ── storage / retention ──
    data_dir: Path = Field(default_factory=_default_data_dir)
    # The one central place every database detail lives. Swapping SQLite for MySQL is a
    # change to this block and nothing else — no other setting mentions the database.
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    payload_days: int = Field(default=7, description="Keep full prompts/responses this long.")
    max_payloads: int = Field(default=500, description="Hard cap on stored payloads, newest kept.")

    # ── search providers (pipeline only) ──
    search_provider: str = Field(default="auto", description="auto | brave | searxng | duckduckgo")
    brave_api_key: str = ""
    searxng_url: str = ""

    # ── monitoring ──
    lms_path: Path | None = None
    claude_projects_dir: Path | None = None

    # ── unattended server start (server_start.py) ──
    # Refuse to load a model when less than this much system RAM is free. The runtime
    # memory-maps the model file, so a load takes host RAM as well as VRAM; starting one
    # on a machine with no RAM to spare makes Windows page to disk and the whole PC
    # crawls, which with nobody home is not noticed until the day is lost. 6144 MiB is
    # 6 GiB, about the host-side cost of the 14B default plus the desktop's own needs.
    min_free_ram_mib: int = Field(default=6144, ge=0)
    # Longest one `lms server start` call may run before it is killed. Cold start prints
    # "Timed out waiting for LM Studio daemon to start" by itself after a while, so this
    # is only the backstop for a call that never returns at all.
    server_start_cmd_timeout_s: float = Field(default=60.0, gt=0)
    # How long to wait for the endpoint to answer after the start commands have run.
    server_start_wait_s: float = Field(default=30.0, gt=0)
    # A lock file older than this is treated as left behind by a crashed starter.
    server_start_lock_stale_s: float = Field(default=600.0, gt=0)

    # ── live Claude agent view ──
    # How long a *finished* agent stays on the dashboard. Finished agents linger instead
    # of vanishing because the dashboard detects a completion by watching a row change
    # status, and a row that disappears the instant it finishes can never be seen to
    # finish — the notification would never fire.
    agent_window_s: float = Field(default=60.0, ge=5.0)
    # How far back to look for transcripts holding a still-running agent. Deliberately
    # much wider than the display window: a parent session writes nothing at all while it
    # waits for an agent, so its transcript's modification time can be minutes old while
    # the agent is very much alive. A tight scan window would skip that file and lose
    # exactly the agents most worth watching.
    agent_scan_hours: float = Field(default=6.0, ge=0.5)
    # When a launch has no result after this long, treat it as dead rather than running.
    # Needed because a session killed mid-agent leaves a tool call with no result *for
    # ever*; without this the panel would report a phantom agent running for days, which
    # destroys the credibility of the one number it exists to provide.
    agent_stale_after_s: float = Field(default=3600.0, ge=60.0)
    # Characters per token, for the agent cost *estimate* only. Subagent token usage is
    # recorded nowhere on this machine (see agents.py), so the panel estimates from the
    # prompt in and the report out. About 4 for English prose; lower for dense code.
    agent_chars_per_token: float = Field(default=4.0, gt=0.0)

    @field_validator("url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        """Normalise the endpoint so paths can be appended with a single slash.

        Every request builds its URL as f"{url}/chat/completions". A configured value of
        "http://localhost:1234/v1/" would therefore produce a double slash, which some
        servers route as a different, non-existent path and answer with 404.

            "http://localhost:1234/v1/"  ->  "http://localhost:1234/v1"
        """
        return value.rstrip("/")

    @property
    def db_path(self) -> Path:
        """The SQLite file path.

        Meaningful only when the engine is SQLite; a MySQL deployment has no file. Kept
        as a property so callers that only ever run SQLite — the smoke test, a status
        line — need not reach into the nested config.
        """
        return self.database.resolved_sqlite_path(self.data_dir)

    @property
    def server_start_lock_path(self) -> Path:
        return self.data_dir / "server-start.lock"

    @property
    def live_dir(self) -> Path:
        return self.data_dir / "live"

    def ensure_dirs(self) -> None:
        """Create the data directories if they are missing.

        Called once by the composition root rather than by each class that writes a file,
        so the "make sure the folder exists" concern lives in exactly one place. Creating
        `live_dir` implicitly creates `data_dir`, since it sits inside it.
        """
        self.live_dir.mkdir(parents=True, exist_ok=True)
