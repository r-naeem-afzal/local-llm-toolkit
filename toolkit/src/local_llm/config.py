"""Configuration: a single data class describing how the toolkit should behave.

`Settings` is deliberately *only* data. It has no methods that do work, because every
other class in the package takes a `Settings` instance through its constructor, and a
configuration object that also performed actions would drag those actions into every
class that merely needed to read a value.

There is no module-level `get_settings()` singleton. That was the earlier shape and it
inverted the dependencies the wrong way: each class reached out to a global to discover
its own configuration, which meant no class could be tested with different settings
without patching a module attribute. Now the entry point builds one `Settings` and hands
it down; see `container.Toolkit`, the single place that does this wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The context length, in tokens, that every `lms load` asks for. A module constant rather
# than a bare default on the field below because `loader.py` uses the same number as the
# default for a loader built by hand, and two copies of one safety figure drift apart.
# The reason for the number is written once, at `Settings.load_context_length`.
DEFAULT_LOAD_CONTEXT_LENGTH = 32_768

# The two floors a model has to clear after it has loaded. Module constants for the same
# reason as the one above: `loader.py` uses them as defaults for a loader built by hand.
# The reasons for the numbers are written once, at the fields `min_tokens_per_second` and
# `min_free_vram_mib` below.
DEFAULT_MIN_TOKENS_PER_SECOND = 20.0
DEFAULT_MIN_FREE_VRAM_MIB = 100
# How long one process waits for another to finish a model load: see the field
# `model_load_lock_wait_s` below.
DEFAULT_LOAD_LOCK_WAIT_S = 180.0


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
    """Where call history is stored: every database detail in one central block.

    Name the engine, give it the connection settings it needs, and the toolkit uses it.
    Nothing outside this block mentions the database, so switching engines is a change
    here and nowhere else.

        # .env: SQLite (the default; works with nothing set at all)
        LOCAL_LLM_DATABASE__ENGINE=sqlite
        LOCAL_LLM_DATABASE__NAME=C:/data/calls.db

        # .env: MySQL
        LOCAL_LLM_DATABASE__ENGINE=mysql
        LOCAL_LLM_DATABASE__NAME=local_llm
        LOCAL_LLM_DATABASE__HOST=127.0.0.1
        LOCAL_LLM_DATABASE__PORT=3306
        LOCAL_LLM_DATABASE__USER=llm
        LOCAL_LLM_DATABASE__PASSWORD=secret

    The double underscore is pydantic-settings' nested delimiter: `DATABASE__HOST` sets
    `settings.database.host`. Two underscores rather than one so a field name that itself
    contains an underscore stays unambiguous: `DATABASE__TIMEOUT_S` splits cleanly into
    `database` and `timeout_s`, which a single delimiter could not do.

    Or set it in code:

        Settings(database=DatabaseConfig(engine="mysql", name="local_llm",
                                         host="db.internal", user="llm", password="…"))

    `name` means different things per engine, a file path for SQLite, a schema name for
    MySQL, because keeping one field means switching engines does not also mean learning
    a different set of keys.
    """

    # A Literal rather than a plain str so a typo is caught by validation at startup,
    # naming the valid options, instead of failing later with "unknown engine", or worse,
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
    # and model output routinely contains both: inserting them would raise
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
    # 127.0.0.1 rather than "localhost", and this is not cosmetic: it was measured.
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
    # thinking *before* answering, Qwen3 14B used 203 output tokens simply to say "OK",
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
    # extracted fine but whose *table* was built by JavaScript: Paddle's supported-
    # countries page yields a perfectly normal-looking 2,715 characters with the country
    # list missing entirely. That case is caught by the caller passing `expect`, not by
    # this threshold, and assuming otherwise is what produced a week of confident wrong
    # answers about which countries are supported.
    browser_escalate_below_chars: int = Field(default=1200, ge=0)
    # How long to wait for a rendered page before giving up on one wait strategy.
    # Generous because the pages worth escalating to a browser for are the slow ones:
    # a documentation site loading a table from an API is the whole use case.
    browser_page_timeout_s: float = Field(default=45.0, gt=0)
    # A fixed settle delay after the page reports ready. Frameworks routinely render a
    # table one tick later, and without this the fetch wins the race and captures the
    # empty shell, which is indistinguishable from a page that genuinely has no table.
    browser_settle_s: float = Field(default=2.5, ge=0)
    # When a membership check attributes a term to a heading this many lines back, re-read
    # the page in a browser before trusting the attribution. Measured: fetched over HTTP,
    # Lemon Squeezy's supported-countries page arrives with its whole country list joined
    # onto a single line, so line-based heading attribution put Pakistan under the coarse
    # "Supported countries" 86 lines back instead of the precise "Bank payouts supported in
    # the following countries:" immediately above it. The browser's DOM text keeps one item
    # per line. Without this, the check answers the question with the wrong list name, and
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
    # change to this block and nothing else: no other setting mentions the database.
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

    # ── loading models (loader.py, gate.py) ──
    # The context length, in tokens, that every `lms load` asks for. The context is how
    # much conversation the loaded model can hold, and the memory it keeps for each token of
    # it (the attention cache, or "KV cache") grows with that number.
    #
    # This setting does NOT bound the memory a load takes on the app version measured on
    # 13 October 2026 (Bionic 1.1.1, engine llama.cpp 2.56.0). The number is accepted and
    # ignored: the app sizes the context to the free graphics memory by itself.
    #
    #     --context-length 8192    ->  loaded with a context of 243,968 tokens
    #     --context-length 16384   ->  243,968
    #     --context-length 32768   ->  243,968
    #
    # It is still sent, because a later version may honour it, and 32768 is the 14B
    # model's own maximum, so one number would serve both models. What was really loaded
    # is read back after every load and reported (`LoadReport` in loader.py). What breaks
    # if this comment is not believed: a reader takes 32768 as the size of the load, and
    # reads a card that is full by design as a fault.
    load_context_length: int = Field(
        default=DEFAULT_LOAD_CONTEXT_LENGTH, gt=0,
        description="Tokens of context passed as --context-length on every `lms load`. "
                    "A request, not a limit: the app may load a different context.",
    )
    # The slowest a freshly loaded model may answer and still be kept, in tokens a second.
    # After every load the loader asks for one short fixed completion and times it
    # (speed.py). Why speed and not free memory: every load fills the card on purpose, so
    # free memory is a few hundred MiB whether the load is healthy or not, and only the
    # speed tells them apart. Measured on the 16 GB card:
    #
    #     healthy load, one prediction slot   ->  63 to 109 tokens a second
    #     overfilled load, four slots         ->  one answer in eight minutes, under 1
    #
    # 20 sits far from both, so a healthy model on a busy desktop passes and a load that
    # has spilled into system memory does not. What breaks without the floor: a spilled
    # load is kept, nothing reports it, and a day of work runs fifteen times slower.
    min_tokens_per_second: float = Field(
        default=DEFAULT_MIN_TOKENS_PER_SECOND, ge=0,
        description="A loaded model answering slower than this is unloaded again.",
    )
    # The least graphics memory that must be free after a load, in MiB. Small on purpose:
    # a healthy load was measured with 148 MiB free, so the 300 MiB this replaced refused
    # loads that ran at full speed. It is kept only for a card that is truly exhausted,
    # where the next allocation by any program spills to system memory.
    min_free_vram_mib: int = Field(
        default=DEFAULT_MIN_FREE_VRAM_MIB, ge=0,
        description="A load that leaves less graphics memory free than this is unloaded.",
    )
    # How long a process waits for another process to finish loading a model before it
    # gives up, in seconds. Several processes share one card (each agent has its own copy
    # of the MCP server), and only one of them may unload and load at a time: the lock
    # file `model-load.lock` in the data directory decides whose turn it is (lockfile.py).
    # A load with its speed check takes about 15 seconds, so three minutes is a holder
    # that has gone wrong, and the waiting process refuses with a sentence naming it.
    # What breaks without a bound: one hung load would hang every agent with it.
    model_load_lock_wait_s: float = Field(default=DEFAULT_LOAD_LOCK_WAIT_S, gt=0)
    # A model-load lock older than this is treated as left behind and taken over. Longer
    # than the longest honest load: `lms load` is given five minutes and the speed check
    # two. A holder that has died is taken over sooner, as soon as it is seen to be gone.
    model_load_lock_stale_s: float = Field(default=600.0, gt=0)
    # Whether the client makes sure the model it is about to name is loaded before it sends
    # a request (gate.py). On by default, because of what was measured on the night of 7 to
    # 8 October 2026 without it: a request reached the running server while no model was
    # loaded, the server loaded the model itself ("just-in-time loading") with four
    # prediction slots and none of this toolkit's checks, 190 MiB was left free and the
    # answer crawled for eight minutes. With the gate on, a missing model is loaded
    # through `ModelLoader` or the request is refused and nothing is sent.
    #
    # Switch it off (LOCAL_LLM_GUARD_MODEL_LOADS=false, or Settings(guard_model_loads=False))
    # only for a server that is not LM Studio, since the check uses the `lms` command, or for
    # a script that loads its models itself and wants nothing in front of each request. The
    # benchmark script is the second kind, but it can also leave the gate on: it loads each
    # model before replaying, so the gate finds the model loaded and costs one cached lookup.
    guard_model_loads: bool = Field(
        default=True,
        description="Load a missing model through the guarded loader before each request.",
    )

    # ── unattended server start (server_start.py) ──
    # The models the unattended starter tries, in this order, when no model is named on its
    # command line. It stops at the first one that loads inside the guards, and says which
    # one that was. The 14B is the better model; the 9B is the one that still fits on a
    # day when other programs hold part of the card. `--model KEY` ignores this list and
    # means exactly that one model, with no fallback. An environment variable has to be a
    # JSON list, which is how pydantic-settings reads any list-valued setting:
    #
    #     LOCAL_LLM_UNATTENDED_MODELS=["qwen/qwen3-14b","qwen/qwen3.5-9b"]
    #
    # At least one entry is required: with an empty list the starter would have nothing to
    # try and nothing honest to report.
    unattended_models: tuple[str, ...] = Field(
        default=("qwen/qwen3-14b", "qwen/qwen3.5-9b"), min_length=1,
        description="Model keys the unattended starter tries in order.",
    )
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
    # How long, in all, the starter keeps trying to bring the server up before it gives
    # up: every `lms server start` call and every wait between them count against it.
    # Measured on 13 October 2026: from cold the service behind `lms` (its "daemon") needs
    # about a minute, and until then each call reports that it timed out. Two minutes is
    # that minute twice over. It was 30 seconds, which gave up on a start that was working.
    server_start_wait_s: float = Field(default=120.0, gt=0)
    # A lock file older than this is treated as left behind by a crashed starter.
    server_start_lock_stale_s: float = Field(default=600.0, gt=0)

    # ── live Claude agent view ──
    # How long a *finished* agent stays on the dashboard. Finished agents linger instead
    # of vanishing because the dashboard detects a completion by watching a row change
    # status, and a row that disappears the instant it finishes can never be seen to
    # finish: the notification would never fire.
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
        as a property so callers that only ever run SQLite, the smoke test, a status
        line, need not reach into the nested config.
        """
        return self.database.resolved_sqlite_path(self.data_dir)

    @property
    def server_start_lock_path(self) -> Path:
        return self.data_dir / "server-start.lock"

    @property
    def model_load_lock_path(self) -> Path:
        """The file every process takes before it unloads or loads a model.

        In the data directory because that is the one folder every copy of the toolkit
        on a machine shares. Two copies pointed at different data directories would not
        see each other's lock, and would also have two call histories.
        """
        return self.data_dir / "model-load.lock"

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
