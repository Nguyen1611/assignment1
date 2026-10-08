"""Versioned prompt templates and message builders (Req 5.2-5.4, 16.2)."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str  # contains {schema} and {question} placeholders


class UnknownPromptVersion(KeyError):
    """Raised when a prompt version is not in PROMPTS."""

    def __init__(self, version: str, available: list[str]):
        self.version = version
        self.available = available
        super().__init__(f"Unknown prompt version {version!r}; available: {available}")

    def __str__(self) -> str:  # KeyError's default repr-quotes the message
        return self.args[0]


PROMPTS: dict[str, PromptTemplate] = {
    "v1": PromptTemplate(
        system=(
            "You are an expert SQLite developer. Given a database schema and a "
            "question, write one SQLite query that answers the question."
        ),
        user=(
            "### Database schema\n{schema}\n\n### Question\n{question}\n\n"
            "Respond with the SQL query in a single ```sql fenced code block."
        ),
    ),
}

RETRY_USER_TEMPLATE = (
    "The query above failed when executed on the database.\n"
    "Execution feedback: {feedback}\n"
    "Fix the query. Respond with the corrected SQL in a single ```sql fenced code block."
)


def get_template(version: str) -> PromptTemplate:
    try:
        return PROMPTS[version]
    except KeyError:
        raise UnknownPromptVersion(version, sorted(PROMPTS)) from None


def build_messages(question: str, schema_text: str, version: str) -> list[dict]:
    """Build [system, user] chat messages. Placeholders are substituted literally
    (no str.format) so braces in schema/question are safe."""
    t = get_template(version)
    head, _, tail = t.user.partition("{schema}")
    mid, _, rest = tail.partition("{question}")
    user = head + schema_text + mid + question + rest
    return [
        {"role": "system", "content": t.system},
        {"role": "user", "content": user},
    ]


def build_retry_messages(base: list[dict], failed_sql: str, feedback: str) -> list[dict]:
    """Return a new list: base + assistant(failed sql) + user(feedback). Does not mutate base."""
    msgs = copy.deepcopy(base)
    msgs.append({"role": "assistant", "content": f"```sql\n{failed_sql}\n```"})
    msgs.append({"role": "user", "content": RETRY_USER_TEMPLATE.replace("{feedback}", feedback)})
    return msgs


def messages_sha256(messages: list[dict]) -> str:
    """Stable SHA-256 over the canonical JSON encoding of the messages."""
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
