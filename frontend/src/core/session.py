from dataclasses import dataclass, field
from datetime import datetime

from adapters.base import Message


@dataclass
class Session:
    """In-memory student session. Will be persisted to SQLite in the data layer."""
    created_at: datetime = field(default_factory=datetime.utcnow)
    history: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))

    def to_messages(self, system_prompt: str) -> list[Message]:
        return [Message(role="system", content=system_prompt)] + self.history

    def clear(self) -> None:
        self.history.clear()
