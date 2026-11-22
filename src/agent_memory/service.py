"""Authenticated, durable HTTP entry point for a single LanceDB writer."""

import hashlib
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Annotated
from uuid import UUID, uuid4

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import PyJWKClientConnectionError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from agent_memory.auth import OIDCConfig, TokenVerifier
from agent_memory.embedder import Embedder, FastEmbedEmbedder
from agent_memory.graph import build_graph
from agent_memory.llm import AgentLLM, create_llm
from agent_memory.store import MemoryStore

_INSTANCE_LOCK = 7091048861303497445
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str
    memory_path: Path
    oidc: OIDCConfig
    model_cache: Path
    max_conversation_messages: int = 40

    def __post_init__(self) -> None:
        if self.max_conversation_messages < 2:
            raise ValueError("max_conversation_messages must be at least 2")

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        required = (
            "DATABASE_URL",
            "MEMORY_DB_PATH",
            "OIDC_ISSUER",
            "OIDC_AUDIENCE",
            "OIDC_JWKS_URL",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Missing service settings: {', '.join(missing)}")
        return cls(
            database_url=os.environ["DATABASE_URL"],
            memory_path=Path(os.environ["MEMORY_DB_PATH"]),
            oidc=OIDCConfig(
                os.environ["OIDC_ISSUER"], os.environ["OIDC_AUDIENCE"], os.environ["OIDC_JWKS_URL"]
            ),
            model_cache=Path(os.environ.get("EMBEDDER_CACHE_DIR", "data/models")),
            max_conversation_messages=int(os.environ.get("MAX_CONVERSATION_MESSAGES", "40")),
        )


def _lock_key(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode(), digest_size=8).digest(), "big", signed=True
    )


class UserService:
    """Owns the one local memory writer and durable Postgres thread state."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        embedder: Embedder | None = None,
        llm: AgentLLM | None = None,
    ) -> None:
        if os.environ.get("LANGGRAPH_STRICT_MSGPACK", "true").lower() != "true":
            raise ValueError("LANGGRAPH_STRICT_MSGPACK must be true for the service")
        os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
        self.pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=8,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
        self._guard = None
        try:
            self._guard = self.pool.getconn()
            row = self._guard.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired", (_INSTANCE_LOCK,)
            ).fetchone()
            if not row["acquired"]:
                raise RuntimeError("Another service instance already owns this memory database")
            with self.pool.connection() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS memory_conversations (
                        id uuid PRIMARY KEY,
                        owner_id text NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now()
                    )"""
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS memory_conversations_owner_idx "
                    "ON memory_conversations (owner_id, created_at DESC)"
                )
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS memory_message_results (
                        conversation_id uuid NOT NULL
                            REFERENCES memory_conversations(id) ON DELETE CASCADE,
                        request_id uuid NOT NULL,
                        body_digest text NOT NULL,
                        answer text NOT NULL,
                        PRIMARY KEY (conversation_id, request_id)
                    )"""
                )
            self.checkpointer = PostgresSaver(self.pool)
            self.checkpointer.setup()
            embedder = embedder or FastEmbedEmbedder(cache_dir=str(settings.model_cache))
            self.store = MemoryStore(settings.memory_path, embedder)
            self._memory_lock = RLock()
            self.graph = build_graph(
                self.store, llm or create_llm(), checkpointer=self.checkpointer
            )
            self.max_conversation_messages = settings.max_conversation_messages
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._guard is not None:
            self.pool.putconn(self._guard)
            self._guard = None
        self.pool.close()

    def health(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT 1").fetchone()

    def create_conversation(self, owner: str) -> dict:
        conversation_id = uuid4()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_lock_key(owner),))
            row = conn.execute(
                "INSERT INTO memory_conversations (id, owner_id) VALUES (%s, %s) "
                "RETURNING id, created_at",
                (conversation_id, owner),
            ).fetchone()
        return {"id": row["id"], "created_at": row["created_at"]}

    def list_conversations(self, owner: str) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(
                "SELECT id, created_at FROM memory_conversations WHERE owner_id = %s "
                "ORDER BY created_at DESC, id DESC LIMIT 100",
                (owner,),
            ).fetchall()

    def _require_owned(self, conn, owner: str, conversation_id: UUID) -> None:
        row = conn.execute(
            "SELECT owner_id FROM memory_conversations WHERE id = %s FOR UPDATE",
            (conversation_id,),
        ).fetchone()
        if row is None or row["owner_id"] != owner:
            raise HTTPException(status_code=404, detail="Conversation not found")

    def send_message(self, owner: str, conversation_id: UUID, request_id: UUID, text: str) -> str:
        config = {"configurable": {"thread_id": str(conversation_id)}}
        digest = hashlib.sha256(text.encode()).hexdigest()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_lock_key(owner),))
            self._require_owned(conn, owner, conversation_id)
            existing = conn.execute(
                "SELECT body_digest, answer FROM memory_message_results "
                "WHERE conversation_id = %s AND request_id = %s",
                (conversation_id, request_id),
            ).fetchone()
            if existing is not None:
                if existing["body_digest"] != digest:
                    raise HTTPException(
                        status_code=409, detail="Request ID reused with different text"
                    )
                return existing["answer"]

            snapshot = self.graph.get_state(config)
            messages = snapshot.values.get("messages", [])
            prior_index = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if isinstance(message, HumanMessage) and message.id == str(request_id)
                ),
                None,
            )
            if prior_index is not None:
                if messages[prior_index].content != text:
                    raise HTTPException(
                        status_code=409, detail="Request ID reused with different text"
                    )
                if snapshot.next:
                    with self._memory_lock:
                        result = self.graph.invoke(None, config=config)
                    messages = result["messages"]
            else:
                if snapshot.next:
                    raise HTTPException(
                        status_code=409, detail="Previous turn incomplete; retry its request ID"
                    )
                if len(messages) + 2 > self.max_conversation_messages:
                    raise HTTPException(
                        status_code=409,
                        detail="Conversation limit reached; start a new conversation",
                    )
                with self._memory_lock:
                    result = self.graph.invoke(
                        {
                            "user_id": owner,
                            "messages": [HumanMessage(content=text, id=str(request_id))],
                        },
                        config=config,
                    )
                messages = result["messages"]
                prior_index = next(
                    index
                    for index, message in enumerate(messages)
                    if isinstance(message, HumanMessage) and message.id == str(request_id)
                )

            if prior_index is None or prior_index + 1 >= len(messages):
                raise RuntimeError("Model returned no text answer")
            answer = messages[prior_index + 1]
            if not isinstance(answer, AIMessage) or not isinstance(answer.content, str):
                raise RuntimeError("Model returned no text answer")
            conn.execute(
                "INSERT INTO memory_message_results "
                "(conversation_id, request_id, body_digest, answer) VALUES (%s, %s, %s, %s)",
                (conversation_id, request_id, digest, answer.content),
            )
            return answer.content

    def messages(self, owner: str, conversation_id: UUID) -> list[dict[str, str]]:
        with self.pool.connection() as conn, conn.transaction():
            self._require_owned(conn, owner, conversation_id)
            state = self.graph.get_state({"configurable": {"thread_id": str(conversation_id)}})
        return [
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content,
            }
            for message in state.values.get("messages", [])
            if isinstance(message, (HumanMessage, AIMessage)) and isinstance(message.content, str)
        ]

    def memories(self, owner: str, limit: int) -> list[dict]:
        with self._memory_lock:
            memories = self.store.search(owner, "", k=limit, mode="recency")
        return [
            {
                "id": memory.id,
                "text": memory.text,
                "kind": memory.kind,
                "topic": memory.topic,
                "created_at": memory.created_at,
            }
            for memory in memories
        ]

    def delete_me(self, owner: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_lock_key(owner),))
            rows = conn.execute(
                "SELECT id FROM memory_conversations WHERE owner_id = %s FOR UPDATE", (owner,)
            ).fetchall()
            with self._memory_lock:
                self.store.delete_user(owner)
                for row in rows:
                    self.checkpointer.delete_thread(str(row["id"]))
            conn.execute("DELETE FROM memory_conversations WHERE owner_id = %s", (owner,))


class NewMessage(BaseModel):
    request_id: UUID
    text: str = Field(min_length=1, max_length=10000)


def get_subject(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        return request.app.state.verifier.subject(credentials.credentials)
    except PyJWKClientConnectionError as exc:
        raise HTTPException(status_code=503, detail="Identity provider unavailable") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc


def create_app(
    settings: ServiceSettings | None = None,
    *,
    runtime_factory: Callable[[ServiceSettings], UserService] = UserService,
) -> FastAPI:
    settings = settings or ServiceSettings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.verifier = TokenVerifier(settings.oidc)
        app.state.service = runtime_factory(settings)
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(title="Agent Memory API", lifespan=lifespan)

    @app.get("/healthz")
    def health(request: Request):
        request.app.state.service.health()
        return {"status": "ok"}

    @app.post("/v1/conversations", status_code=201)
    def create_conversation(request: Request, owner: str = Depends(get_subject)):
        return request.app.state.service.create_conversation(owner)

    @app.get("/v1/conversations")
    def list_conversations(request: Request, owner: str = Depends(get_subject)):
        return request.app.state.service.list_conversations(owner)

    @app.post("/v1/conversations/{conversation_id}/messages")
    def send_message(
        conversation_id: UUID,
        message: NewMessage,
        request: Request,
        owner: str = Depends(get_subject),
    ):
        return {
            "answer": request.app.state.service.send_message(
                owner, conversation_id, message.request_id, message.text
            )
        }

    @app.get("/v1/conversations/{conversation_id}/messages")
    def get_messages(conversation_id: UUID, request: Request, owner: str = Depends(get_subject)):
        return request.app.state.service.messages(owner, conversation_id)

    @app.get("/v1/memories")
    def get_memories(request: Request, limit: int = 20, owner: str = Depends(get_subject)):
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        return request.app.state.service.memories(owner, limit)

    @app.delete("/v1/me", status_code=204)
    def delete_me(request: Request, owner: str = Depends(get_subject)):
        request.app.state.service.delete_me(owner)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
