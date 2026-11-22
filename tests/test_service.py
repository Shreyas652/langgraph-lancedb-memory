"""Authentication and real PostgreSQL restart/isolation tests for the API."""

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from agent_memory.auth import OIDCConfig, TokenVerifier
from agent_memory.embedder import HashEmbedder
from agent_memory.llm import FakeLLM
from agent_memory.service import ServiceSettings, UserService, create_app, get_subject


def test_oidc_verifier_requires_valid_signature_claims_and_audience(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = TokenVerifier(
        OIDCConfig("https://issuer.example", "memory-api", "https://issuer.example/keys")
    )
    monkeypatch.setattr(
        verifier.jwks,
        "get_signing_key_from_jwt",
        lambda token: SimpleNamespace(key=private_key.public_key()),
    )
    claims = {
        "iss": "https://issuer.example",
        "aud": "memory-api",
        "sub": "alice",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    assert verifier.subject(jwt.encode(claims, private_key, algorithm="RS256")) == (
        "https://issuer.example|alice"
    )
    for change in (
        {"aud": "wrong"},
        {"iss": "https://attacker.example"},
        {"exp": datetime.now(UTC) - timedelta(seconds=1)},
        {"sub": ""},
    ):
        bad = claims | change
        with pytest.raises(jwt.PyJWTError):
            verifier.subject(jwt.encode(bad, private_key, algorithm="RS256"))
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(jwt.InvalidSignatureError):
        verifier.subject(jwt.encode(claims, other_key, algorithm="RS256"))


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="PostgreSQL test URL not set")
def test_api_isolation_restart_and_deletion(tmp_path):
    settings = ServiceSettings(
        database_url=os.environ["TEST_DATABASE_URL"],
        memory_path=tmp_path / "memory.lance",
        oidc=OIDCConfig("https://issuer.example", "memory-api", "https://issuer.example/keys"),
        model_cache=tmp_path / "models",
        max_conversation_messages=4,
    )
    identity = {"user": f"alice-{uuid4()}"}
    alice = identity["user"]
    bob = f"bob-{uuid4()}"

    def open_app():
        app = create_app(
            settings,
            runtime_factory=lambda config: UserService(
                config, embedder=HashEmbedder(), llm=FakeLLM()
            ),
        )
        return app

    app = open_app()
    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match="Another service instance"):
            UserService(settings, embedder=HashEmbedder(), llm=FakeLLM())
        assert client.post("/v1/conversations").status_code == 401
        app.dependency_overrides[get_subject] = lambda: identity["user"]
        conversation = client.post("/v1/conversations")
        assert conversation.status_code == 201
        conversation_id = conversation.json()["id"]
        url = f"/v1/conversations/{conversation_id}/messages"
        first_request = {"request_id": str(uuid4()), "text": "I live in Denver."}
        assert client.post(url, json=first_request).status_code == 200
        assert client.post(url, json=first_request).status_code == 200
        assert (
            client.post(
                url, json={"request_id": first_request["request_id"], "text": "Different text"}
            ).status_code
            == 409
        )
        assert (
            "Denver"
            in client.post(
                url, json={"request_id": str(uuid4()), "text": "Where do I live?"}
            ).json()["answer"]
        )
        assert len(client.get(url).json()) == 4
        assert (
            client.post(
                url, json={"request_id": str(uuid4()), "text": "One more question"}
            ).status_code
            == 409
        )
        assert [item["text"] for item in client.get("/v1/memories").json()] == ["I live in Denver"]
        identity["user"] = bob
        assert client.get(url).status_code == 404
        assert (
            client.post(url, json={"request_id": str(uuid4()), "text": "Steal history"}).status_code
            == 404
        )
        assert client.get("/v1/memories").json() == []
        identity["user"] = alice

    app = open_app()
    with TestClient(app) as client:
        app.dependency_overrides[get_subject] = lambda: identity["user"]
        assert client.post(url, json=first_request).status_code == 200
        assert len(client.get(url).json()) == 4
        assert len(client.get("/v1/memories").json()) == 1
        next_conversation = client.post("/v1/conversations").json()["id"]
        answer = client.post(
            f"/v1/conversations/{next_conversation}/messages",
            json={"request_id": str(uuid4()), "text": "Where do I live?"},
        ).json()["answer"]
        assert "Denver" in answer
        recovered_id = uuid4()
        client.app.state.service.graph.invoke(
            {
                "user_id": alice,
                "messages": [HumanMessage(content="What is my city?", id=str(recovered_id))],
            },
            config={"configurable": {"thread_id": next_conversation}},
        )
        recovered_url = f"/v1/conversations/{next_conversation}/messages"
        assert (
            "Denver"
            in client.post(
                recovered_url,
                json={"request_id": str(recovered_id), "text": "What is my city?"},
            ).json()["answer"]
        )
        assert len(client.get(recovered_url).json()) == 4
        assert client.delete("/v1/me").status_code == 204
        assert client.get("/v1/conversations").json() == []
        assert client.get("/v1/memories").json() == []
        assert client.get(url).status_code == 404
