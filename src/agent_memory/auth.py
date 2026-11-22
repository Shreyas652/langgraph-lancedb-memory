"""OIDC access-token verification for the HTTP service."""

from dataclasses import dataclass

import jwt
from jwt import PyJWKClient


@dataclass(frozen=True)
class OIDCConfig:
    issuer: str
    audience: str
    jwks_url: str

    def __post_init__(self) -> None:
        if not self.issuer.startswith("https://") or not self.jwks_url.startswith("https://"):
            raise ValueError("OIDC issuer and JWKS URL must use HTTPS")
        if not self.audience.strip():
            raise ValueError("OIDC audience is required")


class TokenVerifier:
    def __init__(self, config: OIDCConfig) -> None:
        self.config = config
        self.jwks = PyJWKClient(config.jwks_url, cache_jwk_set=True, lifespan=300)

    def subject(self, token: str) -> str:
        """Return a stable issuer-qualified subject after full JWT validation."""
        key = self.jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            audience=self.config.audience,
            issuer=self.config.issuer,
            options={"require": ["iss", "aud", "exp", "iat", "sub"]},
        )
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject.strip():
            raise jwt.InvalidTokenError("Token subject is empty")
        return f"{self.config.issuer}|{subject}"
