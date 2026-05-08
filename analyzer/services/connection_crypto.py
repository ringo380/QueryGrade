"""Symmetric encryption for stored database-connection passwords.

Uses Fernet (AES-128-CBC + HMAC-SHA256) keyed off the ``DB_CONNECTION_KEY``
environment variable. The key must be a Fernet-format urlsafe base64 string
of 32 bytes; generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

In DEBUG, missing key falls back to a deterministic key derived from
``SECRET_KEY`` so dev environments work without extra config. In production
(``DEBUG=False``), absence of ``DB_CONNECTION_KEY`` raises at first encrypt/
decrypt call so misconfiguration surfaces loudly rather than silently
storing recoverable plaintext.
"""

from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings


class ConnectionCryptoError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _get_fernet() -> Fernet:
    raw = os.environ.get("DB_CONNECTION_KEY", "").strip()
    if raw:
        try:
            return Fernet(raw.encode() if isinstance(raw, str) else raw)
        except (ValueError, TypeError) as exc:
            raise ConnectionCryptoError(
                "DB_CONNECTION_KEY is set but not a valid Fernet key. "
                "Generate one with Fernet.generate_key()."
            ) from exc

    if not settings.DEBUG:
        raise ConnectionCryptoError(
            "DB_CONNECTION_KEY env var is required in production. "
            "Generate one with: python -c \"from cryptography.fernet import "
            "Fernet; print(Fernet.generate_key().decode())\""
        )

    derived = hashlib.sha256(settings.SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""
    token = _get_fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    try:
        return _get_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ConnectionCryptoError(
            "Stored credential could not be decrypted. The DB_CONNECTION_KEY "
            "may have changed since the value was encrypted."
        ) from exc
