# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
VoiceSign Crypto - Ed25519 asymmetric cryptography for voice signing.

Provides key generation, digital signing, and verification using Ed25519
elliptic-curve signatures. This replaces the shared-secret (salt + identity)
model with proper asymmetric crypto so that:

- Only the private key holder can sign (embed watermark).
- Anyone with the public key can verify (no secret needed).
- Non-repudiation: the signer cannot deny having signed.

The public key is also used to derive the PN-sequence seed for
spread-spectrum watermarking, replacing the old salt-based derivation.

Dependencies: cryptography >= 41.0.0
"""

import hashlib
import struct

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def generate_keypair() -> tuple[bytes, bytes]:
    """
    Generate an Ed25519 keypair.

    Returns
    -------
    tuple[bytes, bytes]
        (private_key_bytes, public_key_bytes) where private_key_bytes is the
        raw 32-byte private key and public_key_bytes is the raw 32-byte
        public key.
    """
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


# ---------------------------------------------------------------------------
# Seed derivation from public key
# ---------------------------------------------------------------------------


def derive_seed_from_pubkey(public_key: bytes) -> int:
    """
    Derive a deterministic PRNG seed from an Ed25519 public key.

    This replaces the old ``_derive_seed(salt, identity)`` approach. The
    seed is derived from SHA-256 of the raw public key bytes, truncated
    to a 32-bit unsigned integer for use with numpy RandomState.

    Parameters
    ----------
    public_key : bytes
        Raw 32-byte Ed25519 public key.

    Returns
    -------
    int
        A 32-bit unsigned integer seed.

    Raises
    ------
    ValueError
        If the public key is not 32 bytes.
    """
    if len(public_key) != 32:
        raise ValueError(
            f"Ed25519 public key must be exactly 32 bytes, got {len(public_key)}"
        )
    digest = hashlib.sha256(public_key).digest()
    seed = struct.unpack(">I", digest[:4])[0]
    return seed


# ---------------------------------------------------------------------------
# Payload construction and signing
# ---------------------------------------------------------------------------


def _build_payload(identity: str, audio_hash: bytes, timestamp: str) -> bytes:
    """
    Build a canonical payload for signing or verification.

    The payload is the concatenation of:
    - UTF-8 encoded identity
    - a null byte separator
    - the raw audio hash bytes
    - a null byte separator
    - UTF-8 encoded timestamp

    Parameters
    ----------
    identity : str
        Signer identity string.
    audio_hash : bytes
        Hash of the audio content (e.g. SHA-256 digest).
    timestamp : str
        ISO-8601 timestamp string.

    Returns
    -------
    bytes
        The canonical payload ready for signing.
    """
    return identity.encode("utf-8") + b"\x00" + audio_hash + b"\x00" + timestamp.encode("utf-8")


def sign_payload(
    private_key: bytes,
    identity: str,
    audio_hash: bytes,
    timestamp: str,
) -> bytes:
    """
    Produce an Ed25519 signature over a canonical payload.

    The payload binds together the signer identity, a hash of the audio
    content, and a timestamp, then signs the result with the private key.

    Parameters
    ----------
    private_key : bytes
        Raw 32-byte Ed25519 private key.
    identity : str
        Signer identity string (name, email, etc.).
    audio_hash : bytes
        Hash of the audio content (e.g. SHA-256 digest).
    timestamp : str
        ISO-8601 timestamp string.

    Returns
    -------
    bytes
        64-byte Ed25519 signature.

    Raises
    ------
    ValueError
        If the private key is not 32 bytes or any input is empty.
    """
    if len(private_key) != 32:
        raise ValueError(
            f"Ed25519 private key must be exactly 32 bytes, got {len(private_key)}"
        )
    if not identity:
        raise ValueError("Identity must not be empty")
    if not audio_hash:
        raise ValueError("Audio hash must not be empty")
    if not timestamp:
        raise ValueError("Timestamp must not be empty")

    key = Ed25519PrivateKey.from_private_bytes(private_key)
    payload = _build_payload(identity, audio_hash, timestamp)
    return key.sign(payload)


def verify_signature(
    public_key: bytes,
    identity: str,
    audio_hash: bytes,
    timestamp: str,
    signature: bytes,
) -> bool:
    """
    Verify an Ed25519 signature against a canonical payload.

    Reconstructs the same payload that was signed and checks the signature
    using the public key. Returns True if valid, False otherwise.

    Parameters
    ----------
    public_key : bytes
        Raw 32-byte Ed25519 public key.
    identity : str
        Signer identity string.
    audio_hash : bytes
        Hash of the audio content.
    timestamp : str
        ISO-8601 timestamp string.
    signature : bytes
        64-byte Ed25519 signature to verify.

    Returns
    -------
    bool
        True if the signature is valid, False otherwise.
    """
    if len(public_key) != 32:
        return False
    if not identity or not audio_hash or not timestamp or not signature:
        return False

    try:
        key = Ed25519PublicKey.from_public_bytes(public_key)
        payload = _build_payload(identity, audio_hash, timestamp)
        key.verify(signature, payload)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Key persistence (PEM files)
# ---------------------------------------------------------------------------


def save_keypair(private_key: bytes, public_key: bytes, path: str) -> None:
    """
    Save an Ed25519 keypair to PEM files.

    Creates two files:
    - ``{path}_private.pem`` -- PKCS8-encoded private key (unencrypted).
    - ``{path}_public.pem``  -- SubjectPublicKeyInfo-encoded public key.

    Parameters
    ----------
    private_key : bytes
        Raw 32-byte Ed25519 private key.
    public_key : bytes
        Raw 32-byte Ed25519 public key.
    path : str
        Base path (without extension). Two files will be created.

    Raises
    ------
    ValueError
        If key sizes are incorrect.
    """
    if len(private_key) != 32:
        raise ValueError(
            f"Ed25519 private key must be exactly 32 bytes, got {len(private_key)}"
        )
    if len(public_key) != 32:
        raise ValueError(
            f"Ed25519 public key must be exactly 32 bytes, got {len(public_key)}"
        )

    priv_key_obj = Ed25519PrivateKey.from_private_bytes(private_key)
    pub_key_obj = Ed25519PublicKey.from_public_bytes(public_key)

    priv_pem = priv_key_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = pub_key_obj.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    with open(f"{path}_private.pem", "wb") as f:
        f.write(priv_pem)
    with open(f"{path}_public.pem", "wb") as f:
        f.write(pub_pem)


def load_private_key(path: str) -> bytes:
    """
    Load an Ed25519 private key from a PEM file.

    Parameters
    ----------
    path : str
        Path to the PEM file containing a PKCS8-encoded private key.

    Returns
    -------
    bytes
        Raw 32-byte Ed25519 private key.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not contain a valid Ed25519 private key.
    """
    with open(path, "rb") as f:
        pem_data = f.read()

    try:
        key = serialization.load_pem_private_key(pem_data, password=None)
    except Exception as exc:
        raise ValueError(f"Failed to load private key from {path}: {exc}") from exc

    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"Key in {path} is not an Ed25519 private key")

    return key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def load_public_key(path: str) -> bytes:
    """
    Load an Ed25519 public key from a PEM file.

    Parameters
    ----------
    path : str
        Path to the PEM file containing a SubjectPublicKeyInfo-encoded
        public key.

    Returns
    -------
    bytes
        Raw 32-byte Ed25519 public key.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not contain a valid Ed25519 public key.
    """
    with open(path, "rb") as f:
        pem_data = f.read()

    try:
        key = serialization.load_pem_public_key(pem_data)
    except Exception as exc:
        raise ValueError(f"Failed to load public key from {path}: {exc}") from exc

    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"Key in {path} is not an Ed25519 public key")

    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
