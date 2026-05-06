# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
Tests for voicesign.crypto -- Ed25519 asymmetric cryptography module.
"""

import os
import tempfile

import pytest

from voicesign.crypto import (
    derive_seed_from_pubkey,
    generate_keypair,
    load_private_key,
    load_public_key,
    save_keypair,
    sign_payload,
    verify_signature,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def keypair():
    """Generate a fresh Ed25519 keypair for each test."""
    return generate_keypair()


@pytest.fixture
def sample_audio_hash():
    """A deterministic 32-byte hash standing in for an audio file hash."""
    return b"\xab" * 32


@pytest.fixture
def sample_timestamp():
    return "2026-05-06T12:00:00Z"


# ---------------------------------------------------------------------------
# Keypair generation
# ---------------------------------------------------------------------------


class TestKeypairGeneration:
    def test_returns_tuple_of_bytes(self):
        priv, pub = generate_keypair()
        assert isinstance(priv, bytes)
        assert isinstance(pub, bytes)

    def test_key_sizes(self):
        priv, pub = generate_keypair()
        assert len(priv) == 32
        assert len(pub) == 32

    def test_different_each_call(self):
        pair_a = generate_keypair()
        pair_b = generate_keypair()
        assert pair_a[0] != pair_b[0], "Private keys should differ across calls"
        assert pair_a[1] != pair_b[1], "Public keys should differ across calls"


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    def test_deterministic_same_pubkey(self, keypair):
        _, pub = keypair
        seed_a = derive_seed_from_pubkey(pub)
        seed_b = derive_seed_from_pubkey(pub)
        assert seed_a == seed_b

    def test_different_pubkeys_give_different_seeds(self):
        _, pub_a = generate_keypair()
        _, pub_b = generate_keypair()
        seed_a = derive_seed_from_pubkey(pub_a)
        seed_b = derive_seed_from_pubkey(pub_b)
        assert seed_a != seed_b

    def test_returns_32bit_unsigned(self, keypair):
        _, pub = keypair
        seed = derive_seed_from_pubkey(pub)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            derive_seed_from_pubkey(b"\x00" * 16)
        with pytest.raises(ValueError, match="32 bytes"):
            derive_seed_from_pubkey(b"\x00" * 64)


# ---------------------------------------------------------------------------
# Sign / verify roundtrip
# ---------------------------------------------------------------------------


class TestSignVerifyRoundtrip:
    def test_valid_signature(self, keypair, sample_audio_hash, sample_timestamp):
        priv, pub = keypair
        sig = sign_payload(priv, "Alice", sample_audio_hash, sample_timestamp)

        assert isinstance(sig, bytes)
        assert len(sig) == 64

        assert verify_signature(pub, "Alice", sample_audio_hash, sample_timestamp, sig)

    def test_different_identity_same_key(self, keypair, sample_audio_hash, sample_timestamp):
        priv, pub = keypair
        sig_alice = sign_payload(priv, "Alice", sample_audio_hash, sample_timestamp)
        sig_bob = sign_payload(priv, "Bob", sample_audio_hash, sample_timestamp)

        assert sig_alice != sig_bob
        assert verify_signature(pub, "Alice", sample_audio_hash, sample_timestamp, sig_alice)
        assert verify_signature(pub, "Bob", sample_audio_hash, sample_timestamp, sig_bob)

    def test_different_audio_hash(self, keypair, sample_timestamp):
        priv, pub = keypair
        hash_a = b"\x01" * 32
        hash_b = b"\x02" * 32

        sig_a = sign_payload(priv, "Alice", hash_a, sample_timestamp)
        sig_b = sign_payload(priv, "Alice", hash_b, sample_timestamp)

        assert sig_a != sig_b
        assert verify_signature(pub, "Alice", hash_a, sample_timestamp, sig_a)
        assert verify_signature(pub, "Alice", hash_b, sample_timestamp, sig_b)


# ---------------------------------------------------------------------------
# Wrong key rejection
# ---------------------------------------------------------------------------


class TestWrongKeyRejection:
    def test_wrong_public_key(self, sample_audio_hash, sample_timestamp):
        priv_a, _ = generate_keypair()
        _, pub_b = generate_keypair()

        sig = sign_payload(priv_a, "Alice", sample_audio_hash, sample_timestamp)
        assert not verify_signature(pub_b, "Alice", sample_audio_hash, sample_timestamp, sig)

    def test_swapped_keys(self, keypair, sample_audio_hash, sample_timestamp):
        priv, pub = keypair
        sig = sign_payload(priv, "Alice", sample_audio_hash, sample_timestamp)
        # Trying to verify with the private key bytes as if they were a public key
        assert not verify_signature(priv, "Alice", sample_audio_hash, sample_timestamp, sig)


# ---------------------------------------------------------------------------
# Payload tampering detection
# ---------------------------------------------------------------------------


class TestPayloadTampering:
    def test_tampered_identity(self, keypair, sample_audio_hash, sample_timestamp):
        priv, pub = keypair
        sig = sign_payload(priv, "Alice", sample_audio_hash, sample_timestamp)
        assert not verify_signature(pub, "Eve", sample_audio_hash, sample_timestamp, sig)

    def test_tampered_audio_hash(self, keypair, sample_timestamp):
        priv, pub = keypair
        original_hash = b"\xaa" * 32
        tampered_hash = b"\xbb" * 32

        sig = sign_payload(priv, "Alice", original_hash, sample_timestamp)
        assert not verify_signature(pub, "Alice", tampered_hash, sample_timestamp, sig)

    def test_tampered_timestamp(self, keypair, sample_audio_hash):
        priv, pub = keypair
        sig = sign_payload(priv, "Alice", sample_audio_hash, "2026-01-01T00:00:00Z")
        assert not verify_signature(pub, "Alice", sample_audio_hash, "2026-12-31T23:59:59Z", sig)

    def test_tampered_signature_bytes(self, keypair, sample_audio_hash, sample_timestamp):
        priv, pub = keypair
        sig = sign_payload(priv, "Alice", sample_audio_hash, sample_timestamp)
        # Flip every byte in the signature
        tampered_sig = bytes(b ^ 0xFF for b in sig)
        assert not verify_signature(pub, "Alice", sample_audio_hash, sample_timestamp, tampered_sig)

    def test_truncated_signature(self, keypair, sample_audio_hash, sample_timestamp):
        priv, pub = keypair
        sig = sign_payload(priv, "Alice", sample_audio_hash, sample_timestamp)
        assert not verify_signature(pub, "Alice", sample_audio_hash, sample_timestamp, sig[:32])

    def test_empty_signature(self, keypair, sample_audio_hash, sample_timestamp):
        _, pub = keypair
        assert not verify_signature(pub, "Alice", sample_audio_hash, sample_timestamp, b"")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_sign_rejects_empty_identity(self, keypair, sample_audio_hash, sample_timestamp):
        priv, _ = keypair
        with pytest.raises(ValueError, match="Identity"):
            sign_payload(priv, "", sample_audio_hash, sample_timestamp)

    def test_sign_rejects_empty_audio_hash(self, keypair, sample_timestamp):
        priv, _ = keypair
        with pytest.raises(ValueError, match="Audio hash"):
            sign_payload(priv, "Alice", b"", sample_timestamp)

    def test_sign_rejects_empty_timestamp(self, keypair, sample_audio_hash):
        priv, _ = keypair
        with pytest.raises(ValueError, match="Timestamp"):
            sign_payload(priv, "Alice", sample_audio_hash, "")

    def test_sign_rejects_wrong_key_length(self, sample_audio_hash, sample_timestamp):
        with pytest.raises(ValueError, match="32 bytes"):
            sign_payload(b"\x00" * 16, "Alice", sample_audio_hash, sample_timestamp)

    def test_verify_returns_false_for_wrong_key_length(self, sample_audio_hash, sample_timestamp):
        assert not verify_signature(
            b"\x00" * 16, "Alice", sample_audio_hash, sample_timestamp, b"\x00" * 64
        )

    def test_verify_returns_false_for_empty_inputs(self, keypair, sample_audio_hash, sample_timestamp):
        _, pub = keypair
        sig = b"\x00" * 64
        assert not verify_signature(pub, "", sample_audio_hash, sample_timestamp, sig)
        assert not verify_signature(pub, "Alice", b"", sample_timestamp, sig)
        assert not verify_signature(pub, "Alice", sample_audio_hash, "", sig)


# ---------------------------------------------------------------------------
# Key persistence (PEM save / load)
# ---------------------------------------------------------------------------


class TestKeyPersistence:
    def test_save_and_load_roundtrip(self, keypair):
        priv, pub = keypair
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "test_key")
            save_keypair(priv, pub, base_path)

            assert os.path.isfile(f"{base_path}_private.pem")
            assert os.path.isfile(f"{base_path}_public.pem")

            loaded_priv = load_private_key(f"{base_path}_private.pem")
            loaded_pub = load_public_key(f"{base_path}_public.pem")

            assert loaded_priv == priv
            assert loaded_pub == pub

    def test_loaded_keys_produce_valid_signatures(
        self, keypair, sample_audio_hash, sample_timestamp
    ):
        priv, pub = keypair
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "signing_key")
            save_keypair(priv, pub, base_path)

            loaded_priv = load_private_key(f"{base_path}_private.pem")
            loaded_pub = load_public_key(f"{base_path}_public.pem")

            sig = sign_payload(loaded_priv, "Alice", sample_audio_hash, sample_timestamp)
            assert verify_signature(loaded_pub, "Alice", sample_audio_hash, sample_timestamp, sig)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_private_key("/nonexistent/path/key.pem")
        with pytest.raises(FileNotFoundError):
            load_public_key("/nonexistent/path/key.pem")

    def test_load_invalid_pem(self):
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False, mode="wb") as f:
            f.write(b"this is not a PEM file")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Failed to load"):
                load_private_key(tmp_path)
            with pytest.raises(ValueError, match="Failed to load"):
                load_public_key(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_pem_files_are_text_readable(self, keypair):
        priv, pub = keypair
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "readable")
            save_keypair(priv, pub, base_path)

            with open(f"{base_path}_private.pem", "r") as f:
                content = f.read()
                assert "BEGIN PRIVATE KEY" in content
                assert "END PRIVATE KEY" in content

            with open(f"{base_path}_public.pem", "r") as f:
                content = f.read()
                assert "BEGIN PUBLIC KEY" in content
                assert "END PUBLIC KEY" in content
