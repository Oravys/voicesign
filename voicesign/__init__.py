# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
VoiceSign - Open-source voice signing for individuals.
=======================================================

The first open-source tool that lets anyone cryptographically sign their
voice recordings with an inaudible watermark tied to their identity.

Quick start::

    import voicesign

    # Sign a recording
    with open("my_voice.wav", "rb") as f:
        audio = f.read()

    signed = voicesign.sign(audio, identity="Your Name", salt="your-secret")

    with open("signed.wav", "wb") as f:
        f.write(signed)

    # Verify a recording
    with open("signed.wav", "rb") as f:
        audio = f.read()

    result = voicesign.verify(audio, identity="Your Name", salt="your-secret")
    print(result["detected"])     # True
    print(result["confidence"])   # 0.95

Created by Eliot Cohen Bacrie.
Powered by Oravys Inc. (https://oravys.com) - Voice Intelligence.
"""

import sys as _sys

from voicesign.core import sign, sign_with_receipt, verify

try:
    from voicesign.crypto import generate_keypair
except ImportError:
    generate_keypair = None  # type: ignore[assignment]

__version__ = "0.2.0"
__author__ = "Eliot Cohen Bacrie"
__license__ = "Apache-2.0"
__url__ = "https://github.com/oravys/voicesign"

__all__ = ["sign", "sign_with_receipt", "verify", "generate_keypair"]

# ---------------------------------------------------------------------------
# One-time attribution notice (printed once per Python session)
# ---------------------------------------------------------------------------

_NOTICE_ATTR = "_voicesign_notice_shown"

if not getattr(_sys.modules[__name__], _NOTICE_ATTR, False):
    _sys.stderr.write(f"VoiceSign v{__version__} - Powered by Oravys Inc. (https://oravys.com)\n")
    setattr(_sys.modules[__name__], _NOTICE_ATTR, True)
