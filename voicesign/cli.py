# VoiceSign by Oravys Inc. (https://oravys.com) - Created by Eliot Cohen Bacrie
#
# Licensed under the Apache License, Version 2.0. See LICENSE for details.
# Attribution to Oravys Inc. and Eliot Cohen Bacrie is required. See NOTICE.
#
"""
VoiceSign command-line interface.

Usage examples::

    # Sign a recording
    voicesign sign -i "Alice Johnson" recording.wav

    # Sign with a custom salt and explicit output path
    voicesign sign -i "Alice Johnson" -s "my-secret" -o signed.wav recording.wav

    # Verify a signed recording
    voicesign verify -i "Alice Johnson" signed.wav

    # Verify with a custom salt
    voicesign verify -i "Alice Johnson" -s "my-secret" signed.wav
"""

import argparse
import os
import sys

from voicesign import __version__


def _parse_args(argv=None):
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="voicesign",
        description=(
            "VoiceSign - Sign and verify voice recordings with an inaudible "
            "identity watermark. Powered by Oravys Inc."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"voicesign {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- sign subcommand ---------------------------------------------------
    sign_parser = subparsers.add_parser(
        "sign",
        help="Embed an identity watermark into an audio file.",
    )
    sign_parser.add_argument(
        "audio",
        help="Path to the input audio file (WAV, MP3, FLAC, or M4A).",
    )
    sign_parser.add_argument(
        "-i", "--identity",
        required=True,
        help="Identity string to bind to the audio (name, email, etc.).",
    )
    sign_parser.add_argument(
        "-s", "--salt",
        default=None,
        help=(
            "Secret salt for watermark generation. Using a unique salt "
            "prevents third parties from forging your signature. "
            "If omitted, a default salt is used."
        ),
    )
    sign_parser.add_argument(
        "-o", "--output",
        default=None,
        help=(
            "Output file path. Defaults to <input>_signed.wav in the same "
            "directory as the input file."
        ),
    )

    # -- verify subcommand -------------------------------------------------
    verify_parser = subparsers.add_parser(
        "verify",
        help="Check whether an audio file contains an identity watermark.",
    )
    verify_parser.add_argument(
        "audio",
        help="Path to the audio file to verify.",
    )
    verify_parser.add_argument(
        "-i", "--identity",
        required=True,
        help="Identity string to check against.",
    )
    verify_parser.add_argument(
        "-s", "--salt",
        default=None,
        help="Secret salt (must match the one used during signing).",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for the voicesign CLI."""
    args = _parse_args(argv)

    if args.command is None:
        print("VoiceSign v{} - Powered by Oravys Inc.".format(__version__))
        print("Use 'voicesign -h' for usage information.")
        sys.exit(0)

    # Import here to avoid slow numpy import when just showing help
    from voicesign.core import sign, verify

    # Validate input file
    if not os.path.isfile(args.audio):
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Detect format from extension
    ext = os.path.splitext(args.audio)[1].lstrip(".").lower()
    if not ext:
        ext = "wav"

    # Read input
    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    if args.command == "sign":
        _do_sign(args, audio_bytes, ext)
    elif args.command == "verify":
        _do_verify(args, audio_bytes, ext)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


def _do_sign(args, audio_bytes, ext):
    """Handle the sign subcommand."""
    try:
        from voicesign.core import sign

        signed_bytes = sign(
            audio_bytes,
            identity=args.identity,
            salt=args.salt,
            file_format=ext,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.audio)[0]
        output_path = f"{base}_signed.wav"

    with open(output_path, "wb") as f:
        f.write(signed_bytes)

    size_kb = len(signed_bytes) / 1024
    print(f"Signed: {output_path} ({size_kb:.1f} KB)")
    print(f"Identity: {args.identity}")
    if args.salt:
        print("Salt: (provided)")
    else:
        print("Salt: (default - consider using -s for production)")


def _do_verify(args, audio_bytes, ext):
    """Handle the verify subcommand."""
    try:
        from voicesign.core import verify

        result = verify(
            audio_bytes,
            identity=args.identity,
            salt=args.salt,
            file_format=ext,
        )
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    detected = result["detected"]
    confidence = result["confidence"]
    correlation = result["correlation"]

    if detected:
        print(f"MATCH - Voice signature detected for: {args.identity}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Correlation: {correlation:.6f}")
    else:
        print(f"NO MATCH - No signature found for: {args.identity}")
        print(f"  Correlation: {correlation:.6f}")

    sys.exit(0 if detected else 1)


if __name__ == "__main__":
    main()
