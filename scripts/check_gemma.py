"""Go/no-go check: can this API key reach a Gemma model on the Gemini API?

List the Gemma model ids your key can use:

    python scripts/check_gemma.py

Send one image plus a prompt to the model named in GEMMA_MODEL and print the raw reply:

    python scripts/check_gemma.py --image shot.png --prompt "What text is visible?"

This script is standalone on purpose (it does not import buglens), so it can be
run before anything else works. It never prints the API key.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def make_client():
    """Create a Gemini API client from GEMINI_API_KEY, or exit with a clear message."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from google import genai

    return genai.Client(api_key=api_key)


def safe_text(error: Exception) -> str:
    """Error text with the API key blanked out, in case a library ever echoes it."""
    text = f"{type(error).__name__}: {error}"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    return text.replace(api_key, "***") if api_key else text


def list_gemma_models(client) -> list[str]:
    """Print every model whose name contains 'gemma' and return the ids."""
    model_ids: list[str] = []
    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        if "gemma" not in name.lower():
            continue
        model_id = name.removeprefix("models/")
        model_ids.append(model_id)
        display = getattr(model, "display_name", "") or ""
        tokens = getattr(model, "input_token_limit", None)
        actions = getattr(model, "supported_actions", None) or []
        print(f"{model_id}")
        print(f"    display name : {display}")
        print(f"    input tokens : {tokens}")
        print(f"    actions      : {', '.join(actions) if actions else 'unknown'}")
    return model_ids


def send_image_prompt(client, model_id: str, image_path: Path, prompt: str) -> str:
    """Send one image and one prompt as a single user turn and return the raw reply text."""
    from google.genai import types

    mime_type = MIME_BY_SUFFIX.get(image_path.suffix.lower())
    if mime_type is None:
        sys.exit(f"Unsupported image type '{image_path.suffix}'. Use png, jpg or webp.")
    if not image_path.is_file():
        sys.exit(f"Image not found: {image_path}")
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)
    # No system_instruction and no JSON mode: Gemma may not support either.
    response = client.models.generate_content(model=model_id, contents=[image_part, prompt])
    return response.text or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List Gemma models and optionally test an image prompt.")
    parser.add_argument("--image", type=Path, help="Path to a png/jpg/webp image to send.")
    parser.add_argument("--prompt", default="Describe this screenshot in two sentences.", help="Prompt to send.")
    args = parser.parse_args(argv)

    load_dotenv()
    client = make_client()

    print("Gemma models visible to this API key:\n")
    try:
        model_ids = list_gemma_models(client)
    except Exception as exc:  # the go/no-go script should report any failure plainly
        print(f"NO-GO: listing models failed: {safe_text(exc)}")
        return 1
    if not model_ids:
        print("NO-GO: no model with 'gemma' in its name is available to this key.")
        return 1
    print(f"\n{len(model_ids)} Gemma model(s) found. Put the exact id you want into GEMMA_MODEL in .env.")

    if args.image is None:
        print("Next: rerun with --image PATH --prompt \"...\" to test vision.")
        return 0

    model_id = os.environ.get("GEMMA_MODEL", "").strip()
    if not model_id:
        print("\nGEMMA_MODEL is not set, so the image test was skipped. Set it to one of the ids above.")
        return 1
    if model_id not in model_ids:
        print(f"\nWarning: GEMMA_MODEL='{model_id}' is not in the list above. Trying it anyway.")

    print(f"\nSending {args.image} to {model_id} ...\n")
    try:
        reply = send_image_prompt(client, model_id, args.image, args.prompt)
    except Exception as exc:
        print(f"NO-GO: the image request failed: {safe_text(exc)}")
        return 1
    print("----- raw reply -----")
    print(reply)
    print("----- end reply -----")
    print("\nGO: the model answered an image prompt." if reply.strip() else "\nNO-GO: the reply was empty.")
    return 0 if reply.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
