"""Shared fixtures: a tiny fake repository and a PNG screenshot."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

LOGIN_PY = '''"""Login view."""
from app.auth import check_password


def login(request):
    """Handle the sign in form."""
    user = request.form["username"]
    if not check_password(user, request.form["password"]):
        return render("login.html", error="Invalid username or password")
    return redirect("/dashboard")
'''

CART_PY = '''"""Shopping cart totals."""


def cart_total(items):
    """Add up item prices and apply the discount code."""
    return sum(item.price * item.quantity for item in items)
'''

LOGIN_HTML = """<form method="post" class="login-form">
  <input name="username" placeholder="Username">
  <input name="password" type="password">
  <button type="submit">Sign in</button>
  <p class="error">{{ error }}</p>
</form>
"""

BUG_TEMPLATE = """---
name: Bug report
about: Create a report to help us improve
title: "[Bug]: "
labels: bug, needs-triage
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
What you expected to happen.
"""

FEATURE_TEMPLATE = """---
name: Feature request
about: Suggest an idea
---

## Summary
"""


def _write(root: Path, relative: str, content: str | bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A small repository with code, tests, docs, templates and files that must be skipped."""
    root = tmp_path / "fixture_repo"
    _write(root, "src/app/login.py", LOGIN_PY)
    _write(root, "src/app/cart.py", CART_PY)
    _write(root, "src/app/templates/login.html", LOGIN_HTML)
    _write(root, "tests/test_login.py", "def test_login():\n    assert True\n")
    _write(root, "README.md", "# Fixture shop\n\nA tiny shop used in tests.\n")
    _write(root, "CONTRIBUTING.md", "Please run the tests before opening a pull request.\n")
    _write(root, ".github/ISSUE_TEMPLATE/bug_report.md", BUG_TEMPLATE)
    _write(root, ".github/ISSUE_TEMPLATE/feature_request.md", FEATURE_TEMPLATE)
    _write(root, ".github/ISSUE_TEMPLATE/config.yml", "blank_issues_enabled: false\n")
    # Everything below must be ignored by the loader.
    _write(root, "node_modules/lib/index.js", "module.exports = 'Invalid username or password';\n")
    _write(root, "package-lock.json", '{"lockfileVersion": 3}\n')
    _write(root, "static/app.min.js", "var a=1;\n")
    _write(root, "static/logo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    _write(root, "data/blob.dat", b"abc\x00def")
    return root


@pytest.fixture
def png_bytes() -> bytes:
    """A valid 40x20 PNG."""
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()
