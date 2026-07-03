"""Ensure the repository root is importable so tests can do `import src...`.

Running `pytest tests/` from the repo root does not automatically put the root
on sys.path (unlike `python -m pytest`). This root level conftest guarantees it,
so the absolute imports in the test modules resolve in CI and locally.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
