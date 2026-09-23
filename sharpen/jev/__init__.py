"""Jev (TypeSafe System One) integration for the ATL x Jev workstream.

See ``docs/research/atl_jev_strategy_plan_2026-09-23.md``. Jev answers TYPED questions about a text
state and never generates text, so everything here is a question set, a state renderer, or a cache —
never a prompt whose free-form output has to be parsed.
"""
from .client import JevAnswer, JevClient, JevError, JevUsage, question_key

__all__ = ["JevAnswer", "JevClient", "JevError", "JevUsage", "question_key"]
