"""Smart context manager — incremental context injection for web backends.

The key insight: LLM APIs that don't support multi-turn natively (Gemini Web,
Copilot Web) receive the ENTIRE conversation history flattened into a single
prompt on every turn. This wastes tokens and adds latency.

Solution:
  - Track context sent per conversation
  - First message: full context (project tree, files, system prompt)
  - Subsequent messages: only new/changed content + a compact reference
  - Support context budgeting (trim to fit model token limits)

Usage:
    ctx_mgr = ContextManager()
    prompt = ctx_mgr.build_prompt(conversation_id, messages, backend_type)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("super-coder.context")

# Approximate token counts per model family (for budget calculations)
_TOKEN_LIMITS: dict[str, int] = {
    "gemini_web": 128_000,
    "copilot_web": 32_000,
    "cli": 200_000,
    "openai_compatible": 128_000,
    "gemini": 1_000_000,
    "default": 64_000,
}

# Characters per token (rough estimate — 4 chars/token for English text)
_CHARS_PER_TOKEN = 4


@dataclass
class ContextSnapshot:
    """What was sent to the model in a particular turn."""
    turn_index: int
    content_hash: str  # hash of all content parts
    system_hash: str   # hash of system message specifically
    context_hashes: list[str]  # hashes of individual context items
    timestamp: float
    char_count: int


@dataclass
class ConversationState:
    """Tracks what context has been sent in a conversation."""
    conversation_id: str
    turns: list[ContextSnapshot] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_first_turn(self) -> bool:
        return len(self.turns) == 0


class ContextManager:
    """Manages incremental context for conversations with web backends."""

    def __init__(self, max_conversations: int = 100):
        self._conversations: dict[str, ConversationState] = {}
        self._max_conversations = max_conversations

    def _get_or_create(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id,
            )
            self._evict_old()
        state = self._conversations[conversation_id]
        state.last_active = time.time()
        return state

    def _evict_old(self) -> None:
        """Remove oldest conversations when we exceed max."""
        if len(self._conversations) <= self._max_conversations:
            return
        sorted_convs = sorted(
            self._conversations.items(),
            key=lambda x: x[1].last_active,
        )
        while len(self._conversations) > self._max_conversations:
            old_id, _ = sorted_convs.pop(0)
            del self._conversations[old_id]
            logger.debug("Evicted old conversation: %s", old_id)

    @staticmethod
    def _hash_content(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def build_prompt(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        backend_type: str,
    ) -> tuple[str, list[bytes]]:
        """Build an optimized prompt with smart context injection.

        Returns (prompt_text, image_files) — same interface as _flatten_messages
        but with incremental context optimization.

        Strategy:
          - Turn 1: Full prompt (system + context + user message)
          - Turn 2+: Compact format that includes:
            • A brief "continuation" marker referencing previous context
            • Only NEW user messages (not previously sent ones)
            • Any CHANGED context items (detected via hash comparison)
        """
        import base64

        state = self._get_or_create(conversation_id)
        token_limit = _TOKEN_LIMITS.get(backend_type, _TOKEN_LIMITS["default"])
        char_budget = token_limit * _CHARS_PER_TOKEN

        # Separate messages by role and extract content
        system_parts: list[str] = []
        context_parts: list[str] = []  # Context items that repeat (file contents, etc.)
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        image_files: list[bytes] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            # Handle multi-part content (images, etc.)
            text, files = self._extract_content(content)
            image_files.extend(files)

            if role == "system":
                system_parts.append(text)
            elif role == "user":
                # Detect context injections vs actual user messages
                # Context items from Continue are typically long blocks before the user query
                if self._looks_like_context(text):
                    context_parts.append(text)
                else:
                    user_parts.append(text)
            elif role == "assistant":
                assistant_parts.append(text)

        # Compute hashes
        system_text = "\n".join(system_parts)
        system_hash = self._hash_content(system_text)
        context_hashes = [self._hash_content(c) for c in context_parts]
        all_content = system_text + "\n".join(context_parts) + "\n".join(user_parts)
        content_hash = self._hash_content(all_content)

        if state.is_first_turn:
            # First turn: send everything
            prompt = self._build_full_prompt(
                system_parts, context_parts, user_parts, assistant_parts,
            )
            logger.info(
                "Conversation %s: first turn, full context (%d chars)",
                conversation_id[:8], len(prompt),
            )
        else:
            # Subsequent turns: incremental
            prev = state.turns[-1]

            # Determine what changed
            system_changed = system_hash != prev.system_hash
            new_context = [
                c for c, h in zip(context_parts, context_hashes)
                if h not in prev.context_hashes
            ]
            removed_context_count = sum(
                1 for h in prev.context_hashes if h not in context_hashes
            )

            prompt = self._build_incremental_prompt(
                system_parts if system_changed else [],
                new_context,
                removed_context_count,
                user_parts,
                assistant_parts,
                state.turn_count,
            )
            logger.info(
                "Conversation %s: turn %d, incremental (%d chars, "
                "%d new context items, system_changed=%s)",
                conversation_id[:8], state.turn_count + 1, len(prompt),
                len(new_context), system_changed,
            )

        # Apply budget trimming
        if len(prompt) > char_budget:
            prompt = self._trim_to_budget(prompt, char_budget)
            logger.warning(
                "Conversation %s: trimmed to %d chars (budget=%d)",
                conversation_id[:8], len(prompt), char_budget,
            )

        # Record this turn
        state.turns.append(ContextSnapshot(
            turn_index=state.turn_count,
            content_hash=content_hash,
            system_hash=system_hash,
            context_hashes=context_hashes,
            timestamp=time.time(),
            char_count=len(prompt),
        ))

        return prompt, image_files

    @staticmethod
    def _extract_content(content: Any) -> tuple[str, list[bytes]]:
        """Extract text and image files from message content."""
        import base64

        if isinstance(content, str):
            return content, []

        if not isinstance(content, list):
            return str(content), []

        text_pieces: list[str] = []
        files: list[bytes] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_pieces.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        try:
                            _, b64 = url.split(",", 1)
                            files.append(base64.b64decode(b64))
                        except Exception:
                            text_pieces.append("[image attachment]")
                    elif url:
                        files.append(url.encode())
            else:
                text_pieces.append(str(part))

        return "\n".join(text_pieces), files

    @staticmethod
    def _looks_like_context(text: str) -> bool:
        """Heuristic: detect if text is a context injection vs user query.

        Continue.dev injects context as multi-line blocks with file paths,
        code blocks, or terminal output. User queries are typically shorter.
        """
        # Long blocks with code fences are likely context
        if len(text) > 500 and ("```" in text or "File:" in text[:100]):
            return True

        # Blocks starting with common context provider prefixes
        context_indicators = [
            "Contents of ", "File: ", "Terminal output",
            "Diff:", "Selected code", "Folder: ",
            "<context>", "<file_contents>", "<code>",
        ]
        for indicator in context_indicators:
            if text.strip().startswith(indicator):
                return True

        return False

    @staticmethod
    def _build_full_prompt(
        system_parts: list[str],
        context_parts: list[str],
        user_parts: list[str],
        assistant_parts: list[str],
    ) -> str:
        """Build a full prompt with all context."""
        sections: list[str] = []

        if system_parts:
            sections.append("SYSTEM:\n" + "\n".join(system_parts))

        if context_parts:
            sections.append("CONTEXT:\n" + "\n---\n".join(context_parts))

        # Interleave user and assistant turns
        for i, user_msg in enumerate(user_parts):
            sections.append(f"USER: {user_msg}")
            if i < len(assistant_parts):
                sections.append(f"ASSISTANT: {assistant_parts[i]}")

        return "\n\n".join(sections)

    @staticmethod
    def _build_incremental_prompt(
        system_parts: list[str],
        new_context: list[str],
        removed_context_count: int,
        user_parts: list[str],
        assistant_parts: list[str],
        turn_index: int,
    ) -> str:
        """Build an incremental prompt referencing previous context."""
        sections: list[str] = []

        # Continuation marker
        sections.append(
            f"[Continuing conversation from turn {turn_index}. "
            f"Previous context is still active.]"
        )

        if system_parts:
            sections.append("UPDATED SYSTEM:\n" + "\n".join(system_parts))

        if new_context:
            sections.append(
                f"NEW CONTEXT (added since last message):\n"
                + "\n---\n".join(new_context)
            )

        if removed_context_count:
            sections.append(
                f"[{removed_context_count} context item(s) removed since last message.]"
            )

        # Only the latest user messages (skip already-sent ones)
        if user_parts:
            sections.append(f"USER: {user_parts[-1]}")

        return "\n\n".join(sections)

    @staticmethod
    def _trim_to_budget(prompt: str, char_budget: int) -> str:
        """Trim prompt to fit within character budget.

        Strategy: keep the beginning (system/context) and end (user query),
        trim the middle (old conversation history).
        """
        if len(prompt) <= char_budget:
            return prompt

        # Reserve 20% for the end (latest user message), 80% for beginning
        end_budget = int(char_budget * 0.2)
        start_budget = char_budget - end_budget - 50  # 50 for truncation marker

        start = prompt[:start_budget]
        end = prompt[-end_budget:]

        return start + "\n\n[... middle of conversation trimmed for length ...]\n\n" + end

    def reset_conversation(self, conversation_id: str) -> None:
        """Reset state for a conversation (e.g., when user starts new session)."""
        self._conversations.pop(conversation_id, None)

    def get_stats(self, conversation_id: str) -> dict | None:
        """Get conversation statistics."""
        state = self._conversations.get(conversation_id)
        if not state:
            return None
        return {
            "conversation_id": conversation_id,
            "turn_count": state.turn_count,
            "created_at": state.created_at,
            "last_active": state.last_active,
            "total_chars_sent": sum(t.char_count for t in state.turns),
        }
