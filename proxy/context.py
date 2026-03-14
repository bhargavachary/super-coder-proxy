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
            # First turn: send full context (smart-compressed to budget)
            prompt = self._build_full_prompt(
                system_parts, context_parts, user_parts, assistant_parts,
                char_budget=char_budget,
            )
            logger.info(
                "Conversation %s: first turn, full context (%d chars, budget=%d)",
                conversation_id[:8], len(prompt), char_budget,
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
    def _compress_system_prompt(text: str, budget: int) -> str:
        """Compress a Continue.dev system prompt to fit *budget* chars.

        Strategy: score each line by how actionable it is (must/never/always/
        do/don't/should/avoid), keep top-scoring lines while preserving original
        order.  This typically drops ~60% of the generic framing prose while
        keeping all the actual coding rules.
        """
        if len(text) <= budget:
            return text

        _PRIORITY_WORDS = {
            "must", "never", "always", "should not",
            "avoid", "rule:", "important:", "note:",
            "format:", "make sure", "ensure",
            "do not", "you must", "you should",
        }
        lines = text.splitlines()
        scored: list[tuple[int, int, str]] = []
        for i, line in enumerate(lines):
            ll = line.lower()
            score = 1
            if any(w in ll for w in _PRIORITY_WORDS):
                score += 3
            if line.startswith("#"):       # heading
                score += 2
            if line.startswith("-") or line.startswith("*"):  # bullet
                score += 1
            if len(line) > 300:            # very long → lower priority
                score -= 1
            scored.append((score, i, line))

        scored.sort(key=lambda x: (-x[0], x[1]))
        selected: set[int] = set()
        total = 0
        for _, i, line in scored:
            if total + len(line) + 1 > budget:
                break
            selected.add(i)
            total += len(line) + 1

        return "\n".join(lines[i] for i in sorted(selected))

    @staticmethod
    def _compress_file_content(text: str, max_lines: int = 100) -> str:
        """Truncate large file/code blocks to *max_lines*.

        Preserves the opening header (file path + opening fence) and appends a
        concise truncation notice so Gemini knows the file was clipped.
        """
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text

        omitted = len(lines) - max_lines
        kept = lines[:max_lines]

        # If the kept block opened a fence but never closed it, close it
        fence_count = sum(1 for l in kept if l.startswith("```"))
        suffix = f"\n... ({omitted} lines truncated)"
        if fence_count % 2 != 0:          # odd → unclosed fence
            suffix = "\n```" + suffix

        return "\n".join(kept) + suffix

    @classmethod
    def _compress_context_items(
        cls, context_parts: list[str], budget: int
    ) -> str:
        """Compress context items to fit *budget* chars.

        Priority: later items (more recently added by the user) are kept first.
        Each item is individually file-content-compressed before fitting.
        Oldest items are dropped first when the total exceeds the budget.
        """
        if not context_parts:
            return ""

        # Compress each item individually
        compressed = [cls._compress_file_content(p) for p in context_parts]

        # Check total before any dropping
        joined = "\n---\n".join(compressed)
        if len(joined) <= budget:
            return joined

        # Drop from the oldest (front) until we fit
        while len(compressed) > 1:
            compressed.pop(0)  # remove oldest
            joined = "\n---\n".join(compressed)
            if len(joined) <= budget:
                return joined

        # Single item still too large — hard-trim
        single = compressed[0]
        if len(single) > budget:
            single = single[:budget] + "\n... (truncated for length)"
        return single

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
        char_budget: int = 0,
    ) -> str:
        """Build a smart-compressed first-turn prompt.

        Compression is applied only when *char_budget* > 0 (i.e., when the raw
        prompt would exceed the model's limit).  The strategy is:
          1. Always include the latest user query in full.
          2. Compress the system prompt (keep actionable rules, drop boilerplate).
          3. Compress context files (trim large code blocks to 100 lines each).
          4. Drop oldest context items first if still over budget.

        Edge-case: Continue.dev often sends context + the user question inside a
        single user message.  _looks_like_context may classify the whole thing as
        "context", leaving user_parts empty.  We recover the user question by
        splitting the last context item at its last double-newline and treating
        the short trailing paragraph as the true user message.
        """
        # --- Recover user question if buried in last context item ---
        context_parts = list(context_parts)  # don't mutate caller's list
        user_parts = list(user_parts)
        if not user_parts and context_parts:
            last_item = context_parts[-1]
            sep = last_item.rfind("\n\n")
            if sep >= 0:
                tail = last_item[sep + 2:].strip()
                # Short, non-code tail → treat as user question
                if 0 < len(tail) < 400 and not tail.startswith("```"):
                    context_parts[-1] = last_item[:sep]
                    user_parts = [tail]

        user_msg = user_parts[-1] if user_parts else ""

        if char_budget > 0 and char_budget < 500_000:  # only compress if budget is meaningful
            # Reserve at least the user message + some header space
            overhead = len(user_msg) + 200
            remaining = max(0, char_budget - overhead)

            sys_budget = int(remaining * 0.30)
            ctx_budget = int(remaining * 0.60)

            sys_text = ContextManager._compress_system_prompt(
                "\n".join(system_parts), sys_budget
            )
            ctx_text = ContextManager._compress_context_items(context_parts, ctx_budget)
        else:
            sys_text = "\n".join(system_parts)
            ctx_text = "\n---\n".join(
                ContextManager._compress_file_content(p) for p in context_parts
            )

        sections: list[str] = []
        if sys_text.strip():
            sections.append("SYSTEM:\n" + sys_text)
        if ctx_text.strip():
            sections.append("CONTEXT:\n" + ctx_text)

        # Interleave previous user/assistant turns (all but the last user msg)
        for i, u in enumerate(user_parts[:-1]):
            sections.append(f"USER: {u}")
            if i < len(assistant_parts):
                sections.append(f"ASSISTANT: {assistant_parts[i]}")

        sections.append(f"USER: {user_msg}")
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

    def extract_latest_user_message(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[bytes]]:
        """Return only the latest user message and its attached files.

        Called on subsequent turns when a ChatSession is active — no need to
        re-send the full context history since Gemini natively maintains it.
        Falls back to an empty string if no user message is found.
        """
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return self._extract_content(msg.get("content", ""))
        return "", []

    def is_new_conversation(self, conversation_id: str) -> bool:
        """Return True if this conversation_id has never been processed.

        Non-mutating — does not create state.
        """
        state = self._conversations.get(conversation_id)
        return state is None or state.is_first_turn

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
