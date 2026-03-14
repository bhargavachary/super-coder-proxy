"""Native web authentication — Playwright-based login flows.

Opens a real browser window for the user to sign in to Gemini / Copilot / etc.
Captures session cookies automatically and persists them to disk.
Eliminates the need for manual cookie extraction from Chrome DevTools.

Usage (from proxy endpoints):
    from .auth import AuthManager
    auth = AuthManager()
    cookies = await auth.login("gemini")   # Opens browser, waits for login
    status = auth.status("gemini")          # Check if session is valid
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("super-coder.auth")

# Where sessions are persisted
_SESSION_DIR = Path.home() / ".config" / "super-coder" / "sessions"

# Login targets — URL to open and cookies to capture
_AUTH_TARGETS: dict[str, dict[str, Any]] = {
    "gemini": {
        "name": "Google Gemini",
        "login_url": "https://gemini.google.com",
        "cookie_domain": ".google.com",
        "required_cookies": ["__Secure-1PSID"],
        "optional_cookies": ["__Secure-1PSIDTS", "__Secure-3PSID", "SAPISID", "__Secure-3PAPISID"],
        "success_indicator": "__Secure-1PSID",
        "timeout_seconds": 120,
    },
    "copilot": {
        "name": "Microsoft Copilot",
        "login_url": "https://copilot.microsoft.com",
        "cookie_domain": ".bing.com",
        "required_cookies": ["_U"],
        "optional_cookies": ["_SS", "MUID", "SRCHD"],
        "success_indicator": "_U",
        "timeout_seconds": 120,
    },
    "chatgpt": {
        "name": "ChatGPT",
        "login_url": "https://chatgpt.com",
        "cookie_domain": ".chatgpt.com",
        "required_cookies": ["__Secure-next-auth.session-token"],
        "optional_cookies": [],
        "success_indicator": "__Secure-next-auth.session-token",
        "timeout_seconds": 120,
    },
    "claude": {
        "name": "Claude.ai",
        "login_url": "https://claude.ai",
        "cookie_domain": ".claude.ai",
        "required_cookies": ["sessionKey"],
        "optional_cookies": [],
        "success_indicator": "sessionKey",
        "timeout_seconds": 120,
    },
}


@dataclass
class SessionInfo:
    target: str
    cookies: dict[str, str]
    captured_at: float  # timestamp
    expires_hint: float | None = None  # estimated expiry (if detectable)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.captured_at) / 3600

    @property
    def is_likely_fresh(self) -> bool:
        """Heuristic: sessions older than 12h may need refresh."""
        if self.expires_hint and time.time() > self.expires_hint:
            return False
        return self.age_hours < 12

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "cookies": self.cookies,
            "captured_at": self.captured_at,
            "expires_hint": self.expires_hint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionInfo:
        return cls(
            target=d["target"],
            cookies=d["cookies"],
            captured_at=d["captured_at"],
            expires_hint=d.get("expires_hint"),
        )


class AuthManager:
    """Manages browser-based authentication for web backends."""

    def __init__(self, session_dir: Path | None = None):
        self._session_dir = session_dir or _SESSION_DIR
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionInfo] = {}
        self._login_lock = asyncio.Lock()
        self._load_persisted_sessions()

    def _session_file(self, target: str) -> Path:
        return self._session_dir / f"{target}.json"

    def _load_persisted_sessions(self) -> None:
        """Load previously saved sessions from disk."""
        for f in self._session_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                session = SessionInfo.from_dict(data)
                self._sessions[session.target] = session
                logger.info(
                    "Loaded persisted session: %s (age=%.1fh, fresh=%s)",
                    session.target, session.age_hours, session.is_likely_fresh,
                )
            except Exception as e:
                logger.warning("Failed to load session %s: %s", f.name, e)

    def _persist_session(self, session: SessionInfo) -> None:
        """Save session to disk."""
        path = self._session_file(session.target)
        path.write_text(json.dumps(session.to_dict(), indent=2))
        logger.info("Persisted session: %s -> %s", session.target, path)

    def get_cookies(self, target: str) -> dict[str, str] | None:
        """Get stored cookies for a target, or None if no session."""
        session = self._sessions.get(target)
        if session and session.cookies:
            return session.cookies
        return None

    def status(self, target: str) -> dict:
        """Check authentication status for a target."""
        if target not in _AUTH_TARGETS:
            return {"target": target, "supported": False, "authenticated": False}

        session = self._sessions.get(target)
        if not session:
            return {
                "target": target,
                "supported": True,
                "authenticated": False,
                "name": _AUTH_TARGETS[target]["name"],
            }

        return {
            "target": target,
            "supported": True,
            "authenticated": True,
            "fresh": session.is_likely_fresh,
            "age_hours": round(session.age_hours, 1),
            "name": _AUTH_TARGETS[target]["name"],
            "cookie_count": len(session.cookies),
        }

    def status_all(self) -> list[dict]:
        """Get auth status for all supported targets."""
        return [self.status(t) for t in _AUTH_TARGETS]

    async def login(self, target: str) -> dict[str, str]:
        """Open a browser window for interactive login, capture cookies.

        Returns the captured cookies dict.
        Raises ValueError if target is unsupported.
        Raises TimeoutError if login not completed within timeout.
        Raises RuntimeError on browser errors.
        """
        if target not in _AUTH_TARGETS:
            raise ValueError(
                f"Unsupported auth target: {target}. "
                f"Supported: {', '.join(_AUTH_TARGETS)}"
            )

        async with self._login_lock:
            return await self._do_login(target)

    async def _do_login(self, target: str) -> dict[str, str]:
        """Perform the actual browser login flow."""
        spec = _AUTH_TARGETS[target]
        logger.info("Starting login flow for %s (%s)...", target, spec["name"])

        from playwright.async_api import async_playwright

        # Use a persistent profile so the user stays logged in across restarts
        profile_dir = self._session_dir / "browser-profiles" / target
        profile_dir.mkdir(parents=True, exist_ok=True)

        pw = await async_playwright().start()
        try:
            context = await pw.chromium.launch_persistent_context(
                str(profile_dir),
                headless=False,
                channel="chromium",
                viewport={"width": 520, "height": 700},
                args=[
                    "--disable-blink-features=AutomationControlled",
                    f"--window-size=520,700",
                ],
            )

            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(spec["login_url"], wait_until="domcontentloaded")
            logger.info("Browser opened: %s — waiting for login...", spec["login_url"])

            # Poll for the required cookie to appear
            success_cookie = spec["success_indicator"]
            timeout_s = spec["timeout_seconds"]
            start = time.time()
            captured: dict[str, str] = {}

            while time.time() - start < timeout_s:
                all_cookies = await context.cookies()

                for c in all_cookies:
                    name = c["name"]
                    if name in spec["required_cookies"] or name in spec["optional_cookies"]:
                        if c["value"]:
                            captured[name] = c["value"]

                if captured.get(success_cookie):
                    logger.info(
                        "Login successful for %s — captured %d cookies",
                        target, len(captured),
                    )
                    break

                await asyncio.sleep(1.5)
            else:
                await context.close()
                await pw.stop()
                raise TimeoutError(
                    f"Login timed out after {timeout_s}s for {spec['name']}. "
                    f"Please sign in within the browser window."
                )

            # Give a brief moment for any additional cookies to settle
            await asyncio.sleep(2)
            all_cookies = await context.cookies()
            for c in all_cookies:
                name = c["name"]
                if name in spec["required_cookies"] or name in spec["optional_cookies"]:
                    if c["value"]:
                        captured[name] = c["value"]

            await context.close()
            await pw.stop()

            # Persist
            session = SessionInfo(
                target=target,
                cookies=captured,
                captured_at=time.time(),
            )
            self._sessions[target] = session
            self._persist_session(session)

            return captured

        except TimeoutError:
            raise
        except Exception as e:
            try:
                await pw.stop()
            except Exception:
                pass
            raise RuntimeError(f"Browser login failed for {target}: {e}") from e

    async def logout(self, target: str) -> bool:
        """Clear stored session for a target."""
        self._sessions.pop(target, None)
        path = self._session_file(target)
        if path.exists():
            path.unlink()
            logger.info("Cleared session for %s", target)
            return True
        return False

    async def refresh(self, target: str) -> dict[str, str]:
        """Force re-login for a target (clears old session, opens browser)."""
        await self.logout(target)
        return await self.login(target)
