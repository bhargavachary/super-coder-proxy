"""Super Coder — Unified LLM Proxy.

Exposes /v1/chat/completions and routes to:
  1. Local Ollama models
  2. Claude CLI
  3. GitHub Copilot CLI
  4. Gemini Web (browser cookies)
  5. Gemini API (Google AI Studio)
  6. Copilot Web (Microsoft Copilot browser session)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.responses import Response as StarletteResponse

from .auth import AuthManager
from .config import BackendConfig, ProxyConfig, load_config
from .context import ContextManager

logger = logging.getLogger("super-coder")

config: ProxyConfig
http_client: httpx.AsyncClient
gemini_client = None  # Lazy-initialized GeminiClient
_gemini_lock: asyncio.Lock
_gemini_sessions: dict[str, Any] = {}  # conversation_id -> ChatSession
auth_manager: AuthManager
context_manager: ContextManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, http_client, _gemini_lock, auth_manager, context_manager
    global _gemini_sessions
    config = load_config()
    # Optimized HTTP client: connection pooling, keep-alive, faster timeouts
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=5.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30,
        ),
        http2=True,
    )
    _gemini_lock = asyncio.Lock()
    _gemini_sessions = {}  # Reset on startup
    auth_manager = AuthManager()
    context_manager = ContextManager()

    # Auto-inject persisted auth cookies into backend configs
    _inject_auth_cookies()

    logger.info(
        "Proxy router started on %s:%s with %d backends, %d aliases",
        config.host, config.port, len(config.backends), len(config.aliases),
    )
    for name, b in config.backends.items():
        key_status = "set" if b.api_key else "missing"
        logger.info("  backend: %-12s type=%-18s key=%s", name, b.type, key_status)
    yield
    await http_client.aclose()


def _inject_auth_cookies():
    """Inject persisted auth session cookies into backend configs.

    If a backend has no explicit cookies but we have a saved session,
    auto-fill it so the user doesn't need to paste cookies manually.
    """
    # Gemini Web
    gemini_backend = config.backends.get("gemini_web")
    if gemini_backend and gemini_backend.type == "gemini_web" and not gemini_backend.cookies:
        saved = auth_manager.get_cookies("gemini")
        if saved:
            gemini_backend.cookies = saved
            logger.info("Injected saved Gemini session cookies into gemini_web backend")

    # Copilot Web
    copilot_backend = config.backends.get("copilot_web")
    if copilot_backend and copilot_backend.type == "copilot_web" and not copilot_backend.cookies:
        saved = auth_manager.get_cookies("copilot")
        if saved:
            copilot_backend.cookies = saved
            logger.info("Injected saved Copilot session cookies into copilot_web backend")


app = FastAPI(title="Super Coder", lifespan=lifespan)


# ── Latency tracking middleware ────────────────────────────────────────────────

@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    """Track and log request latency for optimization."""
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.0f}"
    if request.url.path == "/v1/chat/completions":
        logger.info("Request latency: %.0fms %s", elapsed_ms, request.url.path)
    return response


# ── Embedded Web Chat Reverse Proxy ───────────────────────────────────────────

_WEB_TARGETS = {
    "gemini": "https://gemini.google.com",
    "copilot": "https://copilot.microsoft.com",
}


@app.api_route(
    "/web/{target}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)
async def web_proxy(target: str, path: str, request: Request):
    """Reverse-proxy web chat services with frame-blocking header removal.

    Enables embedding gemini.google.com and copilot.microsoft.com in VS Code
    webview iframes by stripping X-Frame-Options and CSP frame-ancestors.
    """
    base_url = _WEB_TARGETS.get(target)
    if not base_url:
        raise HTTPException(404, f"Unknown web target: {target}")

    cookies = auth_manager.get_cookies(target) or {}

    url = f"{base_url}/{path}" if path else f"{base_url}/"
    # Forward headers but replace host/origin/referer
    forwarded_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "origin", "referer", "connection")
    }
    forwarded_headers["host"] = base_url.split("//")[1]
    forwarded_headers["referer"] = f"{base_url}/"

    query = str(request.url.query)
    if query:
        url = f"{url}?{query}"

    if request.method == "GET":
        resp = await http_client.get(
            url, headers=forwarded_headers, cookies=cookies, follow_redirects=True,
        )
    else:
        body = await request.body()
        resp = await http_client.request(
            request.method, url, headers=forwarded_headers, cookies=cookies,
            content=body, follow_redirects=True,
        )

    # Strip frame-blocking headers so VS Code webview iframe can render
    response_headers = dict(resp.headers)
    for h in (
        "x-frame-options",
        "content-security-policy",
        "content-security-policy-report-only",
    ):
        response_headers.pop(h, None)

    # Rewrite absolute paths in HTML responses to route through proxy
    content = resp.content
    content_type = response_headers.get("content-type", "")
    if "text/html" in content_type:
        text = content.decode("utf-8", errors="replace")
        text = text.replace('href="/', f'href="/web/{target}/')
        text = text.replace("href='/", f"href='/web/{target}/")
        text = text.replace('src="/', f'src="/web/{target}/')
        text = text.replace("src='/", f"src='/web/{target}/")
        text = text.replace('action="/', f'action="/web/{target}/')
        content = text.encode("utf-8")
        # Update content-length since rewriting may change size
        response_headers.pop("content-length", None)

    # Drop transfer-encoding since we have the full body
    response_headers.pop("transfer-encoding", None)

    return StarletteResponse(
        content=content,
        status_code=resp.status_code,
        headers=response_headers,
    )


@app.get("/v1/models")
async def list_models():
    """List available model aliases and backends."""
    models = []
    for alias in config.aliases:
        models.append({"id": alias, "object": "model", "owned_by": "proxy"})
    # Also list raw backend models
    for bname, bcfg in config.backends.items():
        if bcfg.type == "openai_compatible" and bcfg.base_url:
            models.append({"id": f"{bname}/*", "object": "model", "owned_by": bname})
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_requested = body.get("model", "best")
    stream = body.get("stream", False)

    # Extract or generate conversation ID for incremental context
    conversation_id = (
        body.pop("conversation_id", None)
        or request.headers.get("x-conversation-id")
        or f"auto-{uuid.uuid4().hex[:12]}"
    )

    backend, actual_model = config.resolve_model(model_requested)
    body["model"] = actual_model

    logger.info(
        "Routing '%s' -> backend=%s model=%s stream=%s conv=%s",
        model_requested, backend.name, actual_model, stream, conversation_id[:12],
    )

    if backend.type == "anthropic":
        return await _handle_anthropic(backend, body, stream)
    if backend.type == "cli":
        return await _handle_cli(backend, body, stream)
    if backend.type == "gemini_web":
        return await _handle_gemini_web(backend, body, stream, conversation_id)
    if backend.type == "copilot_web":
        return await _handle_copilot_web(backend, body, stream, conversation_id)
    if backend.type == "gemini":
        return await _handle_gemini(backend, body, stream)
    return await _handle_openai_compatible(backend, body, stream)


async def _handle_openai_compatible(
    backend: BackendConfig, body: dict, stream: bool
) -> JSONResponse | StreamingResponse:
    url = f"{backend.base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"

    if not stream:
        resp = await http_client.post(url, json=body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    # Streaming: proxy SSE chunks
    async def stream_response() -> AsyncIterator[bytes]:
        async with http_client.stream("POST", url, json=body, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n".encode()

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ── ANSI escape code stripper ──
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


async def _handle_cli(
    backend: BackendConfig, body: dict, stream: bool
) -> JSONResponse | StreamingResponse:
    """Route to a local CLI binary (claude or copilot) via subprocess.

    Supported binaries:
      - claude: Claude Code CLI (-p for print mode, --model for model selection)
      - copilot: GitHub Copilot CLI (-p for prompt mode, -s for silent, --model for model)

    Uses a temp file to bypass OS ARG_MAX limits when the prompt is large.
    """
    messages = body.get("messages", [])
    model = body.get("model", "")
    full_prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    # Write prompt to a secure temp file (ARG_MAX bypass)
    fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="llm_proxy_", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(full_prompt)

    # Build CLI command based on the backend's configured binary
    cli_binary = backend.cli_binary or "claude"
    extra_args = list(backend.cli_args or [])

    if cli_binary == "claude":
        # Claude Code: -p (print mode, non-interactive)
        # extra_args may include --model opus/sonnet/haiku
        command = [cli_binary, "-p"] + extra_args + [
            f"Read and respond to the full prompt in this file: {temp_path}",
        ]
    elif cli_binary == "copilot":
        # GitHub Copilot CLI: -p (prompt mode), -s (silent), --allow-all (non-interactive)
        # extra_args may include --model claude-opus-4.6, etc.
        # --allow-all is required for non-interactive mode (no permission prompts)
        extra_flags = extra_args if extra_args else []
        if "--allow-all" not in extra_flags:
            extra_flags = extra_flags + ["--allow-all"]
        command = [cli_binary, "-p",
            f"Read and respond to the full prompt in this file: {temp_path}",
        ] + extra_flags
    else:
        # Generic: pass temp file path as last argument
        command = cli_binary.split() + extra_args + [temp_path]

    logger.info("CLI exec: %s (prompt=%d chars -> %s)", command[0], len(full_prompt), temp_path)

    if not stream:
        # Non-streaming: collect full output
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            text = _ANSI_RE.sub("", stdout.decode("utf-8", errors="replace"))
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                logger.error("CLI error (rc=%d): %s", proc.returncode, err)
                text = f"CLI Error (exit {proc.returncode}):\n{err}\n\n{text}"
            return JSONResponse(content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Streaming mode
    async def stream_cli() -> AsyncIterator[bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text_chunk = _ANSI_RE.sub("", line.decode("utf-8", errors="replace"))
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

            await process.wait()

            # Final stop chunk
            final = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(stream_cli(), media_type="text/event-stream")


async def _handle_anthropic(
    backend: BackendConfig, body: dict, stream: bool
) -> JSONResponse | StreamingResponse:
    """Translate OpenAI format to Anthropic Messages API and back."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": backend.api_key or "",
        "anthropic-version": "2023-06-01",
    }

    # Convert OpenAI messages format to Anthropic format
    messages = body.get("messages", [])
    system_text = ""
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text += msg["content"] + "\n"
        else:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    anthropic_body = {
        "model": body["model"],
        "messages": anthropic_messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    if system_text.strip():
        anthropic_body["system"] = system_text.strip()

    if not stream:
        resp = await http_client.post(url, json=anthropic_body, headers=headers)
        data = resp.json()
        # Convert Anthropic response back to OpenAI format
        openai_response = _anthropic_to_openai(data, body["model"])
        return JSONResponse(content=openai_response, status_code=resp.status_code)

    # Streaming with Anthropic
    anthropic_body["stream"] = True

    async def stream_response() -> AsyncIterator[bytes]:
        collected_text = ""
        async with http_client.stream(
            "POST", url, json=anthropic_body, headers=headers
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                if event_type == "content_block_delta":
                    delta_text = event.get("delta", {}).get("text", "")
                    if delta_text:
                        collected_text += delta_text
                        chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body["model"],
                            "choices": [{
                                "index": 0,
                                "delta": {"content": delta_text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                elif event_type == "message_stop":
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": body["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                    yield b"data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


def _anthropic_to_openai(data: dict, model: str) -> dict:
    """Convert Anthropic Messages API response to OpenAI format."""
    content_blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    return {
        "id": f"chatcmpl-{data.get('id', uuid.uuid4().hex[:8])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": data.get("stop_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            "total_tokens": (
                data.get("usage", {}).get("input_tokens", 0)
                + data.get("usage", {}).get("output_tokens", 0)
            ),
        },
    }


async def _handle_gemini(
    backend: BackendConfig, body: dict, stream: bool
) -> JSONResponse | StreamingResponse:
    """Route to Gemini via its OpenAI-compatible endpoint.

    Google AI Studio provides an OpenAI-compatible API at:
    https://generativelanguage.googleapis.com/v1beta/openai/
    Free tier: 15 RPM on Gemini 2.0 Flash, 2 RPM on Gemini 2.5 Pro.
    """
    base = backend.base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
    url = f"{base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend.api_key}",
    }

    if not stream:
        resp = await http_client.post(url, json=body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    async def stream_response() -> AsyncIterator[bytes]:
        async with http_client.stream("POST", url, json=body, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n".encode()

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ── Model name mapping: proxy alias → gemini_webapi.constants.Model attribute ──
# Run `python -c "from gemini_webapi.constants import Model; print(list(Model))"` to refresh.
_GEMINI_WEB_MODEL_MAP: dict[str, str] = {
    "gemini-3.0-flash":           "G_3_0_FLASH",
    "gemini-3.1-pro":             "G_3_1_PRO",
    "gemini-3.0-flash-thinking":  "G_3_0_FLASH_THINKING",
}
_GEMINI_WEB_DEFAULT_MODEL = "G_3_0_FLASH"


async def _get_gemini_client(backend: BackendConfig):
    """Lazy-initialize the GeminiClient.

    Cookie resolution order (first success wins):
    1. Explicit cookies in backend config (most reliable — paste from Chrome DevTools)
    2. Saved session from auth manager (native browser login via /auth/login)
    3. Auto-extract from Chrome via browser_cookie3 (convenient, may prompt for Keychain)

    Thread-safe via asyncio.Lock. Resets on error so the next call retries fresh.
    """
    global gemini_client, _gemini_lock
    if gemini_client is not None:
        return gemini_client

    async with _gemini_lock:
        if gemini_client is not None:
            return gemini_client

        from gemini_webapi import GeminiClient as _GeminiClient

        # ── Priority 1: Explicit cookies from config.yaml ──────────────────
        if backend.cookies:
            logger.info(
                "Initializing Gemini Web client from config cookies (%d set)...",
                len(backend.cookies),
            )
            try:
                client = _GeminiClient(cookies=backend.cookies)
                await client.init(timeout=30, verbose=False)
                gemini_client = client
                logger.info("Gemini Web client initialized from config cookies ✓")
                return gemini_client
            except Exception as e:
                gemini_client = None
                logger.warning(
                    "Config cookies failed (%s) — trying saved auth session...", e
                )

        # ── Priority 2: Saved session from native browser login ────────────
        saved_cookies = auth_manager.get_cookies("gemini")
        if saved_cookies and saved_cookies.get("__Secure-1PSID"):
            logger.info("Trying Gemini Web client from saved auth session...")
            try:
                client = _GeminiClient(cookies=saved_cookies)
                await client.init(timeout=30, verbose=False)
                gemini_client = client
                logger.info("Gemini Web client initialized from saved auth session ✓")
                return gemini_client
            except Exception as e:
                gemini_client = None
                logger.warning(
                    "Saved auth session failed (%s) — trying browser auto-extract...", e
                )

        # ── Priority 3: Auto-extract from Chrome via browser_cookie3 ───────
        logger.info(
            "Initializing Gemini Web client via Chrome cookies (browser_cookie3)..."
        )
        try:
            import browser_cookie3

            _NEEDED = {"__Secure-1PSID", "__Secure-1PSIDTS"}
            _EXTRA = {"__Secure-3PSID", "SAPISID", "__Secure-3PAPISID"}
            cookies: dict[str, str] = {}
            cj = browser_cookie3.chrome(domain_name=".google.com")
            for cookie in cj:
                if cookie.name in (_NEEDED | _EXTRA) and cookie.value:
                    cookies[cookie.name] = cookie.value

            if not cookies.get("__Secure-1PSID"):
                raise ValueError(
                    "__Secure-1PSID not found.\n"
                    "Fix options:\n"
                    "  A) Use native login: POST http://localhost:8000/auth/login?target=gemini\n"
                    "  B) Open Chrome, sign in at gemini.google.com, restart proxy.\n"
                    "  C) Paste cookies into config.yaml under gemini_web.cookies"
                )

            client = _GeminiClient(
                secure_1psid=cookies["__Secure-1PSID"],
                secure_1psidts=cookies.get("__Secure-1PSIDTS"),
            )
            await client.init(timeout=30, verbose=False)
            gemini_client = client
            logger.info(
                "Gemini Web client initialized via Chrome cookies ✓ (%d extracted)",
                len(cookies),
            )
            return gemini_client
        except Exception as e:
            gemini_client = None
            logger.error("Gemini Web init failed: %s", e)
            raise


def _estimate_complexity(messages: list[dict]) -> str:
    """Estimate prompt complexity for routing and logging decisions.

    Returns 'low', 'medium', or 'high' based on message length and keywords.
    Used by tests and can inform future smart-routing logic.
    """
    _HIGH_KEYWORDS = {
        "refactor", "redesign", "architect", "implement entire", "rewrite",
        "entire", "comprehensive", "all files", "system", "integrate",
    }
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    if len(messages) >= 8 or total_chars >= 3000:
        return "high"
    combined = " ".join(str(m.get("content", "")).lower() for m in messages)
    if any(kw in combined for kw in _HIGH_KEYWORDS):
        return "high"
    if total_chars >= 500:
        return "medium"
    return "low"


def _get_or_create_gemini_session(conv_id: str, client: Any, model_enum: Any) -> tuple[Any, bool]:
    """Return an existing ChatSession or create a new one for *conv_id*.

    Returns (session, is_new).  is_new=True means this is the first turn for
    this conversation — the caller should send the full context prompt.

    Synchronous: ChatSession.start_chat() requires no await.  FIFO eviction
    when the cache grows past 50 sessions to bound memory use.
    """
    global _gemini_sessions
    if conv_id in _gemini_sessions:
        return _gemini_sessions[conv_id], False
    # FIFO eviction — Python 3.7+ dicts preserve insertion order
    if len(_gemini_sessions) >= 50:
        oldest = next(iter(_gemini_sessions))
        del _gemini_sessions[oldest]
        logger.debug("Evicted Gemini session: %s", oldest[:8])
    session = client.start_chat(model=model_enum)
    _gemini_sessions[conv_id] = session
    logger.debug("Created Gemini ChatSession for conv %s", conv_id[:8])
    return session, True


def _flatten_messages(messages: list[dict]) -> tuple[str, list[bytes]]:
    """Flatten OpenAI messages array into a (prompt, files) tuple.

    Handles multi-part content (list of {type, text/image_url} dicts) that
    Continue.dev produces when @file, @folder context or image attachments are
    injected. Images are extracted as raw bytes so callers can pass them to
    APIs that support native file uploads (e.g. gemini_webapi generate_content).
    """
    import base64
    parts = []
    files: list[bytes] = []
    for m in messages:
        role = m["role"].upper()
        content = m.get("content", "")
        if isinstance(content, list):
            text_pieces = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_pieces.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # data:<mime>;base64,<data>
                            try:
                                header, b64 = url.split(",", 1)
                                files.append(base64.b64decode(b64, validate=True))
                            except Exception:
                                text_pieces.append("[image attachment]")
                        elif url:
                            # Remote URL — pass as string path to gemini_webapi
                            files.append(url.encode())  # stored as bytes, decoded at call site
                else:
                    text_pieces.append(str(part))
            content = "\n".join(text_pieces)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts), files


async def _handle_gemini_web(
    backend: BackendConfig, body: dict, stream: bool,
    conversation_id: str = "",
) -> JSONResponse | StreamingResponse:
    """Chat via Gemini Web using a persistent ChatSession per conversation.

    Session strategy:
      - First turn: full context (system + files + user message), creates new ChatSession.
      - Subsequent turns: only the latest user message — ChatSession maintains history
        natively via Gemini's cid/rcid, so no context re-sending and no quota waste.
      - On error: only the affected session is killed; other conversations are unaffected.

    Streaming: uses send_message_stream() for real token-by-token output
    via ModelOutput.text_delta, replacing the previous word-batch simulation.
    """
    messages = body.get("messages", [])
    model_str = body.get("model", "gemini-3.0-flash")

    # Resolve model string → gemini_webapi.constants.Model enum
    from gemini_webapi.constants import Model as _Model
    enum_key = _GEMINI_WEB_MODEL_MAP.get(model_str, _GEMINI_WEB_DEFAULT_MODEL)
    model_enum = getattr(_Model, enum_key, _Model.UNSPECIFIED)

    # Ensure the shared GeminiClient is initialised before accessing sessions
    try:
        client = await _get_gemini_client(backend)
    except Exception as e:
        logger.error("Gemini client init failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": (
                    f"Gemini Web unavailable: {e}\n\n"
                    "To fix, choose one of:\n"
                    "1. Run 'Super Coder: Sign in to Web Service' from VS Code command palette.\n"
                    "2. POST http://localhost:8000/auth/login?target=gemini\n"
                    "3. Paste cookies into config.yaml under gemini_web.cookies"
                ),
                "type": "gemini_web_error",
                "code": "gemini_web_unavailable",
            }},
        )

    # Per-conversation ChatSession — the core fix for quota waste + looping
    session, is_new_session = _get_or_create_gemini_session(
        conversation_id, client, model_enum
    )

    if is_new_session:
        # First turn: send full context (smart-compressed in build_prompt)
        prompt, image_files = context_manager.build_prompt(
            conversation_id, messages, "gemini_web",
        )
        logger.info(
            "Gemini Web [%s]: NEW session, first turn, %d chars",
            conversation_id[:8], len(prompt),
        )
    else:
        # Subsequent turns: only the latest user message
        # ChatSession natively maintains conversation history with Gemini
        prompt, image_files = context_manager.extract_latest_user_message(messages)
        logger.info(
            "Gemini Web [%s]: existing session, %d chars (user msg only)",
            conversation_id[:8], len(prompt),
        )

    files_arg = [f.decode() if f.startswith(b"http") else f for f in image_files] or None
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    ts = int(time.time())
    logger.debug(
        "Gemini Web: model=%s enum=%s new_session=%s files=%d prompt_chars=%d",
        model_str, enum_key, is_new_session, len(image_files), len(prompt),
    )

    try:
        if not stream:
            response = await session.send_message(prompt, files=files_arg)
            text = response.text or ""
            return JSONResponse(content={
                "id": cid,
                "object": "chat.completion",
                "created": ts,
                "model": model_str,
                "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"}],
            })

        # Real streaming via send_message_stream — ModelOutput.text_delta gives
        # only the NEW characters per yield (no need to diff accumulated text).
        async def _stream() -> AsyncIterator[bytes]:
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n".encode()
            try:
                async for output in session.send_message_stream(prompt, files=files_arg):
                    delta = output.text_delta or ""
                    if delta:
                        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}]})}\n\n".encode()
            except Exception as stream_err:
                logger.error(
                    "Gemini Web stream error [%s]: %s", conversation_id[:8], stream_err
                )
                # Kill session so next request gets a fresh one with full context
                _gemini_sessions.pop(conversation_id, None)
                context_manager.reset_conversation(conversation_id)
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_stream(), media_type="text/event-stream")

    except Exception as e:
        # Only kill THIS conversation's session — the shared GeminiClient stays alive
        _gemini_sessions.pop(conversation_id, None)
        context_manager.reset_conversation(conversation_id)
        logger.error("Gemini Web error [%s]: %s", conversation_id[:8], e)
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": (
                    f"Gemini Web unavailable: {e}\n\n"
                    "To fix, choose one of:\n"
                    "1. Run 'Super Coder: Sign in to Web Service' from VS Code command palette.\n"
                    "2. POST http://localhost:8000/auth/login?target=gemini\n"
                    "3. Paste cookies into config.yaml under gemini_web.cookies"
                ),
                "type": "gemini_web_error",
                "code": "gemini_web_unavailable",
            }},
        )


# ── Copilot Web ────────────────────────────────────────────────────────────────

_COPILOT_WEB_STYLE_MAP: dict[str, str] = {
    "copilot-web-creative": "creative",
    "copilot-web-precise":  "precise",
    # default → "balanced"
}


def _get_copilot_u_cookie(backend: BackendConfig) -> str:
    """Resolve the Bing _U cookie for Copilot Web.

    Priority 1: explicit value in config.yaml cookies._U
    Priority 2: saved session from native browser login (/auth/login)
    Priority 3: auto-extract from Chrome via browser_cookie3
    """
    if backend.cookies and backend.cookies.get("_U"):
        return backend.cookies["_U"]

    # Check auth manager for saved session
    saved = auth_manager.get_cookies("copilot")
    if saved and saved.get("_U"):
        logger.info("Copilot Web: _U cookie from saved auth session ✓")
        return saved["_U"]

    try:
        import browser_cookie3
        cj = browser_cookie3.chrome(domain_name=".bing.com")
        for cookie in cj:
            if cookie.name == "_U" and cookie.value:
                logger.info("Copilot Web: _U cookie auto-extracted from Chrome ✓")
                return cookie.value
    except Exception as e:
        logger.debug("browser_cookie3 failed: %s", e)

    raise ValueError(
        "_U cookie not found.\n"
        "Fix options:\n"
        "  A) Use native login: POST http://localhost:8000/auth/login?target=copilot\n"
        "  B) Open Chrome, sign in at copilot.microsoft.com, restart proxy.\n"
        "  C) Paste cookie into config.yaml under copilot_web.cookies._U"
    )


async def _handle_copilot_web(
    backend: BackendConfig, body: dict, stream: bool,
    conversation_id: str = "",
) -> JSONResponse | StreamingResponse:
    """Chat via Microsoft Copilot Web (browser session — no API key needed).

    Uses sydney-py to drive the Copilot WebSocket endpoint with a _U cookie.
    Uses smart incremental context — only sends full context on first turn.
    """
    from sydney import SydneyClient as _SydneyClient

    messages = body.get("messages", [])
    model_str = body.get("model", "copilot-web")

    # Use smart context manager for incremental context
    if conversation_id:
        full_prompt, image_files = context_manager.build_prompt(
            conversation_id, messages, "copilot_web",
        )
    else:
        full_prompt, image_files = _flatten_messages(messages)

    style = _COPILOT_WEB_STYLE_MAP.get(model_str, "balanced")

    try:
        u_cookie = _get_copilot_u_cookie(backend)
    except ValueError as e:
        logger.error("Copilot Web cookie error: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": str(e),
                "type": "copilot_web_error",
                "code": "copilot_web_unavailable",
            }},
        )

    # Save first image to temp file — sydney-py takes a file path
    attachment_path: str | None = None
    if image_files:
        fd, attachment_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(fd, "wb") as f:
            f.write(image_files[0])

    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    ts = int(time.time())
    logger.debug("Copilot Web: style=%s attachments=%d", style, len(image_files))

    if not stream:
        try:
            async with _SydneyClient(style=style, bing_cookies=u_cookie) as sydney:
                text = await sydney.ask(full_prompt, attachment=attachment_path)
                if isinstance(text, tuple):
                    text = text[0]
        except Exception as e:
            logger.error("Copilot Web error: %s", e)
            return JSONResponse(
                status_code=503,
                content={"error": {
                    "message": f"Copilot Web unavailable: {e}",
                    "type": "copilot_web_error",
                    "code": "copilot_web_unavailable",
                }},
            )
        finally:
            if attachment_path:
                try:
                    os.unlink(attachment_path)
                except Exception:
                    pass
        return JSONResponse(content={
            "id": cid,
            "object": "chat.completion",
            "created": ts,
            "model": model_str,
            "choices": [{"index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"}],
        })

    # Streaming — sydney-py yields real token chunks over WebSocket
    async def _stream() -> AsyncIterator[bytes]:
        try:
            async with _SydneyClient(style=style, bing_cookies=u_cookie) as sydney:
                yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n".encode()
                async for chunk in sydney.ask_stream(full_prompt, attachment=attachment_path):
                    if isinstance(chunk, tuple):
                        chunk = chunk[0]
                    if chunk:
                        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n".encode()
                yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
                yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error("Copilot Web stream error: %s", e)
        finally:
            if attachment_path:
                try:
                    os.unlink(attachment_path)
                except Exception:
                    pass

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Context Bridge — Browser Backend ──────────────────────────────────────────

import platform as _platform
import shutil as _shutil
import subprocess as _subprocess

# Chrome Remote Debugging Protocol — how we attach to a real Chrome instance.
# Using CDP means zero automation flags: no "controlled by test software" banner,
# real sessions, and existing logins all work.
_CDP_PORT = 9222

# Dedicated Chrome profile for CDP automation.  Separate from the user's main
# Chrome profile so both can run side-by-side.  Sessions persist here, so the
# user only needs to sign in once per service.
_CDP_PROFILE_DIR = Path.home() / ".config" / "context-bridge" / "chrome-cdp"


def _find_chrome_exe() -> str | None:
    """Locate the Chrome/Chromium binary on this OS."""
    by_os: dict[str, list[str]] = {
        "Darwin": [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ],
        "Linux": [
            "/usr/bin/google-chrome-stable",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
        ],
        "Windows": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
    }
    for path in by_os.get(_platform.system(), []):
        if Path(path).exists():
            return path
    for name in ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser"):
        if found := _shutil.which(name):
            return found
    return None


async def _cdp_is_alive(port: int = _CDP_PORT) -> bool:
    """Return True if a Chrome CDP endpoint is already listening on *port*."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port), timeout=1.0
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def _wait_for_cdp(port: int = _CDP_PORT, timeout: float = 15.0) -> bool:
    """Poll until Chrome's CDP endpoint is ready (up to *timeout* seconds)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await _cdp_is_alive(port):
            return True
        await asyncio.sleep(0.4)
    return False


async def _ensure_cdp_browser() -> None:
    """Ensure a Chrome instance is running with CDP and store the connection.

    Three tiers:
      1. CDP port already open → attach directly.  This handles the power-user
         case where Chrome was launched with --remote-debugging-port=9222 (e.g.
         the user's main browser) — we get real sessions for free.
      2. CDP port not open → launch a *dedicated* Chrome profile from
         _CDP_PROFILE_DIR with --remote-debugging-port, then attach.
         The dedicated profile coexists beside the user's main Chrome window
         and its sessions persist across proxy restarts (one-time sign-in).
      3. Chrome binary not found → raise a clear error.
    """
    global _pw_browser, _pw_context

    # Re-use if still connected
    if _pw_browser is not None:
        try:
            _ = _pw_browser.contexts   # raises if CDP connection dropped
            return
        except Exception:
            logger.warning("CDP connection lost — reconnecting")
            _pw_browser = None
            _pw_context = None

    first_run = not (_CDP_PROFILE_DIR / "Default").exists()

    if await _cdp_is_alive():
        logger.info("Attaching to existing Chrome CDP on port %d", _CDP_PORT)
    else:
        chrome = _find_chrome_exe()
        if not chrome:
            raise RuntimeError(
                "Google Chrome not found. Install Chrome or start it manually with "
                f"--remote-debugging-port={_CDP_PORT} and reload."
            )

        _CDP_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [
            chrome,
            f"--remote-debugging-port={_CDP_PORT}",
            f"--user-data-dir={_CDP_PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-infobars",  # suppress the automation infobar
        ]
        logger.info("Launching Chrome CDP profile: %s", _CDP_PROFILE_DIR)
        _subprocess.Popen(cmd, stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL)

        if not await _wait_for_cdp(timeout=15):
            raise RuntimeError(
                f"Chrome did not open CDP on port {_CDP_PORT} within 15 s. "
                "Check that nothing else is using that port."
            )

    _pw_browser = await _pw_instance.chromium.connect_over_cdp(
        f"http://127.0.0.1:{_CDP_PORT}"
    )
    contexts = _pw_browser.contexts
    _pw_context = contexts[0] if contexts else await _pw_browser.new_context()
    logger.info(
        "Connected to Chrome via CDP (%d existing context(s), first_run=%s)",
        len(contexts), first_run,
    )
    # Stash first_run on the module so /browser/launch can report it
    _pw_browser._vscode_first_run = first_run

_CHAT_TARGETS: dict[str, str] = {
    "gemini":  "https://gemini.google.com",
    "copilot": "https://copilot.microsoft.com",
    "chatgpt": "https://chatgpt.com",
    "claude":  "https://claude.ai",
}

# CSS injected after launch — hides chrome (nav/sidebar) and spotlights the chat
_FOCUS_CSS: dict[str, str] = {
    "gemini": """
        bard-sidenav, .sidenav, mat-drawer, .app-bar-container,
        c-wiz[data-node-index="0;0"] > div > div:first-child,
        [jsname="paAFfc"], .new-chat-btn-container { display:none!important; }
        .chat-history, [class*="conversation-container"] { max-width:100%!important; }
    """,
    "copilot": """
        cib-serp-main::part(header), header, #b_header,
        .cib-action-bar-main-container, .cib-suggestion-bar { display:none!important; }
        cib-conversation { height:100vh!important; }
    """,
    "chatgpt": """
        nav, .sidebar, #__next > div > div:first-child,
        [class*="Sidebar"] { display:none!important; }
        main { max-width:100%!important; margin:0!important; }
    """,
    "claude": """
        [data-testid="sidebar"], nav, .transition-all.duration-200 { display:none!important; }
        main { max-width:100%!important; }
    """,
}

# JS that grabs the last assistant message text
_EXTRACT_JS: dict[str, str] = {
    "gemini":  "const m=document.querySelectorAll('model-response .markdown,.model-response-text');return m.length?m[m.length-1].innerText:'';",
    "copilot": "const m=document.querySelectorAll('cib-message[source=\"bot\"] .content,.response-message-content');return m.length?m[m.length-1].innerText:'';",
    "chatgpt": "const m=document.querySelectorAll('[data-message-author-role=\"assistant\"] .markdown');return m.length?m[m.length-1].innerText:'';",
    "claude":  "const m=document.querySelectorAll('[data-is-streaming=\"false\"] .font-claude-message,.assistant-message');return m.length?m[m.length-1].innerText:'';",
}

# CSS selectors tried in order to locate the chat input field
_INPUT_SELECTORS: dict[str, list[str]] = {
    "gemini":  [
        "rich-textarea div[contenteditable='true']",
        "div[contenteditable='true'][data-placeholder]",
        "textarea[placeholder*='Enter a prompt']",
        "div[contenteditable='true']",
    ],
    "copilot": [
        "cib-text-input textarea",
        "#userInput",
        "textarea[placeholder*='Ask']",
        "textarea",
    ],
    "chatgpt": [
        "#prompt-textarea",
        "div[contenteditable='true'][data-placeholder]",
        "textarea[data-id='root']",
    ],
    "claude":  [
        "div[contenteditable='true'][data-placeholder*='message']",
        ".ProseMirror[contenteditable='true']",
        "div[contenteditable='true']",
    ],
}

# CSS selectors tried in order to locate the submit button
_SUBMIT_SELECTORS: dict[str, list[str]] = {
    "gemini":  [
        "button[aria-label='Send message']",
        "button.send-button",
        "[data-mat-icon-name='send']",
    ],
    "copilot": [
        "button[aria-label='Submit']",
        "button[type='submit']",
        "cib-text-input button",
    ],
    "chatgpt": [
        "button[data-testid='send-button']",
        "button[aria-label='Send prompt']",
    ],
    "claude":  [
        "button[aria-label='Send Message']",
        "button[data-testid='send-button']",
    ],
}

_pw_instance = None
_pw_browser  = None   # CDP-connected Browser (real Chrome, no automation flags)
_pw_context  = None   # Default BrowserContext from CDP connection
_pw_pages: dict[str, object] = {}  # target -> Page
_pw_lock = asyncio.Lock()


async def _get_browser_page(target: str):
    """Return (or lazily open) the Playwright page for *target*.

    Uses Chrome Remote Debugging Protocol — no automation flags injected, no
    'controlled by automated test software' banner, and all cookies/logins
    stored in the CDP profile are available from the very first open.
    """
    global _pw_instance, _pw_pages

    async with _pw_lock:
        # Re-validate existing page (user may have closed it from Chrome)
        if target in _pw_pages:
            try:
                _ = _pw_pages[target].url
                return _pw_pages[target]
            except Exception:
                del _pw_pages[target]

        from playwright.async_api import async_playwright

        if _pw_instance is None:
            _pw_instance = await async_playwright().start()

        await _ensure_cdp_browser()   # connect / launch Chrome via CDP

        page = await _pw_context.new_page()
        url = _CHAT_TARGETS.get(target, _CHAT_TARGETS["gemini"])
        await page.goto(url, wait_until="domcontentloaded")

        tab_label = f"VS Code · {target}"
        try:
            await page.evaluate(f"document.title = {repr(tab_label)}")
        except Exception:
            pass

        css = _FOCUS_CSS.get(target, "")
        if css:
            await page.add_style_tag(content=css)
            page.on("framenavigated", lambda f: asyncio.ensure_future(
                page.add_style_tag(content=css) if f == page.main_frame else asyncio.sleep(0)
            ))

        async def _relabel(frame):
            if frame == page.main_frame:
                try:
                    await page.evaluate(f"document.title = {repr(tab_label)}")
                except Exception:
                    pass
        page.on("framenavigated", lambda f: asyncio.ensure_future(_relabel(f)))

        logger.info("Browser page opened for '%s' → %s", target, url)
        _pw_pages[target] = page
        return page


@app.get("/browser/status")
async def browser_status(target: str = "gemini"):
    """Return browser connection state so the panel can show live status."""
    cdp_ready = await _cdp_is_alive()
    first_run = cdp_ready and not (_CDP_PROFILE_DIR / "Default").exists()
    return {
        "cdp_ready": cdp_ready,
        "browser_connected": _pw_browser is not None,
        "page_open": target in _pw_pages,
        "cdp_port": _CDP_PORT,
        # True when the CDP profile has never been used → user needs to sign in
        "first_run": not (_CDP_PROFILE_DIR / "Default").exists(),
    }


@app.post("/browser/launch")
async def browser_launch(target: str = "gemini"):
    """Open (or focus) a chat browser tab for *target* via Chrome CDP."""
    try:
        first_run_before = not (_CDP_PROFILE_DIR / "Default").exists()
        page = await _get_browser_page(target)
        await page.bring_to_front()
        return {
            "ok": True,
            "url": page.url,
            # Let the panel know if this is the first launch (needs sign-in)
            "first_run": first_run_before,
        }
    except Exception as e:
        logger.error("Browser launch error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/browser/extract")
async def browser_extract(target: str = "gemini"):
    """Extract the last assistant message text from the active chat page."""
    if target not in _pw_pages:
        raise HTTPException(status_code=404, detail=f"No browser page open for '{target}'. Launch it first.")
    try:
        page = _pw_pages[target]
        js = _EXTRACT_JS.get(target, "return document.body.innerText;")
        text = await page.evaluate(f"() => {{ {js} }}")
        return {"text": text or ""}
    except Exception as e:
        logger.error("Browser extract error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class InjectRequest(BaseModel):
    text: str
    submit: bool = True


@app.post("/browser/inject")
async def browser_inject(target: str = "gemini", body: InjectRequest = None):
    """Type text into the chat input of the active browser page and optionally submit.

    The endpoint tries a series of CSS selectors for both the input field and
    submit button.  Works with both <textarea> and contenteditable elements.
    """
    if target not in _pw_pages:
        raise HTTPException(
            status_code=404,
            detail=f"No browser page open for '{target}'. Call /browser/launch first.",
        )
    if body is None or not body.text:
        raise HTTPException(status_code=422, detail="'text' field is required.")

    page = _pw_pages[target]
    text = body.text

    # --- locate the input field ---
    input_el = None
    for sel in _INPUT_SELECTORS.get(target, ["textarea", "div[contenteditable='true']"]):
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                input_el = loc
                break
        except Exception:
            continue

    if input_el is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not find chat input for '{target}'. The page may not be ready.",
        )

    try:
        await input_el.click()
        # Clear existing content first (Ctrl+A → Delete)
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Delete")
        # Type the text — use keyboard.type for contenteditable richtext support
        await page.keyboard.type(text, delay=5)
        logger.info("Injected %d chars into '%s' input", len(text), target)
    except Exception as e:
        logger.error("Browser inject (type) error: %s", e)
        raise HTTPException(status_code=500, detail=f"Typing failed: {e}")

    if not body.submit:
        return {"ok": True, "submitted": False}

    # --- locate and click the submit button ---
    submit_el = None
    for sel in _SUBMIT_SELECTORS.get(target, ["button[type='submit']"]):
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                submit_el = loc
                break
        except Exception:
            continue

    if submit_el:
        try:
            await submit_el.click()
            logger.info("Clicked submit button for '%s'", target)
            return {"ok": True, "submitted": True}
        except Exception as e:
            logger.warning("Submit click failed for '%s': %s — trying Enter", target, e)

    # Fallback: press Enter
    try:
        await page.keyboard.press("Enter")
        logger.info("Submitted via Enter for '%s'", target)
        return {"ok": True, "submitted": True, "method": "enter"}
    except Exception as e:
        logger.error("Browser inject (submit) error: %s", e)
        raise HTTPException(status_code=500, detail=f"Submit failed: {e}")


@app.post("/browser/close")
async def browser_close(target: str | None = None):
    """Close one browser page (or all if target is omitted).

    Does NOT kill the Chrome process — it keeps running so the user can still
    browse normally.  Just disconnects Playwright's handle.
    """
    global _pw_browser, _pw_context, _pw_pages
    if target:
        page = _pw_pages.pop(target, None)
        if page:
            try:
                await page.close()
            except Exception:
                pass
    else:
        for page in list(_pw_pages.values()):
            try:
                await page.close()
            except Exception:
                pass
        _pw_pages.clear()
        # Disconnect from CDP — Chrome keeps running normally
        _pw_browser = None
        _pw_context = None
    return {"ok": True}


# ── Context Bridge Panel HTML ──────────────────────────────────────────────────

_CONTEXT_PANEL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Context Bridge</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1e1e1e; color: #d4d4d4;
    display: flex; flex-direction: column; height: 100vh; padding: 12px; gap: 10px;
  }
  .toolbar {
    display: flex; align-items: center; gap: 8px; flex-shrink: 0; flex-wrap: wrap;
  }
  .toolbar select, .toolbar button, .toolbar input {
    background: #2d2d2d; color: #d4d4d4;
    border: 1px solid #444; border-radius: 4px;
    padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  .toolbar button {
    background: #0e639c; border-color: #0e639c; color: #fff;
    font-weight: 600; padding: 5px 14px;
  }
  .toolbar button:hover:not(:disabled) { background: #1177bb; }
  .toolbar button:disabled { background: #444; border-color: #444; cursor: default; opacity: 0.6; }
  .toolbar button.secondary { background: #2d2d2d; border-color: #555; color: #ccc; font-weight: 400; }
  .toolbar button.secondary:hover:not(:disabled) { border-color: #888; color: #fff; }
  .toolbar label { font-size: 12px; color: #888; white-space: nowrap; }
  .divider { width: 1px; height: 20px; background: #444; margin: 0 4px; }
  .panels { display: flex; gap: 10px; flex: 1; min-height: 0; }
  .pane { display: flex; flex-direction: column; flex: 1; gap: 6px; min-width: 0; }
  .pane-header { display: flex; align-items: center; gap: 6px; }
  .pane-label {
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px; flex: 1;
  }
  .pane-btn {
    font-size: 11px; background: #2d2d2d; border: 1px solid #444;
    color: #888; border-radius: 4px; padding: 2px 8px; cursor: pointer;
  }
  .pane-btn:hover { color: #d4d4d4; border-color: #888; }
  textarea {
    flex: 1; background: #2d2d2d; color: #d4d4d4;
    border: 1px solid #444; border-radius: 4px;
    padding: 10px; font-size: 13px; font-family: inherit;
    resize: none; line-height: 1.5;
  }
  textarea:focus { outline: none; border-color: #0e639c; }
  .status { font-size: 12px; color: #888; margin-left: auto; }
  .status.error { color: #f14c4c; }
  .status.ok { color: #89d185; }
  .status.busy { color: #e8bf6a; }
  .browser-bar {
    display: flex; align-items: center; gap: 8px;
    background: #252526; border: 1px solid #333; border-radius: 4px;
    padding: 6px 10px; flex-shrink: 0;
  }
  .browser-bar label { font-size: 12px; color: #888; }
  .browser-chips { display: flex; gap: 4px; }
  .chip {
    font-size: 12px; padding: 3px 10px; border-radius: 12px; cursor: pointer;
    border: 1px solid #444; background: #2d2d2d; color: #aaa;
    transition: all .15s;
  }
  .chip:hover { border-color: #888; color: #fff; }
  .chip.active { border-color: #0e639c; background: #0e639c22; color: #4fc3f7; }
  .auth-bar {
    display: flex; align-items: center; gap: 8px;
    background: #252526; border: 1px solid #333; border-radius: 4px;
    padding: 6px 10px; flex-shrink: 0;
  }
  .auth-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .auth-dot.ok { background: #89d185; }
  .auth-dot.no { background: #f14c4c; }
  .auth-dot.stale { background: #e8bf6a; }
  .auth-item {
    font-size: 12px; display: flex; align-items: center; gap: 5px; cursor: pointer;
    padding: 2px 6px; border-radius: 4px;
  }
  .auth-item:hover { background: #333; }
</style>
</head>
<body>

<!-- ── Browser launcher bar ─────────────────────────────────── -->
<div class="browser-bar">
  <label>Open in focus mode</label>
  <div class="browser-chips">
    <span class="chip" onclick="launchBrowser('gemini')">Gemini</span>
    <span class="chip" onclick="launchBrowser('copilot')">Copilot</span>
    <span class="chip" onclick="launchBrowser('chatgpt')">ChatGPT</span>
    <span class="chip" onclick="launchBrowser('claude')">Claude.ai</span>
  </div>
  <div class="divider"></div>
  <button class="secondary" style="font-size:12px;padding:3px 10px;" onclick="grabFromBrowser()">Grab last response</button>
  <select id="grab-target" style="font-size:12px;padding:3px 8px;">
    <option value="gemini">Gemini</option>
    <option value="copilot">Copilot</option>
    <option value="chatgpt">ChatGPT</option>
    <option value="claude">Claude.ai</option>
  </select>
</div>

<!-- ── Auth status bar ───────────────────────────────────────── -->
<div class="auth-bar" id="auth-bar">
  <span style="font-size:12px;color:#888;">Auth:</span>
  <span class="auth-item" onclick="toggleAuth('gemini')"><span class="auth-dot no" id="auth-gemini"></span>Gemini</span>
  <span class="auth-item" onclick="toggleAuth('copilot')"><span class="auth-dot no" id="auth-copilot"></span>Copilot</span>
  <span class="auth-item" onclick="toggleAuth('chatgpt')"><span class="auth-dot no" id="auth-chatgpt"></span>ChatGPT</span>
  <span class="auth-item" onclick="toggleAuth('claude')"><span class="auth-dot no" id="auth-claude"></span>Claude</span>
</div>

<!-- ── Summarizer toolbar ────────────────────────────────────── -->
<div class="toolbar">
  <label>Model</label>
  <select id="model">
    <option value="gemini-flash-web">Gemini Flash Web</option>
    <option value="gemini-pro-web">Gemini Pro Web</option>
    <option value="copilot-web">Copilot Web</option>
    <option value="claude-smart">Claude Pro</option>
    <option value="copilot-opus">Copilot Opus</option>
    <option value="copilot-gpt">Copilot GPT</option>
  </select>
  <div class="divider"></div>
  <label>Mode</label>
  <select id="preset">
    <option value="summarize">Summarize as coding context</option>
    <option value="extract">Extract decisions &amp; constraints</option>
    <option value="bullets">Bullet-point key facts</option>
    <option value="custom">Custom...</option>
  </select>
  <input id="custom-instruction" placeholder="Enter instruction..."
    style="display:none; flex:1;" />
  <button id="run-btn" onclick="run()">Summarize ⌘↵</button>
  <span id="status" class="status"></span>
</div>

<!-- ── Two text panes ────────────────────────────────────────── -->
<div class="panels">
  <div class="pane">
    <div class="pane-header">
      <div class="pane-label">Input — paste or grab from browser</div>
      <button class="pane-btn" onclick="clearInput()">Clear</button>
    </div>
    <textarea id="input" placeholder="Paste any text here, or use 'Grab last response' above to pull from an open browser tab..."></textarea>
  </div>
  <div class="pane">
    <div class="pane-header">
      <div class="pane-label">Context output — paste into Continue</div>
      <button class="pane-btn" onclick="copyOutput()">Copy</button>
    </div>
    <textarea id="output" placeholder="Summary appears here..." readonly></textarea>
  </div>
</div>

<script>
const PRESETS = {
  summarize: "Summarize the following into a concise coding context block. Focus on technical decisions, architecture, APIs, constraints, and anything a developer needs to continue the work. Be terse — no preamble.",
  extract:   "Extract key technical decisions, constraints, and open questions. Short bulleted list. No preamble.",
  bullets:   "Condense into the most important facts as brief bullet points. Prioritize technical details.",
};

let activeBrowserTarget = null;

document.getElementById("preset").addEventListener("change", () => {
  const v = document.getElementById("preset").value;
  document.getElementById("custom-instruction").style.display = v === "custom" ? "inline-block" : "none";
});

function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status" + (cls ? " " + cls : "");
}

function getInstruction() {
  const p = document.getElementById("preset").value;
  return p === "custom" ? document.getElementById("custom-instruction").value.trim() : PRESETS[p];
}

async function launchBrowser(target) {
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
  document.querySelector(`.chip[onclick*="${target}"]`).classList.add("active");
  document.getElementById("grab-target").value = target;
  activeBrowserTarget = target;
  setStatus("Opening " + target + "...", "busy");
  try {
    const r = await fetch("/browser/launch?target=" + target, { method: "POST" });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    setStatus(target + " open ✓", "ok");
  } catch(e) { setStatus(e.message, "error"); }
}

async function grabFromBrowser() {
  const target = document.getElementById("grab-target").value;
  setStatus("Grabbing from " + target + "...", "busy");
  try {
    const r = await fetch("/browser/extract?target=" + target);
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    if (!d.text) { setStatus("No response found yet.", "error"); return; }
    document.getElementById("input").value = d.text;
    setStatus("Grabbed ✓", "ok");
  } catch(e) { setStatus(e.message, "error"); }
}

function clearInput() { document.getElementById("input").value = ""; }

async function run() {
  const input = document.getElementById("input").value.trim();
  if (!input) { setStatus("Nothing to summarize.", "error"); return; }
  const model = document.getElementById("model").value;
  const instruction = getInstruction();
  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  setStatus("Running...", "busy");
  document.getElementById("output").value = "";
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model, stream: true,
        messages: [
          { role: "system", content: instruction },
          { role: "user",   content: input }
        ]
      })
    });
    if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e?.error?.message || res.statusText); }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let out = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split("\\n")) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        try {
          const delta = JSON.parse(line.slice(6))?.choices?.[0]?.delta?.content;
          if (delta) { out += delta; document.getElementById("output").value = out; }
        } catch {}
      }
    }
    setStatus("Done ✓", "ok");
  } catch(e) {
    setStatus(e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

function copyOutput() {
  const v = document.getElementById("output").value;
  if (!v) return;
  navigator.clipboard.writeText(v).then(() => setStatus("Copied ✓", "ok"));
}

document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
});

// ── Auth status ─────────────────────────────────────────────
async function loadAuthStatus() {
  try {
    const r = await fetch("/auth/status");
    const d = await r.json();
    const targets = d.targets || [];
    for (const info of targets) {
      const dot = document.getElementById("auth-" + info.target);
      if (!dot) continue;
      dot.className = "auth-dot " + (info.authenticated ? "ok" : "no");
      dot.title = info.authenticated ? "Signed in" : "Not signed in";
    }
  } catch {}
}

async function toggleAuth(target) {
  const dot = document.getElementById("auth-" + target);
  if (!dot) return;
  const wasOk = dot.classList.contains("ok") || dot.classList.contains("stale");
  if (wasOk) {
    // Logout
    dot.className = "auth-dot no";
    setStatus("Signing out of " + target + "...", "busy");
    try {
      await fetch("/auth/logout?target=" + target, { method: "POST" });
      setStatus(target + " signed out", "ok");
    } catch(e) { setStatus(e.message, "error"); }
  } else {
    // Login
    dot.className = "auth-dot stale";
    setStatus("Opening " + target + " login... (complete in browser)", "busy");
    try {
      const r = await fetch("/auth/login?target=" + target, { method: "POST" });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || r.statusText);
      dot.className = "auth-dot ok";
      setStatus(target + " signed in ✓", "ok");
    } catch(e) {
      dot.className = "auth-dot no";
      setStatus(e.message, "error");
    }
  }
}

loadAuthStatus();
</script>
</body>
</html>"""


@app.get("/context", response_class=None)
async def context_panel():
    """Serve the Context Bridge panel."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_CONTEXT_PANEL_HTML)


@app.get("/v1/context/file-tree")
async def file_tree(
    root: str = ".",
    depth: int = 3,
    include_readme: bool = True,
):
    """Return a structured file tree for injecting into Gemini Web chat context.

    Use in Continue.dev by typing the tree output as context before your prompt.
    Query params:
      root    — directory to scan (default: cwd)
      depth   — max directory depth (default: 3)
      include_readme — prepend README.md content if found (default: True)
    """
    import pathlib

    root_path = pathlib.Path(root).resolve()
    if not root_path.exists():
        return JSONResponse(status_code=404, content={"error": f"{root} not found"})

    _IGNORE = {".git", "__pycache__", "node_modules", ".venv", "venv",
               "dist", "build", ".next", ".cache", "*.egg-info"}

    def _tree(path: pathlib.Path, prefix: str = "", current_depth: int = 0) -> list[str]:
        if current_depth > depth:
            return []
        lines = []
        try:
            items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return []
        for i, item in enumerate(items):
            if any(item.match(pat) for pat in _IGNORE) or item.name.startswith("."):
                continue
            connector = "└── " if i == len(items) - 1 else "├── "
            lines.append(f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                extension = "    " if i == len(items) - 1 else "│   "
                lines.extend(_tree(item, prefix + extension, current_depth + 1))
        return lines

    tree_lines = [f"{root_path.name}/"] + _tree(root_path)
    tree_text = "\n".join(tree_lines)

    readme_text = ""
    if include_readme:
        for name in ("README.md", "readme.md", "README.txt", "AGENTS.md"):
            rp = root_path / name
            if rp.exists():
                try:
                    readme_text = rp.read_text(encoding="utf-8", errors="replace")[:4000]
                    readme_text = f"\n\n--- {name} ---\n{readme_text}"
                    break
                except OSError:
                    pass

    return {
        "tree": tree_text,
        "readme": readme_text,
        "prompt_block": (
            f"<project_context>\n"
            f"File tree (depth={depth}):\n```\n{tree_text}\n```"
            f"{readme_text}\n</project_context>"
        ),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "backends": list(config.backends.keys())}


# ── Authentication Endpoints ──────────────────────────────────────────────────

@app.get("/auth/status")
async def auth_status(target: str | None = None):
    """Check auth status for one or all web backends."""
    if target:
        return auth_manager.status(target)
    return {"targets": auth_manager.status_all()}


@app.post("/auth/login")
async def auth_login(target: str = "gemini"):
    """Open a browser window for interactive login. Captures cookies automatically.

    Supported targets: gemini, copilot, chatgpt, claude
    """
    try:
        cookies = await auth_manager.login(target)

        # Inject the new cookies into the running backend config
        _inject_auth_cookies()
        # Reset any cached clients so they reinitialize with new cookies
        global gemini_client
        if target == "gemini":
            gemini_client = None

        return {
            "ok": True,
            "target": target,
            "cookies_captured": len(cookies),
            "message": f"Successfully logged in to {target}. Cookies saved.",
        }
    except TimeoutError as e:
        return JSONResponse(status_code=408, content={"error": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error("Auth login error for %s: %s", target, e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/auth/logout")
async def auth_logout(target: str):
    """Clear saved session for a target."""
    cleared = await auth_manager.logout(target)
    return {"ok": True, "cleared": cleared}


@app.post("/auth/refresh")
async def auth_refresh(target: str = "gemini"):
    """Force re-login (clears old session, opens browser)."""
    try:
        cookies = await auth_manager.refresh(target)
        _inject_auth_cookies()
        global gemini_client
        if target == "gemini":
            gemini_client = None
        return {
            "ok": True,
            "target": target,
            "cookies_captured": len(cookies),
        }
    except TimeoutError as e:
        return JSONResponse(status_code=408, content={"error": str(e)})
    except Exception as e:
        logger.error("Auth refresh error for %s: %s", target, e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Dynamic Model Discovery ──────────────────────────────────────────────────

@app.get("/v1/backends")
async def list_backends():
    """List all backends with their auth status and available models."""
    result = []
    for name, bcfg in config.backends.items():
        entry = {
            "name": name,
            "type": bcfg.type,
            "available": True,
        }

        # Check auth status for web backends
        if bcfg.type == "gemini_web":
            status = auth_manager.status("gemini")
            entry["authenticated"] = status.get("authenticated", False)
            entry["auth_target"] = "gemini"
        elif bcfg.type == "copilot_web":
            status = auth_manager.status("copilot")
            entry["authenticated"] = status.get("authenticated", False)
            entry["auth_target"] = "copilot"
        elif bcfg.type == "cli":
            entry["authenticated"] = True  # CLI auth is separate
        elif bcfg.type in ("openai_compatible", "gemini"):
            entry["authenticated"] = bool(bcfg.api_key)

        # List aliases that point to this backend
        entry["models"] = [
            {"alias": alias, "model": acfg.model}
            for alias, acfg in config.aliases.items()
            if acfg.backend == name
        ]

        result.append(entry)

    return {"backends": result}


def main():
    import uvicorn
    cfg = load_config()
    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))
    uvicorn.run(
        "proxy.server:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
    )


if __name__ == "__main__":
    main()
