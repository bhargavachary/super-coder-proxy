"""Super Coder — Unified LLM Proxy.

Exposes /v1/chat/completions and routes to:
  1. Local Ollama models
  2. Claude CLI
  3. GitHub Copilot CLI
  4. Gemini Web (browser cookies)
  5. Gemini API (Google AI Studio)
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
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import BackendConfig, ProxyConfig, load_config

logger = logging.getLogger("super-coder")

config: ProxyConfig
http_client: httpx.AsyncClient
gemini_client = None  # Lazy-initialized GeminiClient
_gemini_lock: asyncio.Lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, http_client, _gemini_lock
    config = load_config()
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    _gemini_lock = asyncio.Lock()
    logger.info(
        "Proxy router started on %s:%s with %d backends, %d aliases",
        config.host, config.port, len(config.backends), len(config.aliases),
    )
    for name, b in config.backends.items():
        key_status = "set" if b.api_key else "missing"
        logger.info("  backend: %-12s type=%-18s key=%s", name, b.type, key_status)
    yield
    await http_client.aclose()


app = FastAPI(title="Super Coder", lifespan=lifespan)


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

    backend, actual_model = config.resolve_model(model_requested)
    body["model"] = actual_model

    logger.info(
        "Routing '%s' -> backend=%s model=%s stream=%s",
        model_requested, backend.name, actual_model, stream,
    )

    if backend.type == "anthropic":
        return await _handle_anthropic(backend, body, stream)
    if backend.type == "cli":
        return await _handle_cli(backend, body, stream)
    if backend.type == "gemini_web":
        return await _handle_gemini_web(backend, body, stream)
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
    2. Auto-extract from Chrome via browser_cookie3 (convenient, may prompt for Keychain)

    Thread-safe via asyncio.Lock. Resets on error so the next call retries fresh.

    To set explicit cookies, add to config.yaml:
        backends:
          gemini_web:
            type: gemini_web
            cookies:
              __Secure-1PSID: "<value>"
              __Secure-1PSIDTS: "<value>"
        (Chrome DevTools → Application → Cookies → https://gemini.google.com)
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
                    "Config cookies failed (%s) — trying browser auto-extract...", e
                )

        # ── Priority 2: Auto-extract from Chrome via browser_cookie3 ───────
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
                    "__Secure-1PSID not found in Chrome cookies.\n"
                    "Fix options:\n"
                    "  A) Open Chrome, sign in at gemini.google.com, restart proxy.\n"
                    "  B) Paste cookies into config.yaml (most reliable):\n"
                    "       backends:\n"
                    "         gemini_web:\n"
                    "           type: gemini_web\n"
                    "           cookies:\n"
                    "             __Secure-1PSID: '<value>'\n"
                    "             __Secure-1PSIDTS: '<value>'\n"
                    "     Get values: Chrome DevTools (F12) → Application → "
                    "Cookies → https://gemini.google.com"
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


def _flatten_messages(messages: list[dict]) -> str:
    """Flatten OpenAI messages array into a single prompt string.

    Handles multi-part content (list of {type, text/image_url} dicts) that
    Continue.dev produces when @file or @folder context is injected.
    """
    parts = []
    for m in messages:
        role = m["role"].upper()
        content = m.get("content", "")
        if isinstance(content, list):
            # Multi-part content from @file / @folder context providers
            text_pieces = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_pieces.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        text_pieces.append("[image attachment]")
                else:
                    text_pieces.append(str(part))
            content = "\n".join(text_pieces)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


async def _handle_gemini_web(
    backend: BackendConfig, body: dict, stream: bool
) -> JSONResponse | StreamingResponse:
    """Chat via Gemini Web (browser session — generous quota, no API key needed).

    Supports @file, @folder, @diff, @terminal context injection via message flattening.
    Does NOT support function/tool calling.
    """
    global gemini_client
    messages = body.get("messages", [])
    model_str = body.get("model", "gemini-3.0-flash")
    full_prompt = _flatten_messages(messages)

    # Resolve model string → gemini_webapi.constants.Model enum
    from gemini_webapi.constants import Model as _Model
    enum_key = _GEMINI_WEB_MODEL_MAP.get(model_str, _GEMINI_WEB_DEFAULT_MODEL)
    model_enum = getattr(_Model, enum_key, _Model.UNSPECIFIED)
    logger.debug("Gemini Web: model=%s → enum=%s", model_str, enum_key)

    try:
        client = await _get_gemini_client(backend)
        response = await client.generate_content(full_prompt, model=model_enum)
        text = response.text or ""
    except Exception as e:
        gemini_client = None  # Reset so next request retries fresh
        logger.error("Gemini Web error: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": (
                    f"Gemini Web unavailable: {e}\n\n"
                    "To fix, choose one of:\n"
                    "1. Open Chrome and sign in at gemini.google.com, then restart the proxy.\n"
                    "2. Paste cookies into config.yaml under gemini_web.cookies\n"
                    "   (Chrome DevTools F12 → Application → Cookies → gemini.google.com)"
                ),
                "type": "gemini_web_error",
                "code": "gemini_web_unavailable",
            }},
        )

    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    ts = int(time.time())

    if not stream:
        return JSONResponse(content={
            "id": cid,
            "object": "chat.completion",
            "created": ts,
            "model": model_str,
            "choices": [{"index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"}],
        })

    # Streaming simulation: emit OpenAI SSE contract that Continue expects:
    #   1. Role announcement chunk
    #   2. Content chunks (word-batched for a natural feel)
    #   3. Stop chunk  4. [DONE] sentinel
    async def _stream() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n".encode()
        words = text.split(" ")
        for i in range(0, len(words), 8):
            piece = " ".join(words[i:i + 8])
            if i + 8 < len(words):
                piece += " "
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n".encode()
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': ts, 'model': model_str, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


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
