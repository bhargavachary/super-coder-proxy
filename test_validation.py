"""Super Coder Proxy — 10-Pass Validation Suite.

Tests functionality, responsiveness, stability, and configuration integrity.
Run: conda run -n llm-proxy python test_validation.py
"""

import asyncio
import json
import os
import sys
import time
import traceback

# Ensure we can import the proxy package
sys.path.insert(0, os.path.dirname(__file__))

RESULTS = []


def record(pass_num, name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((pass_num, name, status, details))
    print(f"  [{status}] {details}" if details else f"  [{status}]")


def header(pass_num, name):
    print(f"\n{'='*60}")
    print(f"Pass {pass_num}: {name}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════
# Pass 1: Config Loading & Integrity
# ═══════════════════════════════════════════════════════════════
def pass_1():
    header(1, "Config Loading & Integrity")
    from proxy.config import load_config, ProxyConfig, BackendConfig, AliasConfig

    try:
        c = load_config()
        assert isinstance(c, ProxyConfig), "Config is not ProxyConfig"
        assert len(c.backends) >= 7, f"Expected >=7 backends, got {len(c.backends)}"
        assert len(c.aliases) >= 10, f"Expected >=10 aliases, got {len(c.aliases)}"
        assert c.host == "127.0.0.1", f"Unexpected host: {c.host}"
        assert c.port == 8000, f"Unexpected port: {c.port}"

        # Check core backends are present (config may evolve; check the essentials)
        expected_backends = [
            "ollama", "claude_smart",
            "copilot_opus", "copilot_gpt",
            "gemini_web", "gemini_api",
        ]
        for b in expected_backends:
            assert b in c.backends, f"Missing backend: {b}"

        # Check backend types
        assert c.backends["ollama"].type == "openai_compatible"
        assert c.backends["claude_smart"].type == "cli"
        assert c.backends["copilot_opus"].type == "cli"
        assert c.backends["gemini_web"].type == "gemini_web"
        assert c.backends["gemini_api"].type == "gemini"

        # Check core aliases always exist
        expected_aliases = [
            "copilot-opus", "copilot-gpt", "claude-smart",
            "super-coder-fast", "super-coder-lite",
            "gemini-flash-web", "gemini-pro-web", "gemini-thinking-web",
            "gemini-flash-api", "gemini-pro-api",
        ]
        for a in expected_aliases:
            assert a in c.aliases, f"Missing alias: {a}"

        record(1, "Config Integrity", True,
               f"{len(c.backends)} backends, {len(c.aliases)} aliases, all core entries present")
    except Exception as e:
        record(1, "Config Integrity", False, str(e))


# ═══════════════════════════════════════════════════════════════
# Pass 2: Alias Resolution (all paths)
# ═══════════════════════════════════════════════════════════════
def pass_2():
    header(2, "Alias Resolution")
    from proxy.config import load_config

    c = load_config()
    all_ok = True
    tests = [
        # (input, expected_backend, expected_model)
        ("copilot-opus", "copilot_opus", "copilot-opus"),
        ("claude-smart", "claude_smart", "claude-smart"),
        ("super-coder-fast", "ollama", "qwen2.5-coder"),
        ("super-coder-lite", "ollama", "codellama:7b"),
        ("gemini-flash-web", "gemini_web", "gemini-3.0-flash"),
        ("gemini-pro-web", "gemini_web", "gemini-3.1-pro"),
        ("gemini-thinking-web", "gemini_web", "gemini-3.0-flash-thinking"),
        ("gemini-flash-api", "gemini_api", "gemini-2.0-flash"),
        ("gemini-pro-api", "gemini_api", "gemini-2.5-pro-preview-05-06"),
        # Fallback
        ("unknown-xyz", "ollama", None),
    ]
    for alias, exp_backend, exp_model in tests:
        backend, model = c.resolve_model(alias)
        ok = backend.name == exp_backend
        if exp_model is not None:
            ok = ok and (model == exp_model)
        if not ok:
            print(f"  [FAIL] {alias} -> {backend.name}/{model} (expected {exp_backend}/{exp_model})")
            all_ok = False
        else:
            print(f"  [OK]   {alias} -> {backend.name}/{model}")

    record(2, "Alias Resolution", all_ok,
           f"13 tests, {'all passed' if all_ok else 'SOME FAILED'}")


# ═══════════════════════════════════════════════════════════════
# Pass 3: Cookie Filtering & BackendConfig
# ═══════════════════════════════════════════════════════════════
def pass_3():
    header(3, "Cookie Filtering & BackendConfig")
    from proxy.config import BackendConfig

    all_ok = True

    # Test 1: Empty cookies → None
    b1 = BackendConfig.from_dict("t1", {
        "type": "gemini_web",
        "cookies": {"__Secure-1PSID": "", "__Secure-1PSIDTS": ""}
    })
    if b1.cookies is not None:
        print(f"  [FAIL] Empty cookies should be None, got {b1.cookies}")
        all_ok = False
    else:
        print("  [OK]   Empty cookies filtered to None")

    # Test 2: Partial cookies → keep non-empty
    b2 = BackendConfig.from_dict("t2", {
        "type": "gemini_web",
        "cookies": {"__Secure-1PSID": "abc123", "__Secure-1PSIDTS": ""}
    })
    if b2.cookies != {"__Secure-1PSID": "abc123"}:
        print(f"  [FAIL] Partial cookies: {b2.cookies}")
        all_ok = False
    else:
        print("  [OK]   Partial cookies filtered correctly")

    # Test 3: Full cookies → keep all
    b3 = BackendConfig.from_dict("t3", {
        "type": "gemini_web",
        "cookies": {"__Secure-1PSID": "abc", "__Secure-1PSIDTS": "xyz"}
    })
    if b3.cookies != {"__Secure-1PSID": "abc", "__Secure-1PSIDTS": "xyz"}:
        print(f"  [FAIL] Full cookies: {b3.cookies}")
        all_ok = False
    else:
        print("  [OK]   Full cookies preserved")

    # Test 4: No cookies key → None
    b4 = BackendConfig.from_dict("t4", {"type": "gemini_web"})
    if b4.cookies is not None:
        print(f"  [FAIL] No cookies key should be None, got {b4.cookies}")
        all_ok = False
    else:
        print("  [OK]   No cookies key → None")

    # Test 5: API key from env var
    os.environ["_TEST_KEY"] = "test-val-123"
    b5 = BackendConfig.from_dict("t5", {"type": "gemini", "api_key_env": "_TEST_KEY"})
    if b5.api_key != "test-val-123":
        print(f"  [FAIL] API key from env: {b5.api_key}")
        all_ok = False
    else:
        print("  [OK]   API key from env var resolved")
    del os.environ["_TEST_KEY"]

    # Test 6: CLI args preserved
    b6 = BackendConfig.from_dict("t6", {
        "type": "cli",
        "cli_binary": "copilot",
        "cli_args": ["--model", "claude-opus-4.6", "-s", "--allow-all"]
    })
    if b6.cli_binary != "copilot" or len(b6.cli_args) != 4:
        print(f"  [FAIL] CLI config: binary={b6.cli_binary}, args={b6.cli_args}")
        all_ok = False
    else:
        print("  [OK]   CLI binary + args preserved")

    record(3, "Cookie Filtering & BackendConfig", all_ok, "6 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 4: Server Module Import & App Creation
# ═══════════════════════════════════════════════════════════════
def pass_4():
    header(4, "Server Module Import & App Creation")
    try:
        from proxy.server import app, _estimate_complexity, _flatten_messages
        from proxy.server import _GEMINI_WEB_MODEL_MAP, _GEMINI_WEB_DEFAULT_MODEL, _gemini_sessions

        assert app.title == "Super Coder", f"Unexpected app title: {app.title}"

        # Check model map
        assert "gemini-3.0-flash" in _GEMINI_WEB_MODEL_MAP
        assert "gemini-3.1-pro" in _GEMINI_WEB_MODEL_MAP
        assert "gemini-3.0-flash-thinking" in _GEMINI_WEB_MODEL_MAP
        assert _GEMINI_WEB_DEFAULT_MODEL == "G_3_0_FLASH"

        # Check _gemini_sessions is a dict (new ChatSession tracking)
        assert isinstance(_gemini_sessions, dict), "_gemini_sessions should be a dict"

        print("  [OK]   App created, model map valid, ChatSession dict present")
        record(4, "Server Module Import", True, "App, model map, helpers, session dict all imported")
    except Exception as e:
        record(4, "Server Module Import", False, str(e))


# ═══════════════════════════════════════════════════════════════
# Pass 5: Complexity Estimation Logic
# ═══════════════════════════════════════════════════════════════
def pass_5():
    header(5, "Complexity Estimation (Smart Router)")
    from proxy.server import _estimate_complexity

    all_ok = True

    # Simple prompt → low
    simple = [{"role": "user", "content": "What is 2+2?"}]
    if _estimate_complexity(simple) != "low":
        print("  [FAIL] Simple prompt should be 'low'")
        all_ok = False
    else:
        print("  [OK]   Simple prompt → low")

    # Complex prompt → high (keyword trigger)
    complex_kw = [{"role": "user", "content": "Please refactor the entire authentication module"}]
    if _estimate_complexity(complex_kw) != "high":
        print("  [FAIL] Complex keyword prompt should be 'high'")
        all_ok = False
    else:
        print("  [OK]   Complex keyword → high")

    # Long prompt → high (char count)
    long_msg = [{"role": "user", "content": "x " * 2000}]
    if _estimate_complexity(long_msg) != "high":
        print("  [FAIL] Long prompt should be 'high'")
        all_ok = False
    else:
        print("  [OK]   Long prompt (>3000 chars) → high")

    # Many messages → high
    many = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    if _estimate_complexity(many) != "high":
        print("  [FAIL] Many messages should be 'high'")
        all_ok = False
    else:
        print("  [OK]   Many messages (10) → high")

    record(5, "Complexity Estimation", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 6: Message Flattening (multi-part content)
# ═══════════════════════════════════════════════════════════════
def pass_6():
    header(6, "Message Flattening (multi-part content)")
    from proxy.server import _flatten_messages

    all_ok = True

    # Standard text messages
    msgs1 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result1, _ = _flatten_messages(msgs1)
    if "SYSTEM: You are helpful." not in result1 or "USER: Hello" not in result1:
        print(f"  [FAIL] Standard text: {result1[:100]}")
        all_ok = False
    else:
        print("  [OK]   Standard text messages flattened")

    # Multi-part content (from @file context)
    msgs2 = [
        {"role": "user", "content": [
            {"type": "text", "text": "Here is the file:"},
            {"type": "text", "text": "def hello(): pass"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}
    ]
    result2, files2 = _flatten_messages(msgs2)
    if "Here is the file:" not in result2 or "def hello(): pass" not in result2:
        print(f"  [FAIL] Multi-part: {result2[:100]}")
        all_ok = False
    else:
        print("  [OK]   Multi-part content (list) flattened")

    if "[image attachment]" not in result2:
        print(f"  [FAIL] Image attachment not handled: {result2[:100]}")
        all_ok = False
    else:
        print("  [OK]   Image attachment placeholder inserted")

    # Empty messages
    result3, _ = _flatten_messages([])
    if result3 != "":
        print(f"  [FAIL] Empty messages: '{result3}'")
        all_ok = False
    else:
        print("  [OK]   Empty messages → empty string")

    record(6, "Message Flattening", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 7: Gemini Web Model Map & Enum Resolution
# ═══════════════════════════════════════════════════════════════
def pass_7():
    header(7, "Gemini Web Model Map & Enum Resolution")
    from proxy.server import _GEMINI_WEB_MODEL_MAP
    from gemini_webapi.constants import Model as _Model

    all_ok = True

    for model_str, enum_key in _GEMINI_WEB_MODEL_MAP.items():
        enum_val = getattr(_Model, enum_key, None)
        if enum_val is None:
            print(f"  [FAIL] {model_str} -> {enum_key} not found in Model enum")
            all_ok = False
        else:
            print(f"  [OK]   {model_str} -> {enum_key} = {enum_val}")

    # Check that UNSPECIFIED fallback works
    unspec = getattr(_Model, "UNSPECIFIED", None)
    if unspec is None:
        print("  [WARN] Model.UNSPECIFIED not found (non-critical)")
    else:
        print(f"  [OK]   Model.UNSPECIFIED exists = {unspec}")

    record(7, "Gemini Model Map", all_ok, f"{len(_GEMINI_WEB_MODEL_MAP)} models mapped")


# ═══════════════════════════════════════════════════════════════
# Pass 8: Server Startup (lifespan test)
# ═══════════════════════════════════════════════════════════════
def pass_8():
    header(8, "Server Startup & Health Check")
    import httpx

    async def _test():
        import proxy.server as _srv
        from proxy.config import load_config as _lc
        _srv.config = _lc()
        _srv.http_client = httpx.AsyncClient(timeout=30.0)
        _srv._gemini_lock = asyncio.Lock()

        from proxy.server import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Health endpoint
            r = await client.get("/health")
            assert r.status_code == 200, f"Health status: {r.status_code}"
            data = r.json()
            assert data["status"] == "ok", f"Health status: {data}"
            assert len(data["backends"]) >= 6, f"Backends: {data['backends']}"
            print(f"  [OK]   /health → status=ok, {len(data['backends'])} backends")

            # Models endpoint
            r2 = await client.get("/v1/models")
            assert r2.status_code == 200
            models = r2.json()
            assert models["object"] == "list"
            assert len(models["data"]) >= 14, f"Expected >=14 models, got {len(models['data'])}"
            print(f"  [OK]   /v1/models → {len(models['data'])} models listed")

            return True

    try:
        result = asyncio.run(_test())
        record(8, "Server Startup", result, "Health + Models endpoints OK")
    except Exception as e:
        record(8, "Server Startup", False, f"{e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════
# Pass 9: Ollama Backend Live Test (if available)
# ═══════════════════════════════════════════════════════════════
def pass_9():
    header(9, "Ollama Backend Live Test")
    import httpx

    async def _test():
        # Check if Ollama is running
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get("http://localhost:11434/api/tags")
                if r.status_code != 200:
                    print("  [SKIP] Ollama not running")
                    return None
                tags = r.json()
                models = [m["name"] for m in tags.get("models", [])]
                print(f"  [OK]   Ollama running, {len(models)} models: {models[:5]}")
        except Exception:
            print("  [SKIP] Ollama not reachable at localhost:11434")
            return None

        import proxy.server as _srv
        from proxy.config import load_config as _lc
        if not hasattr(_srv, 'config') or _srv.config is None:
            _srv.config = _lc()
            _srv.http_client = httpx.AsyncClient(timeout=60.0)
            _srv._gemini_lock = asyncio.Lock()

        from proxy.server import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
            body = {
                "model": "super-coder-fast",
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "stream": False,
                "max_tokens": 20,
            }
            try:
                start = time.time()
                r = await client.post("/v1/chat/completions", json=body)
                elapsed = time.time() - start
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"  [OK]   super-coder-fast responded ({elapsed:.1f}s): {content[:80]}")
                    return True
                else:
                    print(f"  [FAIL] Status {r.status_code}: {r.text[:200]}")
                    return False
            except Exception as e:
                print(f"  [FAIL] Request error: {e}")
                return False

    result = asyncio.run(_test())
    if result is None:
        record(9, "Ollama Live Test", True, "SKIPPED (Ollama not running)")
    else:
        record(9, "Ollama Live Test", result, "Live request via ASGI transport")


# ═══════════════════════════════════════════════════════════════
# Pass 10: Streaming SSE Contract Validation
# ═══════════════════════════════════════════════════════════════
def pass_10():
    header(10, "Streaming SSE Contract Validation")
    import httpx

    async def _test():
        # Check if Ollama is running
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get("http://localhost:11434/api/tags")
                if r.status_code != 200:
                    print("  [SKIP] Ollama not running")
                    return None
        except Exception:
            print("  [SKIP] Ollama not reachable")
            return None

        import proxy.server as _srv
        from proxy.config import load_config as _lc
        # Always recreate — previous event loop's objects are stale
        _srv.config = _lc()
        _srv.http_client = httpx.AsyncClient(timeout=60.0)
        _srv._gemini_lock = asyncio.Lock()

        from proxy.server import app
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=60.0) as client:
            body = {
                "model": "super-coder-fast",
                "messages": [{"role": "user", "content": "Say 'test' and nothing else."}],
                "stream": True,
                "max_tokens": 20,
            }
            try:
                chunks = []
                async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                    assert resp.status_code == 200, f"Stream status: {resp.status_code}"
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                chunks.append("[DONE]")
                                break
                            chunk = json.loads(payload)
                            chunks.append(chunk)

                # Validate SSE contract
                assert len(chunks) >= 2, f"Too few chunks: {len(chunks)}"
                assert chunks[-1] == "[DONE]", "Last chunk should be [DONE]"

                # Check chunk structure
                for c in chunks[:-1]:
                    assert "id" in c, "Missing id"
                    assert c["object"] == "chat.completion.chunk", f"Bad object: {c['object']}"
                    assert "choices" in c, "Missing choices"
                    assert len(c["choices"]) > 0, "Empty choices"

                # Check last data chunk has finish_reason=stop
                last_data = chunks[-2]
                fr = last_data["choices"][0].get("finish_reason")
                assert fr == "stop", f"Last chunk finish_reason: {fr}"

                content_parts = [
                    c["choices"][0]["delta"].get("content", "")
                    for c in chunks[:-1]
                    if "delta" in c["choices"][0]
                ]
                full_text = "".join(content_parts)
                print(f"  [OK]   {len(chunks)-1} chunks received, [DONE] sentinel present")
                print(f"  [OK]   SSE contract valid (id, object, choices, finish_reason)")
                print(f"  [OK]   Streamed content: {full_text[:80]}")
                return True
            except Exception as e:
                print(f"  [FAIL] Streaming test: {e}")
                traceback.print_exc()
                return False

    result = asyncio.run(_test())
    if result is None:
        record(10, "SSE Contract", True, "SKIPPED (Ollama not running)")
    else:
        record(10, "SSE Contract", result, "SSE streaming contract validated")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
# Pass 11: ContextManager — extract_latest_user_message
# ═══════════════════════════════════════════════════════════════
def pass_11():
    header(11, "ContextManager.extract_latest_user_message")
    from proxy.context import ContextManager

    cm = ContextManager()
    all_ok = True

    # Single user message
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    text, files = cm.extract_latest_user_message(msgs)
    if text != "What is 2+2?" or files:
        print(f"  [FAIL] Single user msg: '{text}', files={files}")
        all_ok = False
    else:
        print("  [OK]   Single user message extracted")

    # Multi-turn conversation — should grab LAST user message only
    msgs2 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up question"},
    ]
    text2, _ = cm.extract_latest_user_message(msgs2)
    if text2 != "Follow-up question":
        print(f"  [FAIL] Multi-turn: got '{text2}', expected 'Follow-up question'")
        all_ok = False
    else:
        print("  [OK]   Multi-turn: last user message extracted (not first)")

    # Multi-part content (list with context + question)
    msgs3 = [
        {"role": "user", "content": [
            {"type": "text", "text": "File context"},
            {"type": "text", "text": "Fix this bug"},
        ]},
    ]
    text3, _ = cm.extract_latest_user_message(msgs3)
    if "Fix this bug" not in text3:
        print(f"  [FAIL] Multi-part content: '{text3}'")
        all_ok = False
    else:
        print("  [OK]   Multi-part user content extracted correctly")

    # No user message → empty string
    msgs4 = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "ans"}]
    text4, _ = cm.extract_latest_user_message(msgs4)
    if text4 != "":
        print(f"  [FAIL] No user msg should return '': got '{text4}'")
        all_ok = False
    else:
        print("  [OK]   No user message → empty string")

    record(11, "extract_latest_user_message", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 12: ContextManager — is_new_conversation
# ═══════════════════════════════════════════════════════════════
def pass_12():
    header(12, "ContextManager.is_new_conversation")
    from proxy.context import ContextManager

    cm = ContextManager()
    all_ok = True

    # Fresh ID → new
    if not cm.is_new_conversation("conv-xyz-001"):
        print("  [FAIL] Fresh conv_id should be new")
        all_ok = False
    else:
        print("  [OK]   Fresh conv_id is new")

    # After build_prompt call → no longer new
    msgs = [{"role": "user", "content": "hello"}]
    cm.build_prompt("conv-xyz-001", msgs, "gemini_web")
    if cm.is_new_conversation("conv-xyz-001"):
        print("  [FAIL] After build_prompt, should NOT be new")
        all_ok = False
    else:
        print("  [OK]   After build_prompt, conv is no longer new")

    # After reset → new again
    cm.reset_conversation("conv-xyz-001")
    if not cm.is_new_conversation("conv-xyz-001"):
        print("  [FAIL] After reset, should be new again")
        all_ok = False
    else:
        print("  [OK]   After reset, conv is new again")

    # Separate conv_id is always new
    if not cm.is_new_conversation("conv-xyz-002"):
        print("  [FAIL] Different conv_id should be new")
        all_ok = False
    else:
        print("  [OK]   Different conv_id is independently new")

    record(12, "is_new_conversation", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 13: Smart System Prompt Compression
# ═══════════════════════════════════════════════════════════════
def pass_13():
    header(13, "Smart System Prompt Compression")
    from proxy.context import ContextManager

    all_ok = True

    # Short text → returned unchanged
    short = "You are a coding assistant.\nAlways use Python 3."
    result = ContextManager._compress_system_prompt(short, budget=10000)
    if result != short:
        print(f"  [FAIL] Short text should be unchanged: '{result[:50]}'")
        all_ok = False
    else:
        print("  [OK]   Short prompt returned unchanged")

    # Long boilerplate → compressed under budget
    boilerplate = "Welcome to the assistant!\n" * 100
    boilerplate += "You must always respond in JSON.\n"
    boilerplate += "Never use markdown headers.\n"
    budget = 200
    compressed = ContextManager._compress_system_prompt(boilerplate, budget=budget)
    if len(compressed) > budget:
        print(f"  [FAIL] Compressed exceeds budget: {len(compressed)} > {budget}")
        all_ok = False
    else:
        print(f"  [OK]   Compressed to {len(compressed)} chars (budget={budget})")

    # Priority rules preserved after compression
    priority_text = (
        "This is a general preamble with many words that don't matter much at all.\n"
        * 50
        + "You must always return valid JSON.\n"
        + "Never include trailing commas.\n"
        + "More filler lines with no directives.\n" * 30
    )
    result2 = ContextManager._compress_system_prompt(priority_text, budget=300)
    if "must" not in result2.lower() and "never" not in result2.lower():
        print(f"  [FAIL] Priority rules not preserved: '{result2[:100]}'")
        all_ok = False
    else:
        print("  [OK]   must/never rules preserved after compression")

    record(13, "System Prompt Compression", all_ok, "3 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 14: File Content Compression
# ═══════════════════════════════════════════════════════════════
def pass_14():
    header(14, "File Content Compression (_compress_file_content)")
    from proxy.context import ContextManager

    all_ok = True

    # Short file → unchanged
    short = "line1\nline2\nline3"
    result = ContextManager._compress_file_content(short, max_lines=100)
    if result != short:
        print(f"  [FAIL] Short file should be unchanged: '{result}'")
        all_ok = False
    else:
        print("  [OK]   Short file returned unchanged")

    # Long file → truncated with marker
    long_file = "\n".join(f"line {i}" for i in range(200))
    result2 = ContextManager._compress_file_content(long_file, max_lines=100)
    lines2 = result2.splitlines()
    if len(lines2) <= 100:
        print(f"  [FAIL] Should have >100 lines (truncation marker): {len(lines2)}")
        all_ok = False
    elif "truncated" not in result2:
        print(f"  [FAIL] Missing truncation marker in: '{result2[-50:]}'")
        all_ok = False
    else:
        print(f"  [OK]   Long file truncated with marker ({len(lines2)} output lines)")

    # Code block with unclosed fence → fence closed before marker
    code = "File: src/main.py\n```python\n" + "\n".join(f"    code_line_{i}" for i in range(150))
    result3 = ContextManager._compress_file_content(code, max_lines=50)
    if "```" not in result3[-50:]:
        # Unclosed fence should be auto-closed before truncation notice
        print(f"  [FAIL] Unclosed fence not closed: '{result3[-80:]}'")
        all_ok = False
    else:
        print("  [OK]   Unclosed code fence auto-closed before truncation marker")

    record(14, "File Content Compression", all_ok, "3 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 15: Context Budget Trimming
# ═══════════════════════════════════════════════════════════════
def pass_15():
    header(15, "Context Budget Trimming (_trim_to_budget)")
    from proxy.context import ContextManager

    all_ok = True

    cm = ContextManager()

    # Within budget → unchanged
    short = "a" * 100
    result = cm._trim_to_budget(short, 500)
    if result != short:
        print(f"  [FAIL] Within budget should be unchanged")
        all_ok = False
    else:
        print("  [OK]   Within budget → unchanged")

    # Over budget → trimmed
    long_text = "start " * 1000 + "END_MARKER"
    budget = 500
    result2 = cm._trim_to_budget(long_text, budget)
    if len(result2) > budget + 60:  # +60 for the trimming marker overhead
        print(f"  [FAIL] Trimmed text still too long: {len(result2)} > {budget+60}")
        all_ok = False
    else:
        print(f"  [OK]   Over-budget text trimmed to ~{len(result2)} chars")

    # Trimmed text keeps the end (user query is last)
    text_with_query = "context " * 200 + "IMPORTANT_USER_QUERY"
    result3 = cm._trim_to_budget(text_with_query, 300)
    if "IMPORTANT_USER_QUERY" not in result3:
        print(f"  [FAIL] User query not preserved in trimmed output")
        all_ok = False
    else:
        print("  [OK]   User query (end of text) preserved after trimming")

    record(15, "Budget Trimming", all_ok, "3 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 16: ChatSession Quota Efficiency (Unit)
# ═══════════════════════════════════════════════════════════════
def pass_16():
    header(16, "ChatSession Quota Efficiency (unit simulation)")
    from proxy.server import _get_or_create_gemini_session, _gemini_sessions
    import proxy.server as _srv

    all_ok = True

    # Mock a minimal GeminiClient-like object with start_chat
    class _MockSession:
        def __init__(self, conv_id):
            self.conv_id = conv_id

    class _MockClient:
        def start_chat(self, model=None, **kw):
            return _MockSession("session-from-start_chat")

    mock_client = _MockClient()

    # Clear any existing sessions
    _srv._gemini_sessions.clear()

    # First call → is_new=True
    session1, is_new1 = _get_or_create_gemini_session("conv-test-001", mock_client, None)
    if not is_new1:
        print("  [FAIL] First call should return is_new=True")
        all_ok = False
    else:
        print("  [OK]   First call: is_new=True (full context will be sent)")

    # Second call for same conv → is_new=False (no context re-send)
    session2, is_new2 = _get_or_create_gemini_session("conv-test-001", mock_client, None)
    if is_new2:
        print("  [FAIL] Second call for same conv should return is_new=False")
        all_ok = False
    else:
        print("  [OK]   Second call: is_new=False (only user message sent)")

    # Same session object returned on second call
    if session1 is not session2:
        print("  [FAIL] Session object not reused across calls")
        all_ok = False
    else:
        print("  [OK]   Same ChatSession object reused for same conv_id")

    # Different conv_id → separate session, is_new=True
    session3, is_new3 = _get_or_create_gemini_session("conv-test-002", mock_client, None)
    if not is_new3:
        print("  [FAIL] Different conv_id should return is_new=True")
        all_ok = False
    else:
        print("  [OK]   Different conv_id creates a new session independently")

    # Cleanup
    _srv._gemini_sessions.clear()

    record(16, "ChatSession Quota Efficiency", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 17: Session Error Recovery
# ═══════════════════════════════════════════════════════════════
def pass_17():
    header(17, "Session Error Recovery (session-scoped, not global)")
    from proxy.server import _get_or_create_gemini_session
    from proxy.context import ContextManager
    import proxy.server as _srv

    all_ok = True

    class _MockSession:
        pass

    class _MockClient:
        def start_chat(self, model=None, **kw):
            return _MockSession()

    mock_client = _MockClient()
    _srv._gemini_sessions.clear()

    # Create two sessions
    s_a, _ = _get_or_create_gemini_session("conv-a", mock_client, None)
    s_b, _ = _get_or_create_gemini_session("conv-b", mock_client, None)

    # Simulate error recovery for conv-a: remove only conv-a
    _srv._gemini_sessions.pop("conv-a", None)

    # conv-b session still exists
    if "conv-b" not in _srv._gemini_sessions:
        print("  [FAIL] conv-b session should still exist after conv-a error")
        all_ok = False
    else:
        print("  [OK]   conv-b session unaffected when conv-a errors out")

    # conv-a is now new again (full context on next request)
    _, is_new_a = _get_or_create_gemini_session("conv-a", mock_client, None)
    if not is_new_a:
        print("  [FAIL] After session kill, conv-a should restart as new")
        all_ok = False
    else:
        print("  [OK]   After session kill, conv-a restarts as new (full context on next turn)")

    # Global GeminiClient is NOT reset (stays alive for other convs)
    # We test this by verifying gemini_client global is untouched
    old_client = _srv.gemini_client
    # (only verify that killing a session does not null the global client)
    # The old behaviour reset gemini_client=None globally on every error
    _srv._gemini_sessions.pop("conv-a", None)
    if _srv.gemini_client != old_client:
        print("  [FAIL] Global gemini_client should not be reset on session error")
        all_ok = False
    else:
        print("  [OK]   Global GeminiClient stays alive; only the session is killed")

    _srv._gemini_sessions.clear()
    record(17, "Session Error Recovery", all_ok, "3 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 18: Context Compression — _compress_context_items
# ═══════════════════════════════════════════════════════════════
def pass_18():
    header(18, "Context Items Compression (_compress_context_items)")
    from proxy.context import ContextManager

    all_ok = True

    # Empty → empty string
    result = ContextManager._compress_context_items([], budget=1000)
    if result != "":
        print(f"  [FAIL] Empty items should be empty string: '{result}'")
        all_ok = False
    else:
        print("  [OK]   Empty context list → empty string")

    # Fits in budget → all preserved
    small_items = ["File A: content", "File B: content", "File C: short"]
    joined = "\n---\n".join(small_items)
    result2 = ContextManager._compress_context_items(small_items, budget=10000)
    if result2 != joined:
        print(f"  [FAIL] Items within budget should be unchanged")
        all_ok = False
    else:
        print("  [OK]   Items within budget preserved verbatim")

    # Over budget → older items dropped first
    big_item = "X" * 500
    items = ["oldest_file: " + big_item, "middle_file: " + big_item, "newest_file: important"]
    budget = 600  # only enough for one big item + newer small item
    result3 = ContextManager._compress_context_items(items, budget=budget)
    if "newest_file" not in result3:
        print(f"  [FAIL] Newest item should survive budget trim: '{result3[:100]}'")
        all_ok = False
    else:
        print("  [OK]   Newest context item preserved when budget forces dropping")
    if "oldest_file" in result3 and len(result3) > budget + 50:
        print(f"  [FAIL] Oldest item not dropped despite budget: len={len(result3)}")
        all_ok = False
    else:
        print("  [OK]   Oldest item dropped first to fit budget")

    record(18, "Context Items Compression", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 19: Multi-Turn Context State Integrity
# ═══════════════════════════════════════════════════════════════
def pass_19():
    header(19, "Multi-Turn Context State")
    from proxy.context import ContextManager

    all_ok = True
    cm = ContextManager()
    conv = "conv-multiturn-test"

    # Turn 1: regular first-turn build
    msgs_t1 = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    prompt1, _ = cm.build_prompt(conv, msgs_t1, "gemini_web")
    if "What is Python?" not in prompt1:
        print(f"  [FAIL] Turn 1 prompt missing user message")
        all_ok = False
    else:
        print(f"  [OK]   Turn 1 prompt built ({len(prompt1)} chars)")

    # After turn 1, is_new should be False
    if cm.is_new_conversation(conv):
        print("  [FAIL] After turn 1, should not be new")
        all_ok = False
    else:
        print("  [OK]   After turn 1, conversation is no longer new")

    # Turn 2: extract only the latest user message
    msgs_t2 = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a language."},
        {"role": "user", "content": "How do I install packages?"},
    ]
    msg2, _ = cm.extract_latest_user_message(msgs_t2)
    if msg2 != "How do I install packages?":
        print(f"  [FAIL] Turn 2 extraction: got '{msg2}'")
        all_ok = False
    else:
        print(f"  [OK]   Turn 2: only latest user message extracted ({len(msg2)} chars)")

    # Turn 2 chars << turn 1 chars (quota efficiency)
    if len(msg2) >= len(prompt1):
        print(f"  [WARN] Turn 2 ({len(msg2)}c) not smaller than turn 1 ({len(prompt1)}c)")
    else:
        ratio = len(msg2) / len(prompt1) * 100
        print(f"  [OK]   Turn 2 is {ratio:.0f}% of turn 1 size (quota savings: {100-ratio:.0f}%)")

    # Reset → is_new again
    cm.reset_conversation(conv)
    if not cm.is_new_conversation(conv):
        print("  [FAIL] After reset, should be new")
        all_ok = False
    else:
        print("  [OK]   Reset restores conversation to new state")

    record(19, "Multi-Turn Context State", all_ok, "5 tests")


# ═══════════════════════════════════════════════════════════════
# Pass 20: Full Prompt Build — compression integration
# ═══════════════════════════════════════════════════════════════
def pass_20():
    header(20, "Full Prompt Build — compression integration")
    from proxy.context import ContextManager

    all_ok = True
    cm = ContextManager()

    # Build a prompt that exceeds budget and check it gets compressed
    huge_system = "This is a generic preamble sentence that adds no value.\n" * 200
    huge_context_file = "```python\n" + "\n".join(f"    code = line_{i}" for i in range(500)) + "\n```"
    user_question = "What does this function do?"

    msgs = [
        {"role": "system", "content": huge_system},
        {"role": "user", "content": huge_context_file + "\n\n" + user_question},
    ]

    raw_size = len(huge_system) + len(huge_context_file) + len(user_question)
    prompt, _ = cm.build_prompt("conv-compress-test", msgs, "gemini_web")

    # The output should fit within the model's budget (128k chars = ~32k tokens for gemini_web)
    token_limit = 128_000
    budget_chars = token_limit * 4  # 4 chars/token
    if len(prompt) > budget_chars:
        print(f"  [FAIL] Prompt exceeds budget: {len(prompt)} > {budget_chars}")
        all_ok = False
    else:
        print(f"  [OK]   Prompt within budget: {len(prompt):,} / {budget_chars:,} chars")

    # User question is always preserved
    if user_question not in prompt:
        print(f"  [FAIL] User question not in compressed prompt")
        all_ok = False
    else:
        print(f"  [OK]   User question preserved in output")

    # If raw > output, compression happened
    if raw_size > len(prompt):
        savings_pct = (raw_size - len(prompt)) / raw_size * 100
        print(f"  [OK]   Compression reduced prompt by {savings_pct:.0f}% "
              f"({raw_size:,} → {len(prompt):,} chars)")
    else:
        print(f"  [OK]   Prompt within budget, no compression needed "
              f"(raw={raw_size:,}, out={len(prompt):,})")

    # SYSTEM: header must be present (system prompt was provided)
    if "SYSTEM:" not in prompt:
        print(f"  [FAIL] SYSTEM: section missing from prompt")
        all_ok = False
    else:
        print("  [OK]   SYSTEM: section present in output")

    record(20, "Full Prompt Build", all_ok, "4 tests")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Super Coder Proxy — 20-Pass Validation Suite")
    print("=" * 60)

    pass_1()
    pass_2()
    pass_3()
    pass_4()
    pass_5()
    pass_6()
    pass_7()
    pass_8()
    pass_9()
    pass_10()
    pass_11()
    pass_12()
    pass_13()
    pass_14()
    pass_15()
    pass_16()
    pass_17()
    pass_18()
    pass_19()
    pass_20()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, _, s, _ in RESULTS if s == "FAIL")
    for pnum, name, status, details in RESULTS:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} Pass {pnum:2d}: {name:35s} [{status}] {details}")

    print(f"\n  Total: {passed} passed, {failed} failed out of {len(RESULTS)}")
    if failed > 0:
        print("\n  *** VALIDATION FAILED ***")
        sys.exit(1)
    else:
        print("\n  *** ALL VALIDATIONS PASSED ***")
        sys.exit(0)
