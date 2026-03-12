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
        assert len(c.backends) == 9, f"Expected 9 backends, got {len(c.backends)}"
        assert len(c.aliases) == 14, f"Expected 14 aliases, got {len(c.aliases)}"
        assert c.host == "127.0.0.1", f"Unexpected host: {c.host}"
        assert c.port == 8000, f"Unexpected port: {c.port}"

        # Check all expected backends exist
        expected_backends = [
            "smart_router", "ollama", "claude_smart",
            "copilot_opus", "copilot_gpt", "copilot_gpt_mini",
            "copilot_raptor_mini", "gemini_web", "gemini_api",
        ]
        for b in expected_backends:
            assert b in c.backends, f"Missing backend: {b}"

        # Check backend types
        assert c.backends["ollama"].type == "openai_compatible"
        assert c.backends["claude_smart"].type == "cli"
        assert c.backends["copilot_opus"].type == "cli"
        assert c.backends["gemini_web"].type == "gemini_web"
        assert c.backends["gemini_api"].type == "gemini"

        # Check all aliases exist
        expected_aliases = [
            "continue-pro", "copilot-opus", "copilot-gpt",
            "copilot-gpt-mini", "copilot-raptor-mini", "claude-smart",
            "super-coder", "super-coder-fast", "super-coder-lite",
            "gemini-flash-web", "gemini-pro-web", "gemini-thinking-web",
            "gemini-flash-api", "gemini-pro-api",
        ]
        for a in expected_aliases:
            assert a in c.aliases, f"Missing alias: {a}"

        record(1, "Config Integrity", True, "9 backends, 14 aliases, all present")
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
        ("continue-pro", "smart_router", "continue-pro"),
        ("copilot-opus", "copilot_opus", "copilot-opus"),
        ("claude-smart", "claude_smart", "claude-smart"),
        ("super-coder", "ollama", "qwen3-coder"),
        ("super-coder-fast", "ollama", "qwen2.5-coder"),
        ("super-coder-lite", "ollama", "codellama:7b"),
        ("gemini-flash-web", "gemini_web", "gemini-3.0-flash"),
        ("gemini-pro-web", "gemini_web", "gemini-3.1-pro"),
        ("gemini-thinking-web", "gemini_web", "gemini-3.0-flash-thinking"),
        ("gemini-flash-api", "gemini_api", "gemini-2.0-flash"),
        ("gemini-pro-api", "gemini_api", "gemini-2.5-pro-preview-05-06"),
        # Fallback
        ("unknown-xyz", "ollama", "unknown-xyz"),
        # Explicit backend/model
        ("ollama/llama3.2", "ollama", "llama3.2"),
        # Heuristic routing
        ("claude-pro", "claude_smart", "claude-pro"),
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
           f"14 tests, {'all passed' if all_ok else 'SOME FAILED'}")


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
        from proxy.server import _GEMINI_WEB_MODEL_MAP, _GEMINI_WEB_DEFAULT_MODEL

        assert app.title == "Super Coder", f"Unexpected app title: {app.title}"

        # Check model map
        assert "gemini-3.0-flash" in _GEMINI_WEB_MODEL_MAP
        assert "gemini-3.1-pro" in _GEMINI_WEB_MODEL_MAP
        assert "gemini-3.0-flash-thinking" in _GEMINI_WEB_MODEL_MAP
        assert _GEMINI_WEB_DEFAULT_MODEL == "G_3_0_FLASH"

        print("  [OK]   App created, model map valid")
        record(4, "Server Module Import", True, "App, model map, helpers all imported")
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
    result1 = _flatten_messages(msgs1)
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
    result2 = _flatten_messages(msgs2)
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
    result3 = _flatten_messages([])
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
            assert len(data["backends"]) == 9, f"Backends: {data['backends']}"
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
if __name__ == "__main__":
    print("=" * 60)
    print("Super Coder Proxy — 10-Pass Validation Suite")
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
