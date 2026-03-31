# AGENTS.md — super-coder-proxy

Unified OpenAI-compatible LLM proxy that routes requests through `localhost:8000/v1` to multiple backends.
Built for [Continue.dev](https://github.com/continuedev/continue) but works with any OpenAI-compatible client.

---

## Backend Routing

| # | Backend | Type | Auth |
|---|---------|------|------|
| 1 | **Copilot** (Claude Opus, GPT-5.3 Codex, etc.) | GitHub Copilot CLI | Copilot subscription |
| 2 | **Claude Pro** (auto-select) | Claude Code CLI | Claude Pro subscription |
| 3 | **Super Coder** (Qwen3 Coder, CodeLlama) | Local Ollama | None (offline) |
| 4 | **Gemini Web** (Flash, Pro, Thinking) | Browser cookies | Google account |
| 5 | **Gemini API** (Flash, Pro) | Google AI Studio | `GOOGLE_API_KEY` (free) |

---

## Structure

```
proxy/              ← core proxy server and backend connectors
start-proxy.sh      ← startup script
config.yaml.example ← config template (copy to config.yaml, never commit filled version)
pyproject.toml      ← Python package definition
test_validation.py  ← validation tests
rebuild-continue.sh ← rebuilds Continue.dev integration
sync-continue.sh    ← syncs Continue.dev config
```

---

## Environment

```bash
conda create -n llm-proxy python=3.11 -y
conda activate llm-proxy
pip install -e .
cp config.yaml.example config.yaml   # edit with your cookies/keys
```

---

## Commands

```bash
conda activate llm-proxy
./start-proxy.sh            # start proxy at localhost:8000
# or: python -m proxy.server
python test_validation.py   # run validation tests
```

---

## Rules

- `config.yaml` is **git-ignored** — never commit it (contains cookies and API keys)
- Use `config.yaml.example` as the template — keep it clean and committed
- Each backend lives in its own module under `proxy/` — keep them decoupled
- OpenAI-compatible API contract must be preserved for all routes (`/v1/chat/completions`, `/v1/models`)
- Context compression must not silently drop tool-call or system message turns
- Tests must pass before committing: `python test_validation.py`
- Do **not** hardcode any credentials, tokens, or cookies in source

---

## Owner

**D K Bhargav Achary** — sole maintainer.
