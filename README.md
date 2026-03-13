# Super Coder Proxy

Unified OpenAI-compatible LLM proxy that routes requests through a single `localhost:8000/v1` endpoint to multiple backends:

| # | Backend | Type | Auth |
|---|---------|------|------|
| 1 | **Copilot** (Claude Opus, GPT-5.3 Codex, etc.) | GitHub Copilot CLI | Copilot subscription |
| 2 | **Claude Pro** (smart auto-select) | Claude Code CLI | Claude Pro subscription |
| 3 | **Super Coder** (Qwen3 Coder, CodeLlama) | Local Ollama | None (offline) |
| 4 | **Gemini Web** (Flash, Pro, Thinking) | Browser cookies | Google account |
| 5 | **Gemini API** (Flash, Pro) | Google AI Studio | `GOOGLE_API_KEY` (free) |

Built for the [Continue.dev](https://github.com/continuedev/continue) VS Code extension but works with any OpenAI-compatible client.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/bhargavachary/super-coder-proxy.git
cd super-coder-proxy

# 2. Create conda environment
conda create -n llm-proxy python=3.11 -y
conda activate llm-proxy

# 3. Install
pip install -e .

# 4. Copy and edit config
cp config.yaml.example config.yaml
# Edit config.yaml — set cookies for Gemini Web, etc.

# 5. (Optional) Set Gemini API key
export GOOGLE_API_KEY="AIza..."

# 6. Start
./start-proxy.sh
# or: python -m proxy.server
```

## Configuration

Copy `config.yaml.example` to `config.yaml` and customize:

- **Ollama**: Runs locally, no config needed (just have Ollama running)
- **Copilot CLI**: Requires `copilot` CLI installed and authenticated (`copilot auth login`)
- **Claude CLI**: Requires `claude` CLI installed and authenticated
- **Gemini Web**: Paste browser cookies from Chrome DevTools (see config comments)
- **Gemini API**: Set `GOOGLE_API_KEY` env var ([get a free key](https://aistudio.google.com/app/apikey))

## Available Models

```
# Via proxy aliases:
copilot-opus          # Copilot → Claude Opus 4.6
copilot-gpt           # Copilot → GPT-5.3 Codex
copilot-gpt-mini      # Copilot → GPT-5 Mini
copilot-raptor-mini   # Copilot → GPT-5.1 Codex Mini
claude-smart          # Claude Code CLI (auto-selects best)
super-coder           # Ollama → Qwen3 Coder 30B
super-coder-fast      # Ollama → Qwen2.5 Coder 7B
super-coder-lite      # Ollama → CodeLlama 7B
gemini-flash-web      # Gemini Web → 3.0 Flash
gemini-pro-web        # Gemini Web → 3.1 Pro
gemini-thinking-web   # Gemini Web → Flash Thinking
gemini-flash-api      # Gemini API → 2.0 Flash
gemini-pro-api        # Gemini API → 2.5 Pro
```

## Continue.dev Integration

Add models to `~/.continue/config.yaml`:

```yaml
models:
  - name: Copilot - Claude Opus
    provider: openai
    model: copilot-opus
    apiBase: http://localhost:8000/v1
    apiKey: local

  - name: Qwen3 Coder 30B
    provider: ollama
    model: qwen3-coder
```

See the companion [Continue fork](https://github.com/bhargavachary/continue/tree/custom-supercoder) for a pre-configured build with offline mode toggle, disabled sign-in/indexing/docs UI, and all proxy models pre-configured.

## Offline Mode

The Continue fork includes an offline/online toggle in the model picker:
- **Offline (default)**: Only local Ollama models shown
- **Online**: All models (proxy + local) shown

First install always defaults to offline mode.

## Architecture

```
Continue.dev (VS Code) → http://localhost:8000/v1/chat/completions
                              ↓
                     Super Coder Proxy (FastAPI)
                         ↓           ↓           ↓
                    Ollama       CLI tools    Gemini
                   (local)    (claude/copilot) (web/api)
```

## Scripts

- `start-proxy.sh` — Start the proxy server
- `sync-continue.sh` — Sync Continue fork with upstream updates
- `rebuild-continue.sh` — Build and install Continue VSIX from source

## License

MIT
