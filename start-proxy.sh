#!/bin/bash
# Super Coder — Unified LLM Proxy
# Five sources: Copilot, Claude Pro, Local Ollama, Gemini Web, Gemini API

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Super Coder Proxy starting..."
echo "  Endpoint: http://127.0.0.1:8000/v1/chat/completions"
echo ""
echo "  1. copilot-opus      Copilot - Claude Opus (claude-opus-4.6)"
echo "  2. claude-smart      Claude Pro - Smart (auto-select)"
echo "  3. super-coder       Super Coder (Qwen3 30B local)"
echo "     super-coder-fast  Super Coder Fast (Qwen2.5 7B)"
echo "     super-coder-lite  Super Coder Lite (CodeLlama 7B)"
echo "  4. gemini-flash-web  Gemini 3.0 Flash (Web, browser cookies)"
echo "     gemini-pro-web    Gemini 3.1 Pro (Web, browser cookies)"
echo "     gemini-thinking-web  Gemini Flash Thinking (Web)"
echo "  5. gemini-flash-api  Gemini 2.0 Flash (API, GOOGLE_API_KEY)"
echo "     gemini-pro-api    Gemini 2.5 Pro (API, GOOGLE_API_KEY)"
echo ""

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "NOTE: GOOGLE_API_KEY is not set — Gemini API models won't work."
  echo "  Gemini Web models still work via browser cookies."
  echo "  For API: export GOOGLE_API_KEY=\"AIza...\""
  echo "  Get a free key at: https://aistudio.google.com/app/apikey"
  echo ""
fi

conda run -n llm-proxy python -m proxy.server
