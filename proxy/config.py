"""Configuration loader for the LLM proxy router."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BackendConfig:
    name: str
    type: str  # "openai_compatible", "anthropic", "cli", "gemini_web", or "gemini"
    base_url: str | None = None
    api_key: str | None = None
    cli_binary: str | None = None  # For type="cli": the binary to invoke
    cli_args: list[str] | None = None  # Extra CLI flags (e.g. ["--model", "opus"])
    cookies: dict[str, str] | None = None  # For type="gemini_web": explicit cookie values

    @classmethod
    def from_dict(cls, name: str, d: dict) -> BackendConfig:
        api_key = d.get("api_key")
        if not api_key and "api_key_env" in d:
            api_key = os.environ.get(d["api_key_env"], "")
        # Only keep non-empty cookie values
        raw_cookies = d.get("cookies") or {}
        cookies = {k: v for k, v in raw_cookies.items() if v and str(v).strip()} or None
        return cls(
            name=name,
            type=d["type"],
            base_url=d.get("base_url"),
            api_key=api_key or "",
            cli_binary=d.get("cli_binary"),
            cli_args=d.get("cli_args"),
            cookies=cookies,
        )


@dataclass
class AliasConfig:
    backend: str
    model: str


@dataclass
class ProxyConfig:
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    aliases: dict[str, AliasConfig] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    def resolve_model(self, model: str) -> tuple[BackendConfig, str]:
        """Resolve a model name/alias to (backend_config, actual_model_name).

        Resolution order:
        1. Exact alias match (e.g. "best" -> ollama/qwen3-coder)
        2. "backend/model" syntax (e.g. "groq/llama-3.3-70b-versatile")
        3. Auto-detect: try each backend's known models
        4. Fallback to ollama
        """
        # 1. Alias lookup
        if model in self.aliases:
            alias = self.aliases[model]
            backend = self.backends.get(alias.backend)
            if backend:
                return backend, alias.model

        # 2. Explicit backend/model syntax
        if "/" in model:
            backend_name, model_name = model.split("/", 1)
            backend = self.backends.get(backend_name)
            if backend:
                return backend, model_name

        # 3. Heuristic routing for the three Super Coder sources
        if model in ("claude", "claude-cli", "claude-pro"):
            if "claude_cli" in self.backends:
                return self.backends["claude_cli"], model
        if model in ("copilot", "copilot-cli"):
            if "copilot_cli" in self.backends:
                return self.backends["copilot_cli"], model

        # 4. Fallback to ollama (local)
        if "ollama" in self.backends:
            return self.backends["ollama"], model

        # Last resort: first available backend
        first = next(iter(self.backends.values()))
        return first, model


def load_config(path: str | Path | None = None) -> ProxyConfig:
    if path is None:
        candidates = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "config.yml",
            Path(__file__).parent.parent / "config.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
    if path is None:
        return ProxyConfig()

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = ProxyConfig()

    if "server" in raw:
        cfg.host = raw["server"].get("host", cfg.host)
        cfg.port = raw["server"].get("port", cfg.port)
        cfg.log_level = raw["server"].get("log_level", cfg.log_level)

    for name, bdict in raw.get("backends", {}).items():
        cfg.backends[name] = BackendConfig.from_dict(name, bdict)

    for name, adict in raw.get("aliases", {}).items():
        cfg.aliases[name] = AliasConfig(
            backend=adict["backend"],
            model=adict["model"],
        )

    return cfg
