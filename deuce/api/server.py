"""Run the HTTP API with Uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("DEUCE_API_HOST", "127.0.0.1")
    port = int(os.getenv("DEUCE_API_PORT", "8000"))
    uvicorn.run(
        "deuce.api.app:app",
        host=host,
        port=port,
        reload=os.getenv("DEUCE_API_RELOAD", "").strip() in ("1", "true", "yes"),
    )


if __name__ == "__main__":
    main()
