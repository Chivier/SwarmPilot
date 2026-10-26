"""A toy "private" inference server: ranks documents against a query.

It stands in for an in-house model server that SwarmPilot knows nothing
about. Runtime grows with the number of documents and the query length, so
the predictor has something to learn.

Routes:
    POST /v1/rank   {"query": str, "documents": [str, ...]}
    GET  /health
    GET  /info      {"build": {"version": "1.4.2"}}

Usage:
    python -m toy_ranker.server --port 9100 --model toy-ranker-small
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "1.4.2"
MS_PER_DOC = 2.0
MS_PER_QUERY_CHAR = 0.05


def rank(query: str, documents: list[str]) -> list[dict]:
    """Score each document by words it shares with the query."""
    words = set(query.lower().split())
    scored = [
        {"index": i, "score": len(words & set(d.lower().split()))}
        for i, d in enumerate(documents)
    ]
    return sorted(scored, key=lambda s: -s["score"])


def make_handler(model: str) -> type[BaseHTTPRequestHandler]:
    """Build a request handler that serves ``model``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict) -> None:
            out = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/info":
                self._send(200, {"model": model, "build": {"version": VERSION}})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/rank":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("content-length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                query = str(body.get("query", ""))
                documents = [str(d) for d in body.get("documents", [])]
            except (ValueError, TypeError, AttributeError):
                self._send(400, {"error": "bad request"})
                return
            time.sleep(
                (MS_PER_DOC * len(documents) + MS_PER_QUERY_CHAR * len(query))
                / 1000
            )
            self._send(200, {"model": model, "results": rank(query, documents)})

        def log_message(self, *args: object) -> None:
            pass

    return Handler


def main() -> None:
    """Serve until interrupted."""
    parser = argparse.ArgumentParser(description="Toy ranking server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="toy-ranker-small")
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(args.model)
    )
    print(f"toy-ranker {VERSION} serving {args.model} on :{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
