#!/usr/bin/env python3
"""web/ 개발 서버.  사용:  python scripts/serve_web.py [포트]

`python -m http.server` 를 그대로 쓰면 안 되는 이유: Windows 의 mimetypes 가
레지스트리를 읽는데, 거기에 .js=text/plain 으로 박혀 있는 경우가 흔하다.
그러면 브라우저가 ES 모듈 로딩을 MIME 검사로 거부해 앱이 아예 안 뜬다
(실측: "Expected a JavaScript-or-Wasm module script but ... text/plain").
여기서는 확장자 -> MIME 을 직접 고정한다. 배포(Vercel)는 알아서 맞게 준다.
"""
from __future__ import annotations

import http.server
import os
import sys
from functools import partial
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
    ".json": "application/json",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = MIME

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # 모델 교체가 즉시 보이게
        # 교차출처 격리 — 이게 있어야 WASM 멀티스레드(SharedArrayBuffer)가 켜진다.
        # 배포(Vercel)는 web/vercel.json 이 같은 헤더를 준다. 둘을 같이 고칠 것.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8377))
    print(f"http://localhost:{port}  ({WEB})")
    http.server.ThreadingHTTPServer(("", port), partial(Handler, directory=str(WEB))).serve_forever()
