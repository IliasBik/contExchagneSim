"""
view_log.py — открыть детальный журнал прогона во вьюере detail_view.html.

Страница статическая (чистый JS, без внешних библиотек), но со страницы,
открытой как file://, браузер не даёт читать локальные файлы, поэтому
скрипт поднимает локальный http-сервер в каталоге проекта и открывает

    http://127.0.0.1:PORT/detail_view.html?file=<журнал>

Запуск:
    python view_log.py                       # detail_log.jsonl из каталога проекта
    python view_log.py --log other.jsonl     # другой журнал (путь относительно проекта)
    python view_log.py --port 9000 --no-browser

Альтернатива без сервера: открыть detail_view.html двойным щелчком и выбрать
файл журнала кнопкой на странице.
"""

from __future__ import annotations

import argparse
import functools
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):      # не шуметь на каждый запрос
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--log", default="detail_log.jsonl",
                    help="файл журнала относительно каталога проекта")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true",
                    help="не открывать браузер автоматически")
    args = ap.parse_args()

    log_path = os.path.join(HERE, args.log)
    if not os.path.exists(log_path):
        print(f"нет файла {log_path}; сначала запустите симуляцию "
              f"(python agent_simulation.py)")
    handler = functools.partial(_QuietHandler, directory=HERE)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = (f"http://127.0.0.1:{args.port}/detail_view.html"
           f"?file={quote(args.log)}")
    print(f"вьюер: {url}")
    print("Ctrl+C — остановить сервер")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
