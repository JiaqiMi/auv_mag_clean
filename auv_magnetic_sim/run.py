import argparse
import webbrowser
from threading import Timer

from magclean_sim.webapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="AUV magnetic-cleanliness simulation UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = "http://%s:%d" % (args.host, args.port)
    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()
    create_app().run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
