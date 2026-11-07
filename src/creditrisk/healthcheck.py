"""Container health check: exit 0 if the API on this machine reports ready, else 1 (used by the Dockerfile)."""
import os
import sys
import urllib.request


def main():
    url = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/ready"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            sys.exit(0 if response.status == 200 else 1)
    except OSError:
        sys.exit(1)


if __name__ == "__main__":
    main()
