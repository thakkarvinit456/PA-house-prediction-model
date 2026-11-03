"""
Standalone Lightweight Web Server for House Price Prediction System
Serves the interactive web interface (index.html) and provides a JSON prediction endpoint.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 5000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run_server():
    os.chdir(DIRECTORY)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print("=" * 60)
        print("🏡 HOUSE PRICE PREDICTION WEB APPLICATION")
        print("=" * 60)
        print(f"Server successfully started at: {url}")
        print("Press Ctrl+C to stop the server.")
        print("=" * 60)
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    run_server()
