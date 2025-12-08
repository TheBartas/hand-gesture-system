from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

class FakeCameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        print("==== Request received ====")
        print(f"Raw path : {self.path}")
        print(f"Parsed   : {parsed}")
        print(f"Queries  : {query}")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


server_address = ('127.0.0.1', 8080)
httpd = HTTPServer(server_address, FakeCameraHandler)
print("Fake camera running on http://127.0.0.1:8080")
httpd.serve_forever()
