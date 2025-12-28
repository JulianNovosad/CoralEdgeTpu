#!/usr/bin/env python3
import http.server
import socketserver
import threading
import time
import os

# Simple HTTP server to serve video stream
class VideoStreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--frame')
            self.end_headers()
            
            # This would normally connect to the camera feed
            # For now, we'll just send a placeholder
            while True:
                try:
                    time.sleep(0.1)  # Simulate frame rate
                except:
                    break
        else:
            self.send_response(404)
            self.end_headers()

def start_http_server():
    port = 8080
    with socketserver.TCPServer(("", port), VideoStreamHandler) as httpd:
        print(f"HTTP server running at http://0.0.0.0:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_http_server()