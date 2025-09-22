#!/usr/bin/env python3
"""
Demo startup script for clean logging
"""

import logging
import subprocess
import sys
import os

def setup_clean_logging():
    """Set up clean logging for demo"""
    # Silence WebSocket debug logs
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.protocol").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Keep important logs
    logging.getLogger("app").setLevel(logging.INFO)
    
    print("🎯 Demo logging configured")

def start_demo_server():
    """Start server with clean demo logging"""
    setup_clean_logging()
    
    print("🚀 Starting Touring Agent Demo Server...")
    print("🔇 WebSocket DEBUG logs silenced for clean demo output")
    print("📞 Ready for phone calls!")
    print("-" * 50)
    
    # Start uvicorn with clean output
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
        "--access-log",  # Keep access logs but clean format
    ]
    
    try:
        subprocess.run(cmd, cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n🛑 Demo server stopped")

if __name__ == "__main__":
    start_demo_server()
