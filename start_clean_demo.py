#!/usr/bin/env python3
"""
Clean demo startup with minimal logging
"""

import subprocess
import sys
import os

def start_clean_demo():
    """Start server with only essential conversation flow logs"""
    
    print("🎯 Starting Touring Agent with Clean Logging")
    print("📞 You'll see only the essential flow:")
    print("   🎤 STT: [transcript]")
    print("   🧠 THINKING: [transcript]") 
    print("   🔧 TOOLS: [tool_names]")
    print("   💬 RESPONSE: [agent_response]")
    print("   🗣️ TTS: [audio_generation]")
    print("   ✅ SENT")
    print("=" * 50)
    
    # Start uvicorn with minimal logging
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "error",  # Only errors
        "--no-access-log",       # No access logs
    ]
    
    try:
        subprocess.run(cmd, cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n🛑 Clean demo stopped")

if __name__ == "__main__":
    start_clean_demo()
