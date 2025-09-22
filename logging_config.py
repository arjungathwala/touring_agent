import logging
import sys

def setup_demo_logging():
    """Configure logging for clean demo output"""
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create console handler with custom format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Custom formatter for clean output
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Clear existing handlers and add our custom one
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("websockets.protocol").setLevel(logging.WARNING)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Keep important loggers at INFO level
    logging.getLogger("app").setLevel(logging.INFO)
    logging.getLogger("app.services").setLevel(logging.INFO)
    logging.getLogger("app.logging.flight_recorder").setLevel(logging.INFO)
    
    print("🔧 Demo logging configured - WebSocket DEBUG logs silenced")

if __name__ == "__main__":
    setup_demo_logging()
