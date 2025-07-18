#!/usr/bin/env python3
"""
AQG System API Server

This script starts the FastAPI server for the Automated Question Generation system.
It provides REST API endpoints for individual agent operations and full workflow execution.
"""

import asyncio
import sys
from pathlib import Path
import uvicorn

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.main import app


def main():
    """Main entry point for the API server."""
    print("🚀 Starting AQG System API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔧 Alternative docs at: http://localhost:8000/redoc")
    print("💡 Health check: http://localhost:8000/health")
    print("📝 Available videos: http://localhost:8000/videos")
    print()
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main() 