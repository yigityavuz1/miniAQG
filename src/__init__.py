"""
Mini AQG (Automated Question Generation) System

A proof-of-concept automated question generation system for educational content,
specifically designed for processing 3Blue1Brown video transcripts.
"""

__version__ = "0.1.0"
__author__ = "AQG Development Team"

from .core.logger import get_logger, set_logger
from .core.prompt_loader import prompt_loader
from .core.llm_client import llm_client

__all__ = [
    "get_logger",
    "set_logger", 
    "prompt_loader",
    "llm_client"
] 