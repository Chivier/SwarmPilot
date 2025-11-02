#!/usr/bin/env python3
"""
Development convenience wrapper for the CLI.

For installed usage, use the 'sscheduler' command directly.
This file allows developers to run './cli.py' during development.
"""

from src.cli import app

if __name__ == "__main__":
    app()
