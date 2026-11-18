#!/usr/bin/env python3
"""CLI entrypoint for JobScout."""
from __future__ import annotations

import sys

from jobscout.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    sys.exit(main())
