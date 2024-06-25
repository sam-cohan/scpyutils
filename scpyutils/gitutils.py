"""
Utilities for interacting with git source control.

Author: Sam Cohan
"""

import subprocess


def get_git_hash() -> str:
    """Get hash of current git revision."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode()
