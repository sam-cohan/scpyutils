"""
Module containing shell utilities.

Author: Sam Cohan
"""

import os
import re
import shutil
import subprocess
from typing import List, Optional


def run_cmd(cmd_list: List[str], log_path: Optional[str] = None) -> int:
    """Run an arbitrary shell command and optionally store its output to a log file.

    Args:
        cmd_list: Arbitrary shell command. e.g. "rsync -av from_path to_path"
            would have to be passed as ["rsync", "-av", "from_path", "to_path"]
        log_path: Optionally provided path to a file to store the stdout.

    Returns:
        Integer return code from running the command.
    """
    process = subprocess.Popen(
        cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if log_path:
        dir_name = os.path.dirname(log_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    if log_path:
        with open(log_path, "w+") as f:
            for line in iter(process.stdout.readline, b""):
                line = line.decode()
                print(line)
                f.write(line)
                f.flush()
    else:
        for line in iter(process.stdout.readline, b""):
            line = line.decode()
            print(line)
    return process.poll()


def rsync(src: str, dest: str) -> int:
    """Wrapper for rsync -av command."""
    return run_cmd(["rsync", "-av", src, dest])


def remove_dir_contents(directory: str, exclude_pattern: Optional[str] = None):
    """Remove contents of a directory without deleting the directory itself.

    Args:
        directory: Path of directory to remove contents from.
        exclude_pattern: Regular expression of files not to remove.
    """
    print("removing contents of " + directory + "  ...", flush=True)
    for filename in os.listdir(directory):
        if exclude_pattern:
            if re.search(exclude_pattern, filename):
                continue
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
