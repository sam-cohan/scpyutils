"""
Project Directory Utilities.

Author: Sam Cohan
"""

import os
import sys
from typing import Optional

import yaml

import scpyutils.logutils as logu

LOGGER = logu.setup_logger(__name__)


CONFIG_DIR_NAME = ".config"
CONFIG_FILE_NAME = "config.yaml"


def get_repo_root(start_path: Optional[str] = None) -> str:
    """Find the root of a python repository.

    Walks up the directory tree to find the root of the project, identified as
    the first directory that does not contain an __init__.py file.

    Args:
        start_path: The starting path to begin the search. Defaults to the
            current working directory.

    Returns:
        The path to the repo root directory.
    """
    if start_path is None:
        start_path = os.getcwd()

    current_path = start_path

    while True:
        if not os.path.exists(os.path.join(current_path, "__init__.py")):
            return current_path

        # Move up one directory level
        parent_path = os.path.dirname(current_path)

        # If the parent directory is the same as the current directory, we are
        # at the root of the filesystem.
        if parent_path == current_path:
            LOGGER.warning("Repo root not found, using filesystem root!")
            return current_path

        current_path = parent_path


def add_repo_root_to_path(start_path: Optional[str] = None) -> str:
    """Find and add the repo root to sys.path.

    Args:
        start_path: The starting path to begin the search. Defaults to the
            current working directory.

    Returns:
        The path to the repo root directory.
    """
    repo_root = get_repo_root(start_path)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        LOGGER.info(f"Added {repo_root} to sys.path")
    else:
        LOGGER.info(f"{repo_root} is already in sys.path")
    return repo_root


def get_package_root(proj_name: str, start_path: Optional[str] = None) -> str:
    """Find root of a project given the project name.

    Finds the root of the project by searching for the first directory that
    matches the project name and contains an __init__.py file.

    Args:
        proj_name: The name of the project to search for.
        start_path: The starting path to begin the search. Defaults to the repo
            root.

    Returns:
        The path to the project root directory.

    Raises:
        FileNotFoundError: If the project root is not found.
    """
    if start_path is None:
        start_path = get_repo_root()

    for root, dirs, files in os.walk(start_path):
        # Remove hidden directories from the search
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # Check if the current directory matches the project name
        if os.path.basename(root) == proj_name and "__init__.py" in files:
            return root

    raise FileNotFoundError(
        f"Project '{proj_name}' with __init__.py not found from '{start_path}'"
    )


def get_config_file_path(
    proj_name: str,
    config_dir_name: str = CONFIG_DIR_NAME,
    config_file_name: str = CONFIG_FILE_NAME,
) -> str:
    """Get the config file path for the project.

    Args:
        proj_name: The name of the project.
        config_dir_name: The name of the config directory. (default to
            `CONFIG_DIR_NAME`)
        config_file_name: The name of the config file. (defaults to
            `CONFIG_FILE_NAME`)

    Returns:
        The path to the config directory.
    """
    return os.path.join(get_package_root(proj_name), config_dir_name, config_file_name)


def load_config(
    proj_name: str,
    config_dir_name: str = CONFIG_DIR_NAME,
    config_file_name: str = CONFIG_FILE_NAME,
) -> dict:
    """Load the project configuration file.

    Args:
        proj_name: The name of the project.
        config_dir_name: The name of the config directory. (default to
            `CONFIG_DIR_NAME`)
        config_file_name: The name of the config file. (defaults to
            `CONFIG_FILE_NAME`)

    Returns:
        The configuration dictionary.
    """
    config_file_path = get_config_file_path(
        proj_name=proj_name,
        config_dir_name=config_dir_name,
        config_file_name=config_file_name,
    )
    LOGGER.info(f"Loading config file from '{config_file_path}'")
    with open(config_file_path, "r") as file:
        return yaml.safe_load(file)
