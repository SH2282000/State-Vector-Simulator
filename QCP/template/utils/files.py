"""File utilities."""

import glob
import os


def get_all_files(folder_path):
    """Get all files in a folder."""
    return glob.glob(os.path.join(folder_path, "*"))
