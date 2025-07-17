"""
Just some utilities functions
"""
from pathlib import Path

def setup_output_directory():
    """Create output directory for saving images"""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir