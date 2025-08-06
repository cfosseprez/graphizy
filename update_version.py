#!/usr/bin/env python3
"""
Version synchronization script for Graphizy
Automatically updates CITATION.cff and other files to match pyproject.toml version
"""

import re
import toml
from pathlib import Path
from datetime import datetime
import yaml


def get_version_from_pyproject():
    """Extract version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    with open(pyproject_path, 'r') as f:
        data = toml.load(f)

    return data['project']['version']


def update_citation_cff(version):
    """Update CITATION.cff with new version and current date"""
    citation_path = Path("CITATION.cff")
    if not citation_path.exists():
        print("CITATION.cff not found, skipping...")
        return

    try:
        with open(citation_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Update the values
        data['version'] = version
        data['date-released'] = datetime.now().strftime('%Y-%m-%d')

        with open(citation_path, 'w', encoding='utf-8') as f:
            # Dump back to YAML, preserving order and style where possible
            yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        print(f"Updated CITATION.cff: version -> {version}, date -> {data['date-released']}")
    except Exception as e:
        print(f"Could not update CITATION.cff with yaml. Error: {e}")


def update_init_version(version):
    """Update __init__.py version if it exists"""
    init_path = Path("src/graphizy/__init__.py")
    if not init_path.exists():
        print("__init__.py not found, skipping...")
        return

    with open(init_path, 'r') as f:
        content = f.read()

    # Add or update __version__ if it doesn't exist
    if '__version__' in content:
        content = re.sub(r'__version__ = ["\'].*["\']', f'__version__ = "{version}"', content)
    else:
        # Add version after other __variables__
        insert_point = content.find('__all__')
        if insert_point == -1:
            # Add at the end of the file before __all__ or at the very end
            content = content.rstrip() + f'\n\n__version__ = "{version}"\n'
        else:
            content = content[:insert_point] + f'__version__ = "{version}"\n\n' + content[insert_point:]

    with open(init_path, 'w') as f:
        f.write(content)

    print(f"Updated __init__.py: __version__ -> {version}")


def main():
    """Main synchronization function"""
    try:
        # Get version from pyproject.toml
        version = get_version_from_pyproject()
        print(f"Found version {version} in pyproject.toml")

        # Update other files
        update_citation_cff(version)
        update_init_version(version)

        print("✅ Version synchronization complete!")

    except Exception as e:
        print(f"❌ Error during synchronization: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())