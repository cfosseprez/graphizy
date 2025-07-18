"""
Sphinx documentation improvements for BehaviorPy.

File: docs/conf.py
"""

# Configuration file for the Sphinx documentation builder.

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder

sys.path.insert(0, os.path.abspath('..'))

sys.path.insert(0, os.path.abspath('../graphizy/'))  # Adjust relative path to your package root

# -- Project information -----------------------------------------------------
project = 'Graphizy'
copyright = '2025, Charles Fosseprez'
author = 'Charles Fosseprez'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'myst_parser',  # For markdown support
    'nbsphinx',  # For Jupyter notebooks
    'sphinx_gallery.gen_gallery',  # For example galleries
]

# Add support for NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# AutoAPI settings
autosummary_generate = True
autosummary_imported_members = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'logo_only': False,
}

html_static_path = ['_static']
html_logo = '_static/logo.png'  # Add your logo
html_favicon = '_static/favicon.ico'  # Add your favicon

# -- Sphinx Gallery configuration --------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': r'\.py',
    'ignore_pattern': r'__init__\.py',
    'within_subsection_order': FileNameSortKey,
    'subsection_order': ExplicitOrder([
        '../examples/1_basic_usage',
        '../examples/2_graph_metrics',
        '../examples/3_advanced_memory',
        '../examples/4_video_analysis',
        # add more if needed
    ]),
}