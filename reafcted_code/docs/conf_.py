# docs/conf.py

import os
import sys
from pathlib import Path

# Add the project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Project information
project = 'Centralized NLP Package'
author = 'Santhosh Kumar'

# Sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# Template path
templates_path = ['_templates']

# Exclude patterns
exclude_patterns = []

# HTML theme
html_theme = 'sphinx_rtd_theme'

# Static files
html_static_path = ['_static']
